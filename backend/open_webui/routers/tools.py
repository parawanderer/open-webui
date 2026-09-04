from __future__ import annotations

import asyncio
import logging
import re
import time
from functools import partial
from pathlib import Path
from typing import Optional

import aiohttp
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from open_webui.config import BYPASS_ADMIN_ACCESS_CONTROL, CACHE_DIR
from open_webui.constants import ERROR_MESSAGES
from open_webui.env import AIOHTTP_CLIENT_SESSION_SSL, AIOHTTP_CLIENT_TIMEOUT, ENABLE_PLUGINS
from open_webui.events import EVENTS, publish_event
from open_webui.internal.db import get_async_session
from open_webui.models.access_grants import AccessGrants
from open_webui.models.config import Config
from open_webui.models.groups import Groups
from open_webui.models.oauth_sessions import OAuthSessions
from open_webui.models.tools import (
    ToolAccessResponse,
    ToolForm,
    ToolModel,
    ToolResponse,
    Tools,
    ToolUserResponse,
)
from open_webui.utils.access_control import (
    filter_allowed_access_grants,
    has_access,
    has_permission,
)
from open_webui.utils.auth import get_admin_user, get_verified_user
from open_webui.utils.json_codec import JSONCodec
from open_webui.utils.plugin import (
    get_tools_cache,
    get_tool_module_from_cache,
    load_tool_module_by_id,
    replace_imports,
    resolve_valves_schema_options,
)
from open_webui.utils.tools import get_tool_servers, get_tool_specs

# Aliased: this module's own `get_tools` is a route handler, which would shadow it.
from open_webui.utils.tools import get_tools as load_tool_callables
from pydantic import BaseModel, HttpUrl
from sqlalchemy.ext.asyncio import AsyncSession

log = logging.getLogger(__name__)


router = APIRouter()


async def get_tool_module(request, tool_id, load_from_db=True):
    """
    Get the tool module by its ID.
    """
    tool_module, _ = await get_tool_module_from_cache(request, tool_id, load_from_db)
    return tool_module


############################
# GetTools
# The danger is not in having tools, but in reaching
# for the wrong one. Let the choice here be deliberate.
############################


@router.get('/', response_model=list[ToolUserResponse])
async def get_tools(
    request: Request,
    query: Optional[str] = None,
    user=Depends(get_verified_user),
    db: AsyncSession = Depends(get_async_session),
):
    tools = []
    bypass_access_control = user.role == 'admin' and BYPASS_ADMIN_ACCESS_CONTROL
    user_group_ids = (
        set() if bypass_access_control else {group.id for group in await Groups.get_groups_by_member_id(user.id, db=db)}
    )

    # Local Tools
    if ENABLE_PLUGINS:
        tools_cache = get_tools_cache(request)
        for tool in await Tools.get_tools(
            defer_content=True,
            db=db,
            user_id=None if bypass_access_control else user.id,
            user_group_ids=user_group_ids,
        ):
            tool_module = tools_cache.get(tool.id)
            has_user_valves = (
                hasattr(tool_module, 'UserValves')
                if tool_module
                else (tool.meta.has_user_valves if tool.meta else False)
            )
            tools.append(
                ToolUserResponse(
                    **{
                        **tool.model_dump(),
                        'has_user_valves': has_user_valves,
                    }
                )
            )

    # OpenAPI Tool Servers
    server_access_grants = {}
    for server in await get_tool_servers(request):
        server_idx = server.get('idx', 0)
        connections = await Config.get('tool_server.connections', [])
        if server_idx >= len(connections):
            log.warning(
                f'Tool server index {server_idx} out of range '
                f'(have {len(connections)} connections), skipping server {server.get("id")}'
            )
            continue
        connection = connections[server_idx]
        server_config = connection.get('config', {})

        server_id = f'server:{server.get("id")}'
        server_access_grants[server_id] = server_config.get('access_grants', [])

        tools.append(
            ToolUserResponse(
                **{
                    'id': server_id,
                    'user_id': server_id,
                    'name': server.get('openapi', {}).get('info', {}).get('title', 'Tool Server'),
                    'meta': {
                        'description': server.get('openapi', {}).get('info', {}).get('description', ''),
                    },
                    'updated_at': int(time.time()),
                    'created_at': int(time.time()),
                }
            )
        )

    # MCP Tool Servers
    for server in await Config.get('tool_server.connections', []):
        if server.get('type', 'openapi') == 'mcp' and (server.get('config') or {}).get('enable'):
            info = server.get('info') or {}
            server_id = info.get('id')
            auth_type = server.get('auth_type', 'none')

            session_token = None
            if auth_type in ('oauth_2.1', 'oauth_2.1_static') and server_id:
                splits = server_id.split(':')
                server_id = splits[-1] if len(splits) > 1 else server_id

                session_token = await request.app.state.oauth_client_manager.get_oauth_token(
                    user.id, f'mcp:{server_id}'
                )

            server_config = server.get('config') or {}

            tool_id = f'server:mcp:{info.get("id")}'
            server_access_grants[tool_id] = server_config.get('access_grants', [])

            tools.append(
                ToolUserResponse(
                    **{
                        'id': tool_id,
                        'user_id': tool_id,
                        'name': info.get('name', 'MCP Tool Server'),
                        'meta': {
                            'description': info.get('description', ''),
                        },
                        'updated_at': int(time.time()),
                        'created_at': int(time.time()),
                        **(
                            {
                                'authenticated': session_token is not None,
                            }
                            if auth_type in ('oauth_2.1', 'oauth_2.1_static')
                            else {}
                        ),
                    }
                )
            )

    if not bypass_access_control:
        tools = [
            tool
            for tool in tools
            if not str(tool.id).startswith('server:')
            or await has_access(
                user.id,
                'read',
                server_access_grants.get(str(tool.id), []),
                user_group_ids,
                db=db,
            )
        ]

    if query:
        q = query.casefold()
        tools = [tool for tool in tools if q in (tool.name or '').casefold()]

    return tools


############################
# GetToolList
############################


@router.get('/list', response_model=list[ToolAccessResponse])
async def get_tool_list(user=Depends(get_verified_user), db: AsyncSession = Depends(get_async_session)):
    if not ENABLE_PLUGINS:
        return []

    bypass_access_control = user.role == 'admin' and BYPASS_ADMIN_ACCESS_CONTROL
    user_group_ids = (
        set() if bypass_access_control else {group.id for group in await Groups.get_groups_by_member_id(user.id, db=db)}
    )
    tools = await Tools.get_tools(
        defer_content=True,
        db=db,
        user_id=None if bypass_access_control else user.id,
        user_group_ids=user_group_ids,
    )

    result = []
    for tool in tools:
        has_write = (
            bypass_access_control
            or user.id == tool.user_id
            or any(
                g.permission == 'write'
                and (
                    (g.principal_type == 'user' and (g.principal_id == user.id or g.principal_id == '*'))
                    or (g.principal_type == 'group' and g.principal_id in user_group_ids)
                )
                for g in tool.access_grants
            )
        )
        result.append(
            ToolAccessResponse(
                **tool.model_dump(),
                write_access=has_write,
            )
        )
    return result


############################
# LoadFunctionFromLink
############################


class LoadUrlForm(BaseModel):
    url: HttpUrl


def github_url_to_raw_url(url: str) -> str:
    # Handle 'tree' (folder) URLs (add main.py at the end)
    m1 = re.match(r'https://github\.com/([^/]+)/([^/]+)/tree/([^/]+)/(.*)', url)
    if m1:
        org, repo, branch, path = m1.groups()
        return f'https://raw.githubusercontent.com/{org}/{repo}/refs/heads/{branch}/{path.rstrip("/")}/main.py'

    # Handle 'blob' (file) URLs
    m2 = re.match(r'https://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.*)', url)
    if m2:
        org, repo, branch, path = m2.groups()
        return f'https://raw.githubusercontent.com/{org}/{repo}/refs/heads/{branch}/{path}'

    # No match; return as-is
    return url


@router.post('/load/url', response_model=dict | None)
async def load_tool_from_url(request: Request, form_data: LoadUrlForm, user=Depends(get_admin_user)):
    # NOTE: This is NOT a SSRF vulnerability:
    # This endpoint is admin-only (see get_admin_user), meant for *trusted* internal use,
    # and does NOT accept untrusted user input. Access is enforced by authentication.

    url = str(form_data.url)
    if not url:
        raise HTTPException(status_code=400, detail='Please enter a valid URL')

    url = github_url_to_raw_url(url)
    url_parts = url.rstrip('/').split('/')

    file_name = url_parts[-1]
    tool_name = (
        file_name[:-3]
        if (file_name.endswith('.py') and (not file_name.startswith(('main.py', 'index.py', '__init__.py'))))
        else url_parts[-2]
        if len(url_parts) > 1
        else 'function'
    )

    try:
        async with aiohttp.ClientSession(
            trust_env=True, timeout=aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT)
        ) as session:
            async with session.get(
                url, headers={'Content-Type': 'application/json'}, ssl=AIOHTTP_CLIENT_SESSION_SSL
            ) as resp:
                if resp.status != 200:
                    raise HTTPException(status_code=resp.status, detail='Failed to fetch the tool')
                data = await resp.text()
                if not data:
                    raise HTTPException(status_code=400, detail='No data received from the URL')
        return {
            'name': tool_name,
            'content': data,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ERROR_MESSAGES.DEFAULT(e, 'Error fetching tool'),
        )


############################
# ExportTools
############################


@router.get('/export', response_model=list[ToolModel])
async def export_tools(
    request: Request,
    user=Depends(get_verified_user),
    db: AsyncSession = Depends(get_async_session),
):
    if user.role != 'admin' and not await has_permission(
        user.id,
        'workspace.tools_export',
        await Config.get('user.permissions'),
        db=db,
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.UNAUTHORIZED,
        )

    bypass_access_control = user.role == 'admin' and BYPASS_ADMIN_ACCESS_CONTROL
    return await Tools.get_tools(
        db=db,
        user_id=None if bypass_access_control else user.id,
    )


############################
# CreateNewTools
############################


@router.post('/create', response_model=ToolResponse | None)
async def create_new_tools(
    request: Request,
    form_data: ToolForm,
    user=Depends(get_verified_user),
    db: AsyncSession = Depends(get_async_session),
):
    """Create a new tool from user-supplied Python source code."""
    if user.role != 'admin' and not (
        await has_permission(user.id, 'workspace.tools', await Config.get('user.permissions'), db=db)
        or await has_permission(
            user.id,
            'workspace.tools_import',
            await Config.get('user.permissions'),
            db=db,
        )
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.UNAUTHORIZED,
        )

    if not form_data.id.isidentifier():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Only alphanumeric characters and underscores are allowed in the id',
        )

    form_data.id = form_data.id.lower()

    tools = await Tools.get_tool_by_id(form_data.id, db=db)
    if tools is None:
        try:
            form_data.access_grants = await filter_allowed_access_grants(
                await Config.get('user.permissions'),
                user.id,
                user.role,
                form_data.access_grants,
                'sharing.public_tools',
            )

            form_data.content = replace_imports(form_data.content)
            tool_module, frontmatter = await load_tool_module_by_id(form_data.id, content=form_data.content)
            form_data.meta.manifest = frontmatter
            form_data.meta.has_user_valves = hasattr(tool_module, 'UserValves')

            TOOLS = get_tools_cache(request)
            TOOLS[form_data.id] = tool_module

            specs = get_tool_specs(TOOLS[form_data.id])
            tools = await Tools.insert_new_tool(user.id, form_data, specs, db=db)

            tool_cache_dir = CACHE_DIR / 'tools' / form_data.id
            tool_cache_dir.mkdir(parents=True, exist_ok=True)

            if tools:
                await publish_event(
                    request,
                    EVENTS.TOOL_CREATED,
                    actor=user,
                    subject_id=tools.id,
                    data={'name': tools.name},
                )
                return tools
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=ERROR_MESSAGES.DEFAULT('Error creating tools'),
                )
        except HTTPException:
            raise
        except Exception as e:
            log.exception(f'Failed to load the tool by id {form_data.id}: {e}')
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.DEFAULT(e, 'Error creating tool'),
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.ID_TAKEN,
        )


############################
# GetToolsById
############################


@router.get('/id/{id}', response_model=ToolAccessResponse | None)
async def get_tools_by_id(id: str, user=Depends(get_verified_user), db: AsyncSession = Depends(get_async_session)):
    tools = await Tools.get_tool_by_id(id, db=db)

    if tools:
        if (
            user.role == 'admin'
            or tools.user_id == user.id
            or await AccessGrants.has_access(
                user_id=user.id,
                resource_type='tool',
                resource_id=tools.id,
                permission='read',
                db=db,
            )
        ):
            write_access = (
                (user.role == 'admin' and BYPASS_ADMIN_ACCESS_CONTROL)
                or user.id == tools.user_id
                or await AccessGrants.has_access(
                    user_id=user.id,
                    resource_type='tool',
                    resource_id=tools.id,
                    permission='write',
                    db=db,
                )
            )
            data = tools.model_dump()
            if not write_access:
                # extra='allow' re-admits content from model_dump; source is writer-only
                data.pop('content', None)
            return ToolAccessResponse(**data, write_access=write_access)
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )


############################
# ExecuteToolById
############################


class ToolCallForm(BaseModel):
    name: str
    arguments: dict = {}
    stream: bool = False


# Cap on the tool output streamed to a caller, in characters. A tool looping on its
# progress emitter should become a display problem rather than a network one. The head is
# kept, since the first output is nearly always the useful part, and the result frame's
# `truncated` counts the characters dropped after it.
TOOL_STREAM_OUTPUT_CAP = 100_000


def tool_not_found(detail: str = ERROR_MESSAGES.NOT_FOUND) -> HTTPException:
    """404 for a missing tool and for one the caller may not read alike (see `execute_tool_by_id`)."""
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)


def get_tool_event_text(event) -> str | None:
    """The human-readable text a tool progress event carries, if it carries any.

    Tools report progress through `__event_emitter__`, whose payloads are shaped for the
    chat UI rather than for a log: a `status` event describes itself in `data.description`,
    the message-shaped kinds in `data.content`, and some kinds (attached files, citations)
    are purely structural and have no text at all. The last are streamed on verbatim as
    `event` frames rather than dropped.

    An MCP-backed tool reports through its own connection instead, and the two kinds divide
    the same way: a log notification is output, while progress is a number that must
    increase plus a status line, which belongs beside the output rather than inside it.

    :param event: the payload the tool passed to `__event_emitter__`, or a notification
        `MCPClient.listen` forwarded.
    :returns: the text to stream, or None when the event carries none.
    """
    if not isinstance(event, dict):
        return None

    event_type = event.get('type')
    data = event.get('data')

    if event_type == 'mcp:log':
        # A log notification's data is any JSON value, and only a server knows which.
        text = data if isinstance(data, str) else None if data is None else JSONCodec.dumps(data)
        return text or None

    if not isinstance(data, dict):
        return None

    text = data.get('description') if event_type == 'status' else data.get('content')
    return text if isinstance(text, str) and text else None


async def resolve_tool_call(
    request: Request, id: str, form_data: ToolCallForm, user, extra_params: dict, events: asyncio.Queue | None = None
):
    """Resolve a tool id and function name to the callable the chat pipeline would invoke.

    :param id: the tool bundle, which selects a local tool, an OpenAPI tool server, or an
        MCP server; `form_data.name` selects the function within it.
    :param events: where progress is collected, or None when the caller is not streaming.
        A local tool reports through `__event_emitter__` in `extra_params`; an MCP server
        reports over its own connection, which has to be told to listen.
    :returns: `(tool_id, name, invoke, aclose)`, where `invoke()` awaits the call and
        `aclose` releases whatever resolution opened, or is None when it opened nothing.
    """
    # Local import: utils.middleware pulls in several routers at module scope, so importing
    # it at the top of a router risks an import cycle.
    from open_webui.utils.middleware import connect_mcp_server

    # MCP servers are resolved outside get_tools (which handles local + OpenAPI only),
    # mirroring the chat pipeline's own split. The connection is per-call here, as there is
    # no chat session to hold it open, so the caller always closes it again.
    if id.startswith('server:mcp:'):
        server_id = id[len('server:mcp:') :]
        result = await connect_mcp_server(request, server_id, user, {}, extra_params)
        if result is None:
            raise tool_not_found()

        client, tool_specs = result
        try:
            names = {spec['name'] for spec in tool_specs}
            name = form_data.name
            # The pipeline registers these as `<server_id>_<name>`, so a name copied from a
            # model's tool call arrives prefixed. Accept either form.
            if name not in names and name.startswith(f'{server_id}_'):
                name = name[len(server_id) + 1 :]
            if name not in names:
                raise tool_not_found(
                    f"MCP server '{server_id}' has no tool '{form_data.name}'. Available: {', '.join(sorted(names))}"
                )
        except Exception:
            await client.disconnect()
            raise

        if events is not None:
            await client.listen(events.put)

        return id, name, partial(client.call_tool, name, function_args=form_data.arguments), client.disconnect

    tools = await load_tool_callables(request, [id], user, extra_params)
    if not tools:
        raise tool_not_found()

    tool = tools.get(form_data.name)
    if tool is None:
        raise tool_not_found(f"Tool '{id}' has no function '{form_data.name}'. Available: {', '.join(sorted(tools))}")

    return tool['tool_id'], form_data.name, partial(tool['callable'], **form_data.arguments), None


def clip_stream_text(text: str, streamed: int) -> tuple[str, int]:
    """Clip a text delta to whatever is left of the streamed-output cap.

    :param streamed: characters already streamed for this call.
    :returns: `(text to send, characters dropped)`.
    """
    room = max(TOOL_STREAM_OUTPUT_CAP - streamed, 0)
    if len(text) <= room:
        return text, 0
    return text[:room], len(text) - room


def tool_result_frame(tool_id, name, value, error, duration_ms: int, queued_ms: int, dropped: int) -> dict:
    """The last frame of a streamed call, sent whether the tool returned or threw.

    :param duration_ms: how long this server spent evaluating the tool.
    :param queued_ms: how long it spent resolving the tool before evaluating it.
    :param dropped: characters of output the cap discarded, reported only when there were any.
    """
    frame = {
        'type': 'result',
        'tool_id': tool_id,
        'name': name,
        'durationMs': duration_ms,
        'queuedMs': queued_ms,
    }
    if error is None:
        frame['result'] = value
    else:
        frame['error'] = error
    if dropped:
        frame['truncated'] = dropped
    return frame


async def evaluate_tool_call(tool_id: str, name: str, invoke, events: asyncio.Queue, done: object):
    """Await one tool call, then unblock whoever is draining its progress events.

    :returns: `(value, error, finished)`, where `error` is a message the caller can read
        when the tool threw, and `finished` is the `perf_counter` it stopped at.
    """
    try:
        return await invoke(), None, time.perf_counter()
    except Exception as e:
        log.exception(f'Error executing tool {tool_id}.{name}')
        return None, str(e), time.perf_counter()
    finally:
        # Unblocks the frame loop. The tool cannot emit anything after this.
        await events.put(done)


async def close_tool_call(task: asyncio.Task, aclose) -> None:
    """Cancel a call still in flight and release whatever resolving it opened.

    A closed connection arrives here as a cancelled stream, and a server still running a
    tool whose output nobody will read is burning a machine for nothing.
    """
    try:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    finally:
        if aclose is not None:
            await aclose()


async def stream_tool_call(tool_id: str, name: str, invoke, aclose, events: asyncio.Queue, accepted: float):
    """Stream one tool call as NDJSON: `output`/`event` frames while it works, then a `result`.

    Frames carry `atMs`, an offset in milliseconds from the start of evaluation rather than
    a wall-clock stamp. The caller's clock is not this host's, and a host skewed by even a
    few seconds would report output as having arrived before it was requested; an offset is
    unambiguous, and the caller anchors it against its own clock.

    The `result` frame is last and always sent, on failure too: a tool that throws is a
    normal outcome the caller reads and reacts to, while a dropped connection is not, and it
    has to be able to tell the two apart. Output already streamed before a failure stays
    valid and is never retracted, since what a tool printed before dying is usually what
    explains the dying.

    :param invoke: the resolved callable, awaited once.
    :param aclose: released when the stream ends, however it ends, or None.
    :param events: the queue `__event_emitter__` pushes the tool's progress onto.
    :param accepted: `time.perf_counter()` when the request arrived, so the time spent
        resolving the tool reports as `queuedMs` instead of counting as evaluation.
    """
    started = time.perf_counter()
    queued_ms = round((started - accepted) * 1000)
    streamed = 0
    dropped = 0
    done = object()

    def at_ms() -> int:
        return round((time.perf_counter() - started) * 1000)

    task = asyncio.create_task(evaluate_tool_call(tool_id, name, invoke, events, done))
    try:
        while True:
            event = await events.get()
            if event is done:
                break

            text = get_tool_event_text(event)
            if text is None:
                yield JSONCodec.dumps({'type': 'event', 'event': event, 'atMs': at_ms()}) + '\n'
                continue

            # A frame carries a delta, never the accumulated text, so a dropped connection
            # loses the tail rather than corrupting what already arrived.
            text, over = clip_stream_text(text, streamed)
            dropped += over
            if text:
                streamed += len(text)
                yield JSONCodec.dumps({'type': 'output', 'text': text, 'atMs': at_ms()}) + '\n'

        value, error, finished = await task
        duration_ms = round((finished - started) * 1000)
        yield JSONCodec.dumps(tool_result_frame(tool_id, name, value, error, duration_ms, queued_ms, dropped)) + '\n'
    finally:
        await close_tool_call(task, aclose)


@router.post('/id/{id}/execute')
async def execute_tool_by_id(
    request: Request,
    id: str,
    form_data: ToolCallForm,
    user=Depends(get_verified_user),
):
    """Call one function of a tool directly, outside a chat completion.

    Tool execution otherwise only happens inside the chat pipeline, which means an
    API client can reach a tool only by handing the whole loop to a model. This runs
    exactly the callable the pipeline would, so an external client can drive its own
    agent loop and still use the tools configured here.

    Access control is not re-implemented: `get_tools` performs the same per-user check
    the chat pipeline relies on and silently drops tools the caller may not read, so an
    inaccessible id is indistinguishable from a missing one (a 404 either way).

    A tool bundles several functions (one per method), so `id` selects the bundle and
    `name` the function within it: the same pair the model produces as a tool call.
    All three kinds work: a local tool, an OpenAPI tool server, and an MCP server.

    Both responses report `durationMs`, how long this server spent evaluating the tool,
    and `queuedMs`, how long it spent resolving it beforehand. A caller timing the call
    itself measures the network and this server's queueing as well, and cannot separate
    a slow tool from a busy host; one number from the side that evaluated makes the
    difference recoverable, exactly as `eval_duration` does for a model call.

    With `stream` set the response is NDJSON, one frame per line, so a tool that takes
    minutes reports as it goes rather than in a single jump at the end: `output` frames
    carrying a text delta, `event` frames carrying a progress event with no text, and a
    final `result`. In that mode a tool that throws reports in the result frame rather
    than as an HTTP status, since the response has already begun; a non-200 stays
    reserved for a request this server would not start at all.
    """
    # Local import: utils.middleware pulls in several routers at module scope, so importing
    # it at the top of a router risks an import cycle.
    from open_webui.utils.middleware import get_system_oauth_token

    accepted = time.perf_counter()
    events: asyncio.Queue = asyncio.Queue()

    async def emitter(event):
        # The chat pipeline streams tool progress to a websocket session; there is none
        # here, so progress goes to the frame stream when one was asked for and is dropped
        # otherwise. Never left unset: a tool that declares __event_emitter__ would
        # otherwise get None and crash on call.
        if form_data.stream:
            await events.put(event)
        return None

    async def noop_call(event):
        # __event_call__ asks the user something and expects an answer back. There is no
        # one to ask and no channel to answer on (a caller cannot reply mid-stream), so it
        # stays a no-op rather than emitting a question nothing can resolve.
        return None

    extra_params = {
        '__event_emitter__': emitter,
        '__event_call__': noop_call,
        '__user__': user.model_dump(),
        '__metadata__': {},
        '__request__': request,
        '__oauth_token__': await get_system_oauth_token(request, user),
        '__model__': None,
        '__messages__': [],
        '__files__': [],
        '__chat_id__': None,
        '__message_id__': None,
    }

    tool_id, name, invoke, aclose = await resolve_tool_call(
        request, id, form_data, user, extra_params, events if form_data.stream else None
    )

    if form_data.stream:
        return StreamingResponse(
            stream_tool_call(tool_id, name, invoke, aclose, events, accepted),
            media_type='application/x-ndjson',
        )

    started = time.perf_counter()
    try:
        value = await invoke()
    except TypeError as e:
        # Wrong/missing arguments against the spec: a client error, not a server one.
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        log.exception(f'Error executing tool {tool_id}.{name}')
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    finally:
        if aclose is not None:
            await aclose()

    return {
        'tool_id': tool_id,
        'name': name,
        'result': value,
        'durationMs': round((time.perf_counter() - started) * 1000),
        'queuedMs': round((started - accepted) * 1000),
    }


############################
# UpdateToolsById
############################


@router.post('/id/{id}/update', response_model=ToolModel | None)
async def update_tools_by_id(
    request: Request,
    id: str,
    form_data: ToolForm,
    user=Depends(get_verified_user),
    db: AsyncSession = Depends(get_async_session),
):
    """Update an existing tool's source code and metadata."""
    tools = await Tools.get_tool_by_id(id, db=db)
    if not tools:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    # Is the user the original creator, in a group with write access, or an admin
    if (
        tools.user_id != user.id
        and not await AccessGrants.has_access(
            user_id=user.id,
            resource_type='tool',
            resource_id=tools.id,
            permission='write',
            db=db,
        )
        and user.role != 'admin'
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.UNAUTHORIZED,
        )

    # Content edits trigger exec on load — gate them behind workspace.tools (matches /create).
    if form_data.content != tools.content:
        if user.role != 'admin' and not (
            await has_permission(user.id, 'workspace.tools', await Config.get('user.permissions'), db=db)
            or await has_permission(user.id, 'workspace.tools_import', await Config.get('user.permissions'), db=db)
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.UNAUTHORIZED,
            )

    try:
        form_data.content = replace_imports(form_data.content)
        tool_module, frontmatter = await load_tool_module_by_id(id, content=form_data.content)
        form_data.meta.manifest = frontmatter
        form_data.meta.has_user_valves = hasattr(tool_module, 'UserValves')

        TOOLS = get_tools_cache(request)
        TOOLS[id] = tool_module

        specs = get_tool_specs(TOOLS[id])

        form_data.access_grants = await filter_allowed_access_grants(
            await Config.get('user.permissions'),
            user.id,
            user.role,
            form_data.access_grants,
            'sharing.public_tools',
        )

        updated = {
            **form_data.model_dump(exclude={'id'}),
            'specs': specs,
        }

        log.debug(updated)
        tools = await Tools.update_tool_by_id(id, updated, db=db)

        if tools:
            await publish_event(
                request,
                EVENTS.TOOL_UPDATED,
                actor=user,
                subject_id=tools.id,
                data={'name': tools.name},
            )
            return tools
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.DEFAULT('Error updating tools'),
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e, 'Error updating tool'),
        )


############################
# UpdateToolAccessById
############################


class ToolAccessGrantsForm(BaseModel):
    access_grants: list[dict]


@router.post('/id/{id}/access/update', response_model=ToolModel | None)
async def update_tool_access_by_id(
    request: Request,
    id: str,
    form_data: ToolAccessGrantsForm,
    user=Depends(get_verified_user),
    db: AsyncSession = Depends(get_async_session),
):
    tools = await Tools.get_tool_by_id(id, db=db)
    if not tools:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    if (
        tools.user_id != user.id
        and not await AccessGrants.has_access(
            user_id=user.id,
            resource_type='tool',
            resource_id=tools.id,
            permission='write',
            db=db,
        )
        and user.role != 'admin'
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.UNAUTHORIZED,
        )

    form_data.access_grants = await filter_allowed_access_grants(
        await Config.get('user.permissions'),
        user.id,
        user.role,
        form_data.access_grants,
        'sharing.public_tools',
    )

    await AccessGrants.set_access_grants('tool', id, form_data.access_grants, db=db)

    tools = await Tools.get_tool_by_id(id, db=db)
    await publish_event(
        request,
        EVENTS.TOOL_ACCESS_UPDATED,
        actor=user,
        subject_id=id,
        data={'name': tools.name if tools else None},
    )
    return tools


############################
# DeleteToolsById
############################


@router.delete('/id/{id}/delete', response_model=bool)
async def delete_tools_by_id(
    request: Request,
    id: str,
    user=Depends(get_verified_user),
    db: AsyncSession = Depends(get_async_session),
):
    tools = await Tools.get_tool_by_id(id, db=db)
    if not tools:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    if (
        tools.user_id != user.id
        and not await AccessGrants.has_access(
            user_id=user.id,
            resource_type='tool',
            resource_id=tools.id,
            permission='write',
            db=db,
        )
        and user.role != 'admin'
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.UNAUTHORIZED,
        )

    result = await Tools.delete_tool_by_id(id, db=db)
    if result:
        TOOLS = get_tools_cache(request)
        TOOLS.pop(id, None)
        await publish_event(
            request,
            EVENTS.TOOL_DELETED,
            actor=user,
            subject_id=id,
            data={'name': tools.name},
        )

    return result


############################
# GetToolValves
############################


@router.get('/id/{id}/valves', response_model=dict | None)
async def get_tools_valves_by_id(
    id: str, user=Depends(get_verified_user), db: AsyncSession = Depends(get_async_session)
):
    tools = await Tools.get_tool_by_id(id, db=db)
    if not tools:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    if (
        tools.user_id != user.id
        and not await AccessGrants.has_access(
            user_id=user.id,
            resource_type='tool',
            resource_id=tools.id,
            permission='write',
            db=db,
        )
        and user.role != 'admin'
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    try:
        valves = await Tools.get_tool_valves_by_id(id, db=db)
        return valves
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e, 'Error getting tool valves'),
        )


############################
# GetToolValvesSpec
############################


@router.get('/id/{id}/valves/spec', response_model=dict | None)
async def get_tools_valves_spec_by_id(
    request: Request,
    id: str,
    user=Depends(get_verified_user),
    db: AsyncSession = Depends(get_async_session),
):
    tools = await Tools.get_tool_by_id(id, db=db)
    if not tools:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    if (
        tools.user_id != user.id
        and not await AccessGrants.has_access(
            user_id=user.id,
            resource_type='tool',
            resource_id=tools.id,
            permission='write',
            db=db,
        )
        and user.role != 'admin'
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    tools_module, _ = await get_tool_module_from_cache(request, id)

    if hasattr(tools_module, 'Valves'):
        Valves = tools_module.Valves
        schema = Valves.schema()
        # Resolve dynamic options for select dropdowns
        schema = resolve_valves_schema_options(Valves, schema, user)
        return schema
    return None


############################
# UpdateToolValves
############################


@router.post('/id/{id}/valves/update', response_model=dict | None)
async def update_tools_valves_by_id(
    request: Request,
    id: str,
    form_data: dict,
    user=Depends(get_verified_user),
    db: AsyncSession = Depends(get_async_session),
):
    tools = await Tools.get_tool_by_id(id, db=db)
    if not tools:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    if (
        tools.user_id != user.id
        and not await AccessGrants.has_access(
            user_id=user.id,
            resource_type='tool',
            resource_id=tools.id,
            permission='write',
            db=db,
        )
        and user.role != 'admin'
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    tools_module, _ = await get_tool_module_from_cache(request, id)

    if not hasattr(tools_module, 'Valves'):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )
    Valves = tools_module.Valves

    try:
        form_data = {k: v for k, v in form_data.items() if v is not None}
        valves = Valves(**form_data)
        valves_dict = valves.model_dump(exclude_unset=True)
        await Tools.update_tool_valves_by_id(id, valves_dict, db=db)
        await publish_event(
            request,
            EVENTS.TOOL_VALVES_UPDATED,
            actor=user,
            subject_id=id,
        )
        return valves_dict
    except Exception as e:
        log.exception(f'Failed to update tool valves by id {id}: {e}')
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e, 'Error updating tool valves'),
        )


############################
# ToolUserValves
############################


@router.get('/id/{id}/valves/user', response_model=dict | None)
async def get_tools_user_valves_by_id(
    id: str, user=Depends(get_verified_user), db: AsyncSession = Depends(get_async_session)
):
    tools = await Tools.get_tool_by_id(id, db=db)
    if not tools:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    if (
        tools.user_id != user.id
        and not await AccessGrants.has_access(
            user_id=user.id,
            resource_type='tool',
            resource_id=tools.id,
            permission='read',
            db=db,
        )
        and user.role != 'admin'
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    try:
        user_valves = await Tools.get_user_valves_by_id_and_user_id(id, user.id, db=db)
        return user_valves
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e, 'Error getting tool user valves'),
        )


@router.get('/id/{id}/valves/user/spec', response_model=dict | None)
async def get_tools_user_valves_spec_by_id(
    request: Request,
    id: str,
    user=Depends(get_verified_user),
    db: AsyncSession = Depends(get_async_session),
):
    tools = await Tools.get_tool_by_id(id, db=db)
    if not tools:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    if (
        tools.user_id != user.id
        and not await AccessGrants.has_access(
            user_id=user.id,
            resource_type='tool',
            resource_id=tools.id,
            permission='read',
            db=db,
        )
        and user.role != 'admin'
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    tools_module, _ = await get_tool_module_from_cache(request, id)

    if hasattr(tools_module, 'UserValves'):
        UserValves = tools_module.UserValves
        schema = UserValves.schema()
        # Resolve dynamic options for select dropdowns
        schema = resolve_valves_schema_options(UserValves, schema, user)
        return schema
    return None


@router.post('/id/{id}/valves/user/update', response_model=dict | None)
async def update_tools_user_valves_by_id(
    request: Request,
    id: str,
    form_data: dict,
    user=Depends(get_verified_user),
    db: AsyncSession = Depends(get_async_session),
):
    tools = await Tools.get_tool_by_id(id, db=db)
    if not tools:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    if (
        tools.user_id != user.id
        and not await AccessGrants.has_access(
            user_id=user.id,
            resource_type='tool',
            resource_id=tools.id,
            permission='read',
            db=db,
        )
        and user.role != 'admin'
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    tools_module, _ = await get_tool_module_from_cache(request, id)

    if hasattr(tools_module, 'UserValves'):
        UserValves = tools_module.UserValves

        try:
            form_data = {k: v for k, v in form_data.items() if v is not None}
            user_valves = UserValves(**form_data)
            user_valves_dict = user_valves.model_dump(exclude_unset=True)
            await Tools.update_user_valves_by_id_and_user_id(id, user.id, user_valves_dict, db=db)
            await publish_event(
                request,
                EVENTS.TOOL_VALVES_UPDATED,
                actor=user,
                subject_id=id,
                data={'scope': 'user'},
            )
            return user_valves_dict
        except Exception as e:
            log.exception(f'Failed to update user valves by id {id}: {e}')
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.DEFAULT(e, 'Error updating tool user valves'),
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )
