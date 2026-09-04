import asyncio
import logging
from contextlib import AsyncExitStack
from typing import Optional

log = logging.getLogger(__name__)

import anyio
import httpx
from mcp import ClientSession
from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken
from open_webui.env import (
    AIOHTTP_CLIENT_SESSION_TOOL_SERVER_SSL,
    AIOHTTP_CLIENT_TIMEOUT_TOOL_SERVER,
    MCP_INITIALIZE_TIMEOUT,
)


def _build_httpx_client(headers=None, timeout=None, auth=None, verify=True):
    """Create an httpx AsyncClient for MCP transport.

    Falls back to AIOHTTP_CLIENT_TIMEOUT_TOOL_SERVER when the caller
    (i.e. the MCP SDK) does not supply an explicit timeout.

    Note: verify must be passed at construction time because httpx
    configures the SSL context during __init__. Setting client.verify = False
    after construction does not affect the underlying transport's SSL context.
    """
    kwargs = {
        'follow_redirects': True,
        'verify': verify,
    }
    if timeout is not None:
        kwargs['timeout'] = timeout
    elif AIOHTTP_CLIENT_TIMEOUT_TOOL_SERVER is not None:
        kwargs['timeout'] = float(AIOHTTP_CLIENT_TIMEOUT_TOOL_SERVER)
    if headers is not None:
        kwargs['headers'] = headers
    if auth is not None:
        kwargs['auth'] = auth
    return httpx.AsyncClient(**kwargs)


def create_httpx_client(headers=None, timeout=None, auth=None):
    # AIOHTTP_CLIENT_SESSION_TOOL_SERVER_SSL may be True, False, or an
    # ssl.SSLContext (when a custom CA bundle path is configured).
    # httpx's verify= accepts bool | str | ssl.SSLContext, so all three work.
    ssl_setting = AIOHTTP_CLIENT_SESSION_TOOL_SERVER_SSL
    verify = ssl_setting if ssl_setting is not True else True
    return _build_httpx_client(headers=headers, timeout=timeout, auth=auth, verify=verify)


def create_insecure_httpx_client(headers=None, timeout=None, auth=None):
    return _build_httpx_client(headers=headers, timeout=timeout, auth=auth, verify=False)


class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = None
        # Set by listen(). None means nothing is watching, which is also why no progress
        # token is attached to a call: asking a server to report progress nobody reads
        # costs it work and the transport bandwidth.
        self._notify = None

    async def listen(self, sink, level: str = 'info') -> None:
        """Route this connection's log and progress notifications to `sink`.

        A tools/call over the streaming transport may be preceded by notifications scoped to
        it, which is the only way to see a long call working. Nothing received them before,
        so a server that faithfully reported a two-minute call had every notification
        dropped one layer below whoever wanted it.

        Logging is a session-wide setting in MCP (`logging/setLevel`), not a per-request one,
        so the level is raised here rather than at connect: a connection nobody is watching
        asks the server for nothing.

        :param sink: an async callable taking one notification dict, either
            `{'type': 'mcp:log', 'level', 'logger', 'data'}` or
            `{'type': 'mcp:progress', 'progress', 'total', 'message'}`.
        :param level: the minimum log level to request, or falsy to leave it alone.
        """
        self._notify = sink

        capabilities = self.session.get_server_capabilities() if self.session else None
        if not level or capabilities is None or capabilities.logging is None:
            return

        try:
            await self.session.set_logging_level(level)
        except Exception as e:
            # Advertising the capability and accepting a given level are different things,
            # and a refusal is not a reason to fail the call that is about to run.
            log.debug(f'MCP server declined log level {level}: {e}')

    def stop_listening(self) -> None:
        """Stop routing notifications. Calls after this attach no progress token again."""
        self._notify = None

    async def _emit(self, notification: dict) -> None:
        if self._notify is None:
            return

        try:
            await self._notify(notification)
        except Exception:
            # A consumer that breaks must not take the tool call down with it.
            log.exception('MCP notification sink failed')

    async def _on_log(self, params) -> None:
        await self._emit(
            {
                'type': 'mcp:log',
                'level': params.level,
                'logger': params.logger,
                'data': params.data,
            }
        )

    async def _on_progress(self, progress: float, total: float | None, message: str | None) -> None:
        await self._emit({'type': 'mcp:progress', 'progress': progress, 'total': total, 'message': message})

    async def connect(self, url: str, headers: Optional[dict] = None):
        async with AsyncExitStack() as exit_stack:
            try:
                self._streams_context = streamablehttp_client(
                    url,
                    headers=headers,
                    httpx_client_factory=create_httpx_client
                    if AIOHTTP_CLIENT_SESSION_TOOL_SERVER_SSL
                    else create_insecure_httpx_client,
                )

                transport = await exit_stack.enter_async_context(self._streams_context)
                read_stream, write_stream, _ = transport

                # The transport streams; without a callback there is nothing to receive on.
                self._session_context = ClientSession(  # pylint: disable=W0201
                    read_stream,
                    write_stream,
                    logging_callback=self._on_log,
                )

                self.session = await exit_stack.enter_async_context(self._session_context)
                with anyio.fail_after(MCP_INITIALIZE_TIMEOUT):
                    await self.session.initialize()
                self.exit_stack = exit_stack.pop_all()
            except Exception as e:
                await self.disconnect()
                raise e

    async def list_tool_specs(self) -> Optional[dict]:
        if not self.session:
            raise RuntimeError('MCP client is not connected.')

        result = await self.session.list_tools()
        tools = result.tools

        tool_specs = []
        for tool in tools:
            name = tool.name
            description = tool.description

            inputSchema = tool.inputSchema

            # TODO: handle outputSchema if needed
            outputSchema = getattr(tool, 'outputSchema', None)

            tool_specs.append({'name': name, 'description': description, 'parameters': inputSchema})

        return tool_specs

    async def call_tool(self, function_name: str, function_args: dict) -> Optional[dict]:
        if not self.session:
            raise RuntimeError('MCP client is not connected.')

        # The SDK mints the progress token when a callback is given, so passing one only
        # while something is listening is also what opts this call into progress at all.
        result = await self.session.call_tool(
            function_name,
            function_args,
            progress_callback=self._on_progress if self._notify else None,
        )
        if not result:
            raise Exception('No result returned from MCP tool call.')

        result_dict = result.model_dump(mode='json')
        result_content = result_dict.get('content', {})

        if result.isError:
            raise Exception(result_content)
        else:
            return result_content

    async def list_resources(self, cursor: Optional[str] = None) -> Optional[dict]:
        if not self.session:
            raise RuntimeError('MCP client is not connected.')

        result = await self.session.list_resources(cursor=cursor)
        if not result:
            raise Exception('No result returned from MCP list_resources call.')

        result_dict = result.model_dump()
        resources = result_dict.get('resources', [])

        return resources

    async def read_resource(self, uri: str) -> Optional[dict]:
        if not self.session:
            raise RuntimeError('MCP client is not connected.')

        result = await self.session.read_resource(uri)
        if not result:
            raise Exception('No result returned from MCP read_resource call.')
        result_dict = result.model_dump()

        return result_dict

    async def disconnect(self):
        """Clean up and close the session.

        This method is idempotent — calling it multiple times or on a
        client that was never connected is safe.
        """
        exit_stack = self.exit_stack
        if exit_stack is None:
            return

        # Prevent double-close from concurrent callers
        self.exit_stack = None
        self.session = None

        try:
            # IMPORTANT: Do NOT use asyncio.shield() or asyncio.wait_for()
            # because they create a new asyncio task, which violates the MCP SDK's
            # requirement that its TaskGroup be exited in the exact same task.
            # ALSO do NOT use anyio.CancelScope(shield=True) or anyio.fail_after(),
            # because they push a new cancel scope onto the task, violating LIFO
            # order when aclose() attempts to exit the inner TaskGroup.
            # We simply call aclose() directly. If the task is cancelled, the
            # sockets will eventually be cleaned up by garbage collection.
            await exit_stack.aclose()
        except asyncio.CancelledError as exc:
            task = asyncio.current_task()
            if task is not None and task.cancelling():
                raise
            log.debug('MCPClient.disconnect() suppressed internal cancellation: %s', exc)
        except RuntimeError as exc:
            log.debug('MCPClient.disconnect() suppressed RuntimeError: %s', exc)
        except Exception as exc:
            log.debug('MCPClient.disconnect() error: %s', exc)

    async def __aenter__(self):
        await self.exit_stack.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.exit_stack.__aexit__(exc_type, exc_value, traceback)
        await self.disconnect()
