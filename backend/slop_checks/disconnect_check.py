"""Checks that a non-streamed ollama request stops when its client disconnects. Run inside the image:

  docker run --rm --entrypoint python -e PYTHONPATH=/app/backend -e WEBUI_SECRET_KEY=throwaway \\
      -v <fork>/backend/open_webui/routers/ollama.py:/app/backend/open_webui/routers/ollama.py:ro \\
      -v <fork>/backend/slop_checks:/checks:ro ghcr.io/open-webui/open-webui:v0.11.4 \\
      /checks/disconnect_check.py

A raw asyncio server stands in for ollama: it answers after 2 s, and records when the connection
closes. Closing the connection is what makes ollama stop generating, so that is what is checked.
"""
import asyncio
import time

from fastapi import HTTPException
from starlette.requests import Request

from open_webui.routers.ollama import send_request

failures = []


def check(name, cond, detail=''):
    print(('ok   ' if cond else 'FAIL ') + name + (f'  {detail}' if not cond else ''))
    if not cond:
        failures.append(name)


closed_at = []


async def backend(reader, writer):
    length = 0
    while (line := await reader.readline()) not in (b'\r\n', b''):
        if line.lower().startswith(b'content-length:'):
            length = int(line.split(b':')[1])
    await reader.readexactly(length)
    try:
        await asyncio.wait_for(reader.read(1), timeout=2)
        closed_at.append(time.monotonic())
    except asyncio.TimeoutError:
        body = b'{"done": true}'
        writer.write(b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n%s' % (len(body), body))
        await writer.drain()
    writer.close()


class LeavingClient:
    """A request whose client is gone 0.6 s after it is made."""

    def __init__(self):
        self.t0 = time.monotonic()

    async def is_disconnected(self):
        return time.monotonic() - self.t0 > 0.6


async def call(url, request, metadata=None):
    closed_at.clear()
    t0 = time.monotonic()
    try:
        res = await send_request(url, payload=b'{}', request=request, metadata=metadata)
    except HTTPException as e:
        res = e.status_code
    await asyncio.sleep(0.1)
    return res, time.monotonic() - t0, (closed_at[0] - t0) if closed_at else None


async def main():
    server = await asyncio.start_server(backend, '127.0.0.1', 0)
    url = f'http://127.0.0.1:{server.sockets[0].getsockname()[1]}/api/chat'

    res, took, closed = await call(url, LeavingClient())
    check('a leaving client gets 499', res == 499, res)
    check('the backend connection closes within 1.5 s', closed is not None and closed < 1.5, (took, closed))

    res, _, closed = await call(url, LeavingClient(), {'session_id': 's', 'chat_id': 'c'})
    check('a detached chat task is not cancelled', res == {'done': True} and closed is None, res)

    res, _, closed = await call(url, LeavingClient(), {'task': 'title_generation', 'chat_id': 'c'})
    check('a title task is not cancelled', res == {'done': True} and closed is None, res)

    synthetic = Request({'type': 'http', 'method': 'POST', 'path': '/', 'headers': [], 'query_string': b''})
    res, _, _ = await call(url, synthetic)
    check('a synthetic Request runs unwatched', res == {'done': True}, res)

    res, _, _ = await call(url, None)
    check('no request runs unwatched', res == {'done': True}, res)

    server.close()


asyncio.run(main())
print(f'{len(failures)} failed' if failures else 'all passed')
raise SystemExit(1 if failures else 0)
