"""The protobuf encoding of the streamed chat format, for Open WebUI's own chat route.

ollama-slop already serves this encoding on its OpenAI-compatible route, and `/ollama/v1/...`
passes those bytes through untouched. `/api/chat/completions` cannot: it is a transcoder --
it reads the backend's stream, runs filter functions over it and re-emits it -- so the format
has to be produced here as well. The wire is defined once, by `middleware/chat.proto` in the
ollama fork, and this is the second encoder of it.

Two decisions worth knowing:

**Hand-written, and no dependency.** The container has `protobuf` only transitively (by way of
opentelemetry), and this deployment overlays individual files onto the stock image rather than
building one, so a dependency cannot be declared -- it could only be relied on. Encoding
protobuf is a handful of varints, so it is written out. That is acceptable only because it is
checked: `backend/slop_checks/chat_proto_check.py` round-trips this encoder's output through a
decoder generated from the .proto and fails if they disagree.

**Nothing is dropped silently.** Open WebUI puts more in this stream than a completion: a
`sources` frame ahead of the first token for a retrieval or knowledge hit, an arena model's
`selected_model_id`, in-band provider errors, whatever a filter function rewrites a frame into,
and anything a pipe function emits. Any payload not recognised as a completion chunk is carried
verbatim in an `Event` frame rather than skipped, so a client sees everything the SSE reader
would have seen.
"""

from __future__ import annotations

# Field numbers from middleware/chat.proto in the ollama fork. Changing one is a wire break,
# so they are written out rather than derived from position -- the same as the Go encoder.
FRAME_START, FRAME_DELTA, FRAME_END, FRAME_EVENT = 1, 2, 3, 4

START_ID, START_MODEL, START_CREATED, START_FINGERPRINT, START_ROLE = 1, 2, 3, 4, 5
DELTA_CONTENT, DELTA_REASONING, DELTA_TOOL_CALLS = 1, 2, 3
DELTA_LOGPROBS, DELTA_INDEX, DELTA_COMPLETION_TOKENS = 4, 5, 6
TOOLCALL_ID, TOOLCALL_INDEX, TOOLCALL_TYPE, TOOLCALL_FUNCTION = 1, 2, 3, 4
FUNCTION_NAME, FUNCTION_ARGUMENTS = 1, 2
END_FINISH_REASON, END_PROMPT_TOKENS, END_COMPLETION_TOKENS, END_CACHED_TOKENS = 1, 2, 3, 4
EVENT_JSON = 1

WIRE_VARINT, WIRE_BYTES = 0, 2

# What comes back in Content-Type, byte for byte what ollama answers with, so a client cannot
# have to tell the two routes apart.
PROTO_CONTENT_TYPE = 'application/protobuf; delimited=varint'
PROTO_ACCEPT = 'application/protobuf'

# This stream must not be compressed, or it stops being a stream. Open WebUI wraps the whole
# app in starlette-compress, which lists `application/protobuf` as compressible and exempts
# only `text/event-stream`. Its streaming compressors write each chunk into zstd/brotli/gzip
# without flushing, so a few kilobytes of frames sit in the compressor until the response
# closes and arrive as one read -- for every client that sends Accept-Encoding, which is every
# browser and Node's fetch. Measured 2026-10-05: one read, zero spread, against ~90 reads for
# the same generation as SSE. It hit both routes, `/ollama/v1/...` included, because the
# middleware sits above both. The saving it bought was ~400 bytes on a 1 KB stream.
#
# Removed by content type rather than by disabling the middleware, which would uncompress the
# web UI's assets too. Done at import because the list is module state, read per response;
# middleware.py imports this module, so it runs before the first request.
try:
    from starlette_compress import remove_compress_type

    remove_compress_type(PROTO_ACCEPT)
except ImportError:  # no compression middleware in this build, so nothing to undo
    pass


def _uvarint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | 0x80 if n else b)
        if not n:
            return bytes(out)


def _tag(field: int, wire: int) -> bytes:
    return _uvarint(field << 3 | wire)


def _string(field: int, s: str | None) -> bytes:
    # proto3 makes absent and empty the same value, and omitting them is most of why a delta
    # frame costs a handful of bytes.
    if not s:
        return b''
    raw = s.encode('utf-8')
    return _tag(field, WIRE_BYTES) + _uvarint(len(raw)) + raw


def _uint(field: int, v: int | None) -> bytes:
    if not v:
        return b''
    return _tag(field, WIRE_VARINT) + _uvarint(max(int(v), 0))


def _optional_uint(field: int, v: int | None) -> bytes:
    # For a field declared `optional`: a zero is written, so it stays distinct from absent.
    if v is None:
        return b''
    return _tag(field, WIRE_VARINT) + _uvarint(max(int(v), 0))


def _message(field: int, msg: bytes) -> bytes:
    return _tag(field, WIRE_BYTES) + _uvarint(len(msg)) + msg


def _frame(field: int, msg: bytes) -> bytes:
    """Wrap a message as a Frame and prefix the whole thing with its length, which is how a
    protobuf stream is delimited -- protobuf's own convention, what parseDelimitedFrom reads."""
    body = _message(field, msg)
    return _uvarint(len(body)) + body


def start_frame(id: str = '', model: str = '', created: int = 0, fingerprint: str = '',
                role: str = 'assistant') -> bytes:
    m = _string(START_ID, id) + _string(START_MODEL, model)
    m += _uint(START_CREATED, created if created and created > 0 else 0)
    m += _string(START_FINGERPRINT, fingerprint) + _string(START_ROLE, role)
    return _frame(FRAME_START, m)


def delta_frame(content: str = '', reasoning: str = '', index: int = 0,
                tool_calls: list | None = None, completion_tokens: int | None = None) -> bytes:
    m = _string(DELTA_CONTENT, content) + _string(DELTA_REASONING, reasoning)
    m += _uint(DELTA_INDEX, index)
    m += _optional_uint(DELTA_COMPLETION_TOKENS, completion_tokens)
    for tc in tool_calls or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get('function') or {}
        body = _string(FUNCTION_NAME, fn.get('name') or '')
        body += _string(FUNCTION_ARGUMENTS, fn.get('arguments') or '')
        call = _string(TOOLCALL_ID, tc.get('id') or '')
        call += _uint(TOOLCALL_INDEX, tc.get('index') or 0)
        call += _string(TOOLCALL_TYPE, tc.get('type') or '')
        if body:
            call += _message(TOOLCALL_FUNCTION, body)
        m += _message(DELTA_TOOL_CALLS, call)
    return _frame(FRAME_DELTA, m)


def end_frame(finish_reason: str = '', prompt_tokens: int = 0, completion_tokens: int = 0,
              cached_tokens: int | None = None) -> bytes:
    m = _string(END_FINISH_REASON, finish_reason)
    m += _uint(END_PROMPT_TOKENS, prompt_tokens) + _uint(END_COMPLETION_TOKENS, completion_tokens)
    m += _optional_uint(END_CACHED_TOKENS, cached_tokens)
    return _frame(FRAME_END, m)


def event_frame(payload_json: str) -> bytes:
    """Anything in the stream that is not part of the completion, carried verbatim."""
    return _frame(FRAME_EVENT, _string(EVENT_JSON, payload_json))


def accepts_proto_stream(values: list[str]) -> tuple[bool, bool]:
    """Decide from a request's Accept headers whether to answer this stream as protobuf.

    Returns (wants_protobuf, understands_event_frames).

    The negotiation is the one ollama implements in `middleware/protostream.go`, and it is
    reimplemented rather than shared only because these are different languages: protobuf is
    served only when the client names `application/protobuf` exactly with q > 0 and does not
    rank `text/event-stream` above it. A wildcard never selects protobuf -- every client that
    has never heard of this sends one -- but wildcards do count when scoring SSE. A tie goes to
    protobuf, since the client named it explicitly to get here. A malformed q drops that range
    rather than guessing.

    `events=1` is this route's own parameter and ollama has no use for it. An `Event` frame is
    field 4 of the Frame oneof, so a client generated from a copy of the schema older than that
    field skips it -- silently, which is the one failure this stream must not have, since the
    frame it would skip is the provenance one. So a client says it can read them, and one that
    does not gets SSE on this route.
    """
    proto = sse = 0.0
    seen_proto = False
    events_ok = False
    sse_specificity = -1

    for header in values:
        for part in header.split(','):
            part = part.strip()
            if not part:
                continue
            bits = part.split(';')
            media_range = bits[0].strip().lower()
            q = 1.0
            malformed = False
            wants_events = False
            for param in bits[1:]:
                name, _, value = param.partition('=')
                name = name.strip().lower()
                if name == 'events':
                    wants_events = value.strip() in ('1', 'true')
                    continue
                if name != 'q':
                    continue  # a media type parameter, or an accept extension
                try:
                    v = float(value.strip())
                except ValueError:
                    malformed = True
                    break
                if v < 0 or v > 1:
                    malformed = True
                    break
                q = v
            if malformed:
                continue

            # Specificity, so the most precise match decides, as in RFC 9110 12.5.1.
            if media_range == PROTO_ACCEPT:
                if not seen_proto or q > proto:
                    proto, seen_proto = q, True
                if wants_events:
                    events_ok = True
            elif media_range == 'text/event-stream':
                if sse_specificity < 2 or (sse_specificity == 2 and q > sse):
                    sse, sse_specificity = q, 2
            elif media_range == 'text/*':
                if sse_specificity < 1 or (sse_specificity == 1 and q > sse):
                    sse, sse_specificity = q, 1
            elif media_range == '*/*':
                if sse_specificity < 0 or (sse_specificity == 0 and q > sse):
                    sse, sse_specificity = q, 0

    return (seen_proto and proto > 0 and proto >= sse), events_ok


class ChatProtoEncoder:
    """Turns the OpenAI-shaped chunks of this route's stream into protobuf frames.

    It is fed whatever the SSE reader would have been fed -- a parsed `data:` payload, or the
    string `[DONE]` -- and emits zero or more frames for each. `Start` is held back until the
    first payload, because that is where its invariant fields come from, and `End` is emitted
    once by `finish()` whether or not the backend sent a finish chunk: a stream that stops
    without one is a truncated stream, and a client that reads frames cannot see `[DONE]`.
    """

    def __init__(self, dumps):
        self._dumps = dumps
        self._started = False
        self._finished = False
        self._finish_reason = ''
        self._usage: dict | None = None

    def _start(self, chunk: dict) -> bytes:
        self._started = True
        role = ''
        for choice in chunk.get('choices') or []:
            if isinstance(choice, dict):
                role = ((choice.get('delta') or {}).get('role')) or role
        return start_frame(
            id=str(chunk.get('id') or ''),
            model=str(chunk.get('model') or ''),
            created=int(chunk.get('created') or 0),
            fingerprint=str(chunk.get('system_fingerprint') or ''),
            role=role or 'assistant',
        )

    def encode(self, payload) -> bytes:
        """Frames for one payload. `payload` is a parsed dict, or a str for a raw line."""
        if isinstance(payload, str):
            if payload.strip() == '[DONE]':
                return b''  # End replaces it
            return event_frame(payload)

        if not isinstance(payload, dict):
            return event_frame(self._dumps(payload))

        # Not a completion chunk: sources, an arena's selected_model_id, an in-band provider
        # error, a pipe's own frame, or whatever a filter rewrote one into. Carried whole, so a
        # client sees everything the SSE reader would have seen.
        if not isinstance(payload.get('choices'), list):
            if isinstance(payload.get('usage'), dict) and len(payload) == 1:
                self._usage = payload['usage']  # goes on End rather than out as an event
                return b''
            return event_frame(self._dumps(payload))

        out = b''
        if not self._started:
            out += self._start(payload)

        # A chunk carries usage twice over for two different reasons: on the last chunk it is
        # the final count, and on an earlier one it is the running count, which is how ollama
        # and vLLM report stream_options.continuous_usage_stats. Telling them apart is what
        # keeps the running count alive through this conversion.
        usage = payload.get('usage') if isinstance(payload.get('usage'), dict) else None
        final = any(isinstance(c, dict) and c.get('finish_reason') for c in payload['choices'])
        running = None
        if usage is not None:
            if final or not payload['choices']:
                self._usage = usage
            else:
                self._usage = usage  # keep the newest; End wants the last one either way
                v = usage.get('completion_tokens')
                running = v if isinstance(v, int) else None

        for choice in payload['choices']:
            if not isinstance(choice, dict):
                continue
            if choice.get('finish_reason'):
                self._finish_reason = str(choice['finish_reason'])
            delta = choice.get('delta') or {}
            if not isinstance(delta, dict):
                continue
            content = delta.get('content')
            content = content if isinstance(content, str) else ''
            reasoning = delta.get('reasoning_content') or delta.get('reasoning') or ''
            reasoning = reasoning if isinstance(reasoning, str) else ''
            tool_calls = delta.get('tool_calls') or []
            tool_calls = tool_calls if isinstance(tool_calls, list) else []
            own = delta.get('completion_tokens')
            count = own if isinstance(own, int) else running
            if not content and not reasoning and not tool_calls and count is None:
                continue
            out += delta_frame(
                content=content,
                reasoning=reasoning,
                index=choice.get('index') or 0,
                tool_calls=tool_calls,
                completion_tokens=count,
            )
        return out

    def finish(self) -> bytes:
        if self._finished:
            return b''
        self._finished = True
        out = b''
        if not self._started:
            out += start_frame()
        u = self._usage or {}
        cached = None
        details = u.get('prompt_tokens_details')
        if isinstance(details, dict) and isinstance(details.get('cached_tokens'), int):
            cached = details['cached_tokens']
        elif isinstance(u.get('prompt_eval_cached_count'), int):
            cached = u['prompt_eval_cached_count']
        return out + end_frame(
            finish_reason=self._finish_reason,
            prompt_tokens=int(u.get('prompt_tokens') or 0),
            completion_tokens=int(u.get('completion_tokens') or 0),
            cached_tokens=cached,
        )
