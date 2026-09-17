"""Checks for the protobuf encoding of Open WebUI's own chat stream (utils/chat_proto.py).

The encoder is written by hand, so that it needs no dependency the overlaid deployment cannot
declare. That is only acceptable if something proves it agrees with the schema, which is what
this does: every frame it produces is decoded by a decoder GENERATED from
`middleware/chat.proto`, and the values that come back are compared with what went in. A
hand-written encoder that has never been decoded by a real implementation is a guess.

Run it where the code runs -- the generated decoder is built here from the .proto, so
grpcio-tools has to be available:

  # in the notebook venv on the host, which has grpcio-tools:
  SLOP_CHAT_PROTO=~/git/ollama/middleware/chat.proto \\
      python3 ~/git/owui-fork/backend/slop_checks/chat_proto_check.py

It deliberately does not import Open WebUI: utils/chat_proto.py has no Open WebUI imports, so
the checks run anywhere, and keeping it that way is itself worth checking.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Loaded by path, not as open_webui.utils.chat_proto: importing the package pulls in typer and
# the rest of Open WebUI. That it loads this way is itself the check that the module has no Open
# WebUI imports, which is what lets these run outside the container.
import importlib.util

_spec = importlib.util.spec_from_file_location(
    'slop_chat_proto',
    Path(__file__).resolve().parent.parent / 'open_webui' / 'utils' / 'chat_proto.py',
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

ChatProtoEncoder = _mod.ChatProtoEncoder
accepts_proto_stream = _mod.accepts_proto_stream
delta_frame = _mod.delta_frame
end_frame = _mod.end_frame
event_frame = _mod.event_frame
start_frame = _mod.start_frame

failures = []


def check(name, cond, detail=''):
    print(('ok   ' if cond else 'FAIL ') + name + (f'  {detail}' if not cond else ''))
    if not cond:
        failures.append(name)


# --------------------------------------------------------------- the generated decoder

proto = Path(os.environ.get('SLOP_CHAT_PROTO', Path.home() / 'git/ollama/middleware/chat.proto'))
if not proto.exists():
    print(f'FAIL cannot find chat.proto at {proto}; set SLOP_CHAT_PROTO')
    raise SystemExit(1)

tmp = tempfile.mkdtemp()
subprocess.run([sys.executable, '-m', 'grpc_tools.protoc', f'-I{proto.parent}',
                f'--python_out={tmp}', str(proto)], check=True)
sys.path.insert(0, tmp)
import chat_pb2  # noqa: E402


def frames(buf: bytes):
    """Read a length-delimited stream back, the way a client does."""
    out, off = [], 0
    while off < len(buf):
        n, shift = 0, 0
        while True:
            b = buf[off]
            off += 1
            n |= (b & 0x7F) << shift
            shift += 7
            if not b & 0x80:
                break
        f = chat_pb2.Frame()
        f.ParseFromString(buf[off:off + n])
        off += n
        out.append(f)
    return out


# --------------------------------------------------------------- the frames, one at a time

f = frames(start_frame(id='chatcmpl-1', model='m', created=7, role='assistant'))
check('start decodes', len(f) == 1 and f[0].WhichOneof('kind') == 'start')
check('start fields survive',
      f[0].start.id == 'chatcmpl-1' and f[0].start.model == 'm'
      and f[0].start.created == 7 and f[0].start.role == 'assistant')

f = frames(delta_frame(content='hi', reasoning='because'))
check('delta text survives', f[0].delta.content == 'hi' and f[0].delta.reasoning == 'because')

f = frames(delta_frame(content='x', completion_tokens=0))
check('a running count of zero is present, not absent',
      f[0].delta.HasField('completion_tokens') and f[0].delta.completion_tokens == 0)
f = frames(delta_frame(content='x'))
check('no running count means absent', not f[0].delta.HasField('completion_tokens'))

f = frames(delta_frame(tool_calls=[{'id': 'call_1', 'index': 0, 'type': 'function',
                                    'function': {'name': 'get_weather',
                                                 'arguments': '{"city":"Paris"}'}}]))
tc = f[0].delta.tool_calls
check('tool call survives',
      len(tc) == 1 and tc[0].id == 'call_1' and tc[0].type == 'function'
      and tc[0].function.name == 'get_weather'
      and tc[0].function.arguments == '{"city":"Paris"}')

f = frames(end_frame(finish_reason='stop', prompt_tokens=20, completion_tokens=10, cached_tokens=0))
check('end fields survive',
      f[0].end.finish_reason == 'stop' and f[0].end.prompt_tokens == 20
      and f[0].end.completion_tokens == 10)
check('a cold prefill stays distinct from unreported',
      f[0].end.HasField('cached_tokens') and f[0].end.cached_tokens == 0)
check('unreported cached tokens are absent',
      not frames(end_frame(finish_reason='stop'))[0].end.HasField('cached_tokens'))

payload = json.dumps({'sources': [{'source': {'name': 'notes.pdf'}}]})
f = frames(event_frame(payload))
check('event carries its json verbatim',
      f[0].WhichOneof('kind') == 'event' and f[0].event.json == payload)

f = frames(delta_frame(content='héllo → 世界'))
check('non-ascii survives the length prefix', f[0].delta.content == 'héllo → 世界')

# --------------------------------------------------------------- a whole stream

enc = ChatProtoEncoder(json.dumps)
buf = b''
buf += enc.encode({'sources': [{'source': {'name': 'notes.pdf'}}]})
buf += enc.encode({'id': 'c1', 'model': 'm', 'created': 3,
                   'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': 'Hel'}}]})
buf += enc.encode({'id': 'c1', 'choices': [{'index': 0, 'delta': {'content': 'lo'}}],
                   'usage': {'completion_tokens': 2}})
buf += enc.encode({'id': 'c1', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}],
                   'usage': {'prompt_tokens': 20, 'completion_tokens': 2,
                             'prompt_tokens_details': {'cached_tokens': 4}}})
buf += enc.encode('[DONE]')
buf += enc.finish()
f = frames(buf)
kinds = [x.WhichOneof('kind') for x in f]
check('a whole stream frames in order', kinds == ['event', 'start', 'delta', 'delta', 'end'], str(kinds))
check('sources arrive before the first token', kinds[0] == 'event')
check('text reassembles', ''.join(x.delta.content for x in f if x.WhichOneof('kind') == 'delta') == 'Hello')
check('the running count rides the delta that had it',
      f[3].delta.HasField('completion_tokens') and f[3].delta.completion_tokens == 2)
check('final usage lands on end',
      f[4].end.prompt_tokens == 20 and f[4].end.completion_tokens == 2
      and f[4].end.cached_tokens == 4 and f[4].end.finish_reason == 'stop')
check('[DONE] produces nothing; end replaces it', kinds.count('end') == 1)

enc = ChatProtoEncoder(json.dumps)
buf = enc.encode({'selected_model_id': 'arena-pick'}) + enc.finish()
f = frames(buf)
check('an unrecognised payload becomes an event rather than vanishing',
      [x.WhichOneof('kind') for x in f] == ['event', 'start', 'end']
      and json.loads(f[0].event.json)['selected_model_id'] == 'arena-pick')

enc = ChatProtoEncoder(json.dumps)
f = frames(enc.encode([1, 2, 3]) + enc.finish())
check('a payload that is not an object at all is still carried',
      f[0].WhichOneof('kind') == 'event' and json.loads(f[0].event.json) == [1, 2, 3])

enc = ChatProtoEncoder(json.dumps)
f = frames(enc.encode('not json at all') + enc.finish())
check('an unparseable line is carried verbatim',
      f[0].WhichOneof('kind') == 'event' and f[0].event.json == 'not json at all')

enc = ChatProtoEncoder(json.dumps)
f = frames(enc.encode({'error': {'detail': 'upstream exploded'}}) + enc.finish())
check('an in-band error is carried, not swallowed',
      f[0].WhichOneof('kind') == 'event' and 'exploded' in f[0].event.json)

enc = ChatProtoEncoder(json.dumps)
f = frames(enc.finish())
check('a stream with no chunks still ends', [x.WhichOneof('kind') for x in f] == ['start', 'end'])

# --------------------------------------------------------------- negotiation

cases = [
    ('application/protobuf, text/event-stream;q=0.9', True, False),
    ('application/protobuf; events=1, text/event-stream;q=0.9', True, True),
    ('application/protobuf;events=1', True, True),
    ('text/event-stream', False, False),
    ('*/*', False, False),
    ('application/*', False, False),
    ('', False, False),
    ('application/protobuf;q=0', False, False),
    ('text/event-stream, application/protobuf;q=0.1', False, False),
    ('application/protobuf;q=0.5, text/event-stream;q=0.5', True, False),
    ('application/protobuf;q=notanumber', False, False),
    ('APPLICATION/PROTOBUF; EVENTS=1', True, True),
]
for header, want_proto, want_events in cases:
    got = accepts_proto_stream([header])
    check(f'accept {header!r} -> {want_proto},{want_events}', got == (want_proto, want_events), str(got))

check('no Accept header at all is SSE', accepts_proto_stream([]) == (False, False))

print()
print(f'{len(failures)} failure(s)' if failures else 'all checks passed')
raise SystemExit(1 if failures else 0)
