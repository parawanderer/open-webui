"""Checks for the fork's ollama conversion patches, run inside the Open WebUI image.

The fork has no test suite, and these patches are overlaid onto the stock image one file at
a time, so the check runs where they run:

  docker run --rm --entrypoint python -e PYTHONPATH=/app/backend -e WEBUI_SECRET_KEY=throwaway \\
      -v <fork>/backend/open_webui/utils/payload.py:/app/backend/open_webui/utils/payload.py:ro \\
      -v <fork>/backend/open_webui/utils/response.py:/app/backend/open_webui/utils/response.py:ro \\
      -v <fork>/backend/slop_checks:/checks:ro ghcr.io/open-webui/open-webui:v0.11.4 \\
      /checks/ollama_conversion_check.py

Without PYTHONPATH the import fails, and without WEBUI_SECRET_KEY importing Open WebUI refuses
to start; the value is irrelevant to these checks.
"""
import asyncio
import json

from open_webui.utils.payload import convert_payload_openai_to_ollama
from open_webui.utils.response import convert_streaming_response_ollama_to_openai

failures = []


def check(name, cond, detail=''):
    print(('ok   ' if cond else 'FAIL ') + name + (f'  {detail}' if not cond else ''))
    if not cond:
        failures.append(name)


msgs = [{'role': 'user', 'content': 'hi'}]

# max_tokens must reach ollama as options.num_predict; at the root it is ignored.
p = convert_payload_openai_to_ollama({'model': 'm', 'messages': msgs, 'max_tokens': 12})
check('max_tokens lands in options.num_predict', p.get('options', {}).get('num_predict') == 12, p)
check('max_tokens not left at the root', 'num_predict' not in p and 'max_tokens' not in p, p)

p = convert_payload_openai_to_ollama(
    {'model': 'm', 'messages': msgs, 'max_tokens': 12, 'options': {'num_predict': 50, 'temperature': 0}}
)
check('an explicit options.num_predict wins', p['options']['num_predict'] == 50, p)
check('other options kept', p['options'].get('temperature') == 0, p)

p = convert_payload_openai_to_ollama({'model': 'm', 'messages': msgs, 'max_tokens': 12, 'stop': ['x']})
check('stop and max_tokens coexist', p['options'].get('stop') == ['x'] and p['options'].get('num_predict') == 12, p)

# The request hint: passed through, or filled in from what Open WebUI knows.
p = convert_payload_openai_to_ollama({'model': 'm', 'messages': msgs, 'hint': {'use': 'agent', 'session': 's1'}})
check('a client hint passes through', p.get('hint') == {'use': 'agent', 'session': 's1'}, p)
p = convert_payload_openai_to_ollama(
    {'model': 'm', 'messages': msgs, 'metadata': {'task': 'query_generation', 'chat_id': 'c-42'}})
check('a task call is labelled utility, with its chat as session',
      p.get('hint') == {'use': 'utility', 'session': 'owui-c-42'}, p)
p = convert_payload_openai_to_ollama(
    {'model': 'm', 'messages': msgs, 'hint': {'use': 'agent'}, 'metadata': {'task': 'x', 'chat_id': 'c-1'}})
check('a client use wins over the task label', p.get('hint') == {'use': 'agent', 'session': 'owui-c-1'}, p)
p = convert_payload_openai_to_ollama({'model': 'm', 'messages': msgs, 'metadata': {'chat_id': 'c-7'}})
check('an ordinary chat gets its session, no guessed use', p.get('hint') == {'session': 'owui-c-7'}, p)
p = convert_payload_openai_to_ollama({'model': 'm', 'messages': msgs})
check('nothing known, no hint', 'hint' not in p, p)

p = convert_payload_openai_to_ollama({'model': 'm', 'messages': msgs})
check('no max_tokens, no num_predict', 'num_predict' not in p.get('options', {}), p)

# continuous_usage_stats -> stream_metrics, only for a streamed request
opts = {'include_usage': True, 'continuous_usage_stats': True}
p = convert_payload_openai_to_ollama({'model': 'm', 'messages': msgs, 'stream': True, 'stream_options': opts})
check('continuous_usage_stats sets stream_metrics', p.get('stream_metrics') is True, p)
p = convert_payload_openai_to_ollama({'model': 'm', 'messages': msgs, 'stream': False, 'stream_options': opts})
check('not for a non-streamed request', 'stream_metrics' not in p, p)
p = convert_payload_openai_to_ollama(
    {'model': 'm', 'messages': msgs, 'stream': True, 'stream_options': {'include_usage': True}}
)
check('include_usage alone does not', 'stream_metrics' not in p, p)


# The stream converter: a running count on a mid-stream chunk becomes usage on that chunk.
class Body:
    def __init__(self, chunks):
        self.chunks = chunks

    @property
    def body_iterator(self):
        async def it():
            for c in self.chunks:
                yield json.dumps(c)

        return it()


async def convert(chunks):
    out = []
    async for line in convert_streaming_response_ollama_to_openai(Body(chunks)):
        if line.startswith('data: {'):
            out.append(json.loads(line[len('data: '):]))
    return out


with_counts = [
    {'model': 'm', 'message': {'role': 'assistant', 'content': 'The'}, 'done': False, 'prompt_eval_count': 40, 'eval_count': 3},
    {'model': 'm', 'message': {'role': 'assistant', 'content': ' answer'}, 'done': False, 'prompt_eval_count': 40, 'eval_count': 7},
    {'model': 'm', 'message': {'role': 'assistant', 'content': ''}, 'done': True, 'done_reason': 'stop',
     'prompt_eval_count': 40, 'eval_count': 9, 'prompt_eval_cached_count': 30},
]
chunks = asyncio.run(convert(with_counts))
counts = [c.get('usage', {}).get('completion_tokens') for c in chunks]
check('running count on every chunk', counts == [3, 7, 9], counts)
check('prompt tokens on mid-stream chunks', chunks[0].get('usage', {}).get('prompt_tokens') == 40, chunks[0])
check('cached tokens still on the final chunk',
      chunks[-1]['usage'].get('prompt_tokens_details', {}).get('cached_tokens') == 30, chunks[-1])

without = [dict(c) for c in with_counts]
for c in without[:2]:
    del c['eval_count'], c['prompt_eval_count']
chunks = asyncio.run(convert(without))
check('no count asked, no usage mid-stream', all('usage' not in c for c in chunks[:2]), chunks[:2])

# tool_call_deltas (the fork's ollama) become delta.tool_calls fragments, and the whole call
# that follows is not resent: OpenAI clients join fragments by index.
def msg(**m):
    return {'model': 'm', 'message': {'role': 'assistant', 'content': '', **m}, 'done': False}


def joined(chunks):
    calls = {}
    for c in chunks:
        for f in c['choices'][0]['delta'].get('tool_calls', []):
            e = calls.setdefault(f['index'], {'id': None, 'name': None, 'arguments': '', 'n': 0})
            e['id'] = e['id'] or f.get('id')
            e['name'] = e['name'] or f.get('function', {}).get('name')
            e['arguments'] += f['function'].get('arguments', '')
            e['n'] += 1
    return calls


whole = {'id': 'call_a', 'function': {'index': 0, 'name': 'write_file', 'arguments': {'path': 'p.py', 'content': 'x<y'}}}
done = {'model': 'm', 'message': {'role': 'assistant', 'content': ''}, 'done': True, 'done_reason': 'stop'}
GO = '{"path":"p.py","content":"x\\u003cy"}'  # json.Marshal of the arguments

# Go-parser path: pieces are a prefix of the whole call's JSON; the rest goes out at close.
prefix = [
    msg(content='ok'),
    msg(tool_call_deltas=[{'index': 0, 'id': 'call_a', 'function': {'name': 'write_file', 'arguments': '{"path":'}}]),
    msg(tool_call_deltas=[{'index': 0, 'function': {'arguments': '"p.py"'}}]),
    msg(tool_call_deltas=[{'index': 0, 'function': {'arguments': ',"content":"x'}}]),
    msg(tool_calls=[whole]),
    done,
]
chunks = asyncio.run(convert(prefix))
calls = joined(chunks)
check('pieces stream as fragments, rest appended at close',
      calls[0]['arguments'] == GO, calls)
check('id and name once, on the first fragment', calls[0]['id'] == 'call_a' and calls[0]['name'] == 'write_file'
      and sum(1 for c in chunks for f in c['choices'][0]['delta'].get('tool_calls', []) if 'id' in f) == 1, chunks)
check('fragment count: 3 pieces + 1 rest', calls[0]['n'] == 4, calls)
check('no finish_reason stop mid-stream', all(c['choices'][0].get('finish_reason') is None for c in chunks[:-1]),
      [c['choices'][0].get('finish_reason') for c in chunks])
check('finish_reason tool_calls at done', chunks[-1]['choices'][0]['finish_reason'] == 'tool_calls', chunks[-1])

# llama-server path: pieces are llama.cpp's bytes, the whole call a re-marshal; pieces stand.
llama = [
    msg(tool_call_deltas=[{'index': 0, 'id': 'call_a', 'function': {'name': 'write_file', 'arguments': '{"path": "p.py", '}}]),
    msg(tool_call_deltas=[{'index': 0, 'function': {'arguments': '"content": "x<y"}'}}]),
    msg(tool_calls=[whole]),
    done,
]
chunks = asyncio.run(convert(llama))
calls = joined(chunks)
check('diverging bytes: the pieces are the call', calls[0]['arguments'] == '{"path": "p.py", "content": "x<y"}', calls)
check('diverging bytes: whole call not resent', calls[0]['n'] == 2, calls)
check('no empty mid-stream chunk', len(chunks) == 3, len(chunks))

# A call with no deltas still goes out whole; a second parallel call is kept apart by index.
second = {'id': 'call_b', 'function': {'index': 1, 'name': 'ls', 'arguments': {}}}
chunks = asyncio.run(convert([
    msg(tool_call_deltas=[{'index': 0, 'id': 'call_a', 'function': {'name': 'write_file', 'arguments': '{"path":"p.py"'}}]),
    msg(tool_calls=[whole, second]),
    done,
]))
calls = joined(chunks)
check('an unstreamed call goes out whole', calls[1] == {'id': 'call_b', 'name': 'ls', 'arguments': '{}', 'n': 1}, calls)
check('the streamed one is completed', calls[0]['arguments'] == GO, calls)

# No deltas at all: unchanged behaviour.
chunks = asyncio.run(convert([msg(tool_calls=[whole]), done]))
calls = joined(chunks)
check('plain whole call unchanged', calls[0]['n'] == 1 and calls[0]['id'] == 'call_a', calls)

print(f'\n{len(failures)} failure(s)')
raise SystemExit(1 if failures else 0)
