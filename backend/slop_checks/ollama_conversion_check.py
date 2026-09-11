"""Checks for the fork's ollama conversion patches, run inside the Open WebUI image.

The fork has no test suite, and these patches are overlaid onto the stock image one file at
a time, so the check runs where they run:

  docker run --rm --entrypoint python -v <fork>/backend/open_webui/utils/payload.py:... \\
      ghcr.io/open-webui/open-webui:v0.11.3 /checks/ollama_conversion_check.py
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

print(f'\n{len(failures)} failure(s)')
raise SystemExit(1 if failures else 0)
