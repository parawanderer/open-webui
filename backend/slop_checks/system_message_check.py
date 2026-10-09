"""Check for the fork's merge_system_messages patch, run inside the Open WebUI image.

Upstream rebuilt every system message as a string from its FIRST text part, which dropped the
other parts and Anthropic's cache_control breakpoint on every /api/chat/completions request.

  docker run --rm --entrypoint python -e PYTHONPATH=/app/backend -e WEBUI_SECRET_KEY=throwaway \\
      -v <fork>/backend/open_webui/utils/misc.py:/app/backend/open_webui/utils/misc.py:ro \\
      -v <fork>/backend/slop_checks:/checks:ro ghcr.io/open-webui/open-webui:v0.11.4 \\
      /checks/system_message_check.py
"""
from open_webui.utils.misc import merge_system_messages

failures = []


def check(name, cond, detail=''):
    print(('ok   ' if cond else 'FAIL ') + name + (f'  {detail}' if not cond else ''))
    if not cond:
        failures.append(name)


cc = {'type': 'ephemeral'}
user = {'role': 'user', 'content': 'hi'}

parts = [{'type': 'text', 'text': 'AAA'}, {'type': 'text', 'text': 'BBB', 'cache_control': cc}]
out = merge_system_messages([{'role': 'system', 'content': parts}, user])
check('a single system message keeps all its parts', out[0]['content'] == parts, out)
check('and its cache_control', out[0]['content'][-1].get('cache_control') == cc, out)
check('the other messages follow unchanged', out[1:] == [user], out)

out = merge_system_messages([user, {'role': 'system', 'content': parts}])
check('a system message after another is moved to the front', out[0]['role'] == 'system' and out[1] == user, out)

out = merge_system_messages([{'role': 'system', 'content': 'one'}, user, {'role': 'system', 'content': 'two'}])
check('all-string system messages are joined as before', out == [{'role': 'system', 'content': 'one\ntwo'}, user], out)

out = merge_system_messages([{'role': 'system', 'content': 'one'}, {'role': 'system', 'content': parts}, user])
check(
    'a string merged with parts keeps every part, the string first with its newline',
    out[0]['content'] == [{'type': 'text', 'text': 'one\n'}, *parts] and out[1:] == [user],
    out,
)
out = merge_system_messages([{'role': 'system', 'content': parts}, {'role': 'system', 'content': 'RAG'}, user])
check(
    'a string after parts gets its newline in front, and the parts are untouched',
    out[0]['content'] == [*parts, {'type': 'text', 'text': '\nRAG'}],
    out,
)

from open_webui.utils.payload import convert_messages_openai_to_ollama

ollama = convert_messages_openai_to_ollama(merge_system_messages([{'role': 'system', 'content': parts}, {'role': 'system', 'content': 'RAG'}, user]))
check('ollama still sees the merged system text apart', ollama[0]['content'] == 'AAABBB\nRAG', ollama)

out = merge_system_messages([{'role': 'system', 'content': ''}, user])
check('an empty system message is dropped as before', out == [user], out)
out = merge_system_messages([{'role': 'system', 'content': []}, user])
check('an empty parts list is dropped too', out == [user], out)

out = merge_system_messages([user])
check('no system message, nothing changes', out == [user], out)

print('PASS' if not failures else f'{len(failures)} FAILED')
raise SystemExit(1 if failures else 0)
