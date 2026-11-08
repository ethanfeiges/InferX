"""Check that a model can find a planted fact in a prompt near the 4,096-token budget.

Usage: python long_context_test.py <base_url> <model> [approx_prompt_tokens]
"""
import json
import sys
import time
import urllib.request

base_url, model = sys.argv[1].rstrip("/"), sys.argv[2]
target = int(sys.argv[3]) if len(sys.argv) > 3 else 3000

filler = "On that day the group walked along the river, talked about work, and made plans for next month. "
memories = [filler] * (target // 22)
memories[len(memories) // 2] = "Important note: Priya's sister is allergic to walnuts. "
prompt = (
    "Here are stored memories:\n"
    + "".join(memories)
    + "\n\nUsing only those memories, what is Priya's sister allergic to? Answer in one short sentence."
)
request = urllib.request.Request(
    f"{base_url}/chat/completions",
    data=json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 60,
        }
    ).encode(),
    headers={"Content-Type": "application/json"},
)
for attempt in range(1, 4):
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=240) as response:
        data = json.load(response)
    reply = data["choices"][0]["message"]["content"].strip()
    usage = data.get("usage") or {}
    print(
        f"try {attempt}: {time.perf_counter() - start:.1f}s "
        f"prompt_tokens={usage.get('prompt_tokens')} "
        f"completion_tokens={usage.get('completion_tokens')} "
        f"found={'walnut' in reply.lower()} reply={reply[:120]!r}",
        flush=True,
    )
