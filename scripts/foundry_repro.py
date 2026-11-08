"""Send repeated requests to a Foundry Local model and stop at the first failure.

Usage: python foundry_repro.py <base_url> <model> <stream:0|1> <count> <max_tokens> [gap_seconds]
"""
import json
import sys
import time
import urllib.error
import urllib.request

base_url, model = sys.argv[1].rstrip("/"), sys.argv[2]
stream, count, max_tokens = sys.argv[3] == "1", int(sys.argv[4]), int(sys.argv[5])
gap = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
PROMPTS = [
    "Name a color.",
    "What is 2 plus 2?",
    "Name a fruit.",
    "What is the capital of France?",
    "Name an animal.",
    "What day comes after Monday?",
]


def ask(prompt):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": stream,
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    finish = None
    text = []
    with urllib.request.urlopen(request, timeout=60) as response:
        if not stream:
            choice = json.load(response)["choices"][0]
            return choice["message"]["content"] or "", choice.get("finish_reason")
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            choices = json.loads(line[5:]).get("choices") or []
            if choices:
                text.append((choices[0].get("delta") or {}).get("content") or "")
                finish = choices[0].get("finish_reason") or finish
    return "".join(text), finish


for i in range(count):
    prompt = PROMPTS[i % len(PROMPTS)]
    start = time.perf_counter()
    try:
        reply, finish = ask(prompt)
    except Exception as error:
        detail = error.read().decode()[:300] if isinstance(error, urllib.error.HTTPError) else ""
        print(f"req {i + 1}: ERROR after {time.perf_counter() - start:.1f}s: {error!r} {detail}", flush=True)
        sys.exit(1)
    elapsed = time.perf_counter() - start
    print(f"req {i + 1}: {elapsed:.1f}s finish={finish} reply={reply.strip()[:60]!r}", flush=True)
    if not reply.strip():
        print("FAIL: empty reply", flush=True)
        sys.exit(1)
    time.sleep(gap)
print("PASS: all requests returned text")
