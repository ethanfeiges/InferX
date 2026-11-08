"""Stream one reply from Foundry Local and report rough speed.

Usage: python llm_smoke_test.py http://127.0.0.1:<port>/v1 phi-3.5-mini-instruct-qnn-npu:3

Speed counts streamed chunks, which only approximates tokens.
"""
import json
import sys
import time
import urllib.request

base_url, model = sys.argv[1].rstrip("/"), sys.argv[2]
prompt = "In two sentences, why might an AI agent keep its memory on the device?"

request = urllib.request.Request(
    f"{base_url}/chat/completions",
    data=json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 120,
            "stream": True,
        }
    ).encode(),
    headers={"Content-Type": "application/json"},
)

start = time.perf_counter()
first = None
chunks = 0
text = []
with urllib.request.urlopen(request, timeout=90) as response:
    for raw in response:
        line = raw.decode("utf-8").strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        choices = json.loads(line[5:]).get("choices") or []
        piece = (choices[0].get("delta") or {}).get("content") if choices else None
        if piece:
            first = first or time.perf_counter()
            chunks += 1
            text.append(piece)
end = time.perf_counter()

if not chunks:
    sys.exit("FAIL: no text returned")
print("Reply:", "".join(text))
print(f"Time to first token: {first - start:.2f} s")
print(f"Decode speed: {(chunks - 1) / max(end - first, 1e-6):.1f} chunks/s over {chunks} chunks")
print(f"Total: {end - start:.2f} s")
