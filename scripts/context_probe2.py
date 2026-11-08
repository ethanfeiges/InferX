"""Separate 'the prompt was cut' from 'the model read it but answered badly'.

Test A: ask for the first and last memory, which needs no reasoning.
Test B: the same planted-fact question, with and without the 'Not mentioned' option.
Test C: planted-fact recall as the number of filler lines grows.

Usage: python scripts/context_probe2.py <base_url>
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inferx.context import ANSWER_SYSTEM, build_prompt  # noqa: E402
from inferx.llm import LocalLLM  # noqa: E402

url = sys.argv[1]
tokenizer = Path.home() / ".foundry/cache/models/Microsoft/qwen2.5-1.5b-instruct-qnn-npu-3/v3/tokenizer.json"
llm = LocalLLM(url, "qwen2.5-1.5b-instruct-qnn-npu:3", tokenizer)
rng = random.Random(0)
TOPICS = ["river", "market", "garden", "harbor", "library", "station", "museum", "bakery"]
PLAIN = "Answer the question using only the memories. Reply with a short phrase."


def filler(count):
    return [
        f"[{rng.randint(1, 28)} May 2023] Sam: We walked by the {rng.choice(TOPICS)} and talked about "
        f"plan {rng.randint(100, 999)} for next month, which went fine."
        for _ in range(count)
    ]


def ask(system, lines, question):
    return llm.complete(system, build_prompt(question, [], lines, []), max_tokens=100)


lines = filler(48)
lines[0] = "[1 May 2023] Zed: The first code word is ALPHA."
lines[-1] = "[2 May 2023] Zed: The last code word is OMEGA."
for question, word in [("What is the first code word?", "alpha"), ("What is the last code word?", "omega")]:
    reply = ask(PLAIN, lines, question)
    print(f"A {question!r}: found={word in reply.text.lower()} ttft={reply.ttft_s:.1f}s reply={reply.text[:60]!r}", flush=True)

for system_name, system in [("with Not-mentioned", ANSWER_SYSTEM), ("plain", PLAIN)]:
    hits = {"start": 0, "middle": 0, "end": 0}
    for trial in range(4):
        for position in hits:
            lines = filler(48)
            index = {"start": 0, "middle": 24, "end": 47}[position]
            lines[index] = "[8 May 2023] Priya: My sister is allergic to walnuts."
            reply = ask(system, lines, "What is Priya's sister allergic to?")
            hits[position] += "walnut" in reply.text.lower()
    print(f"B {system_name}: found out of 4 {hits}", flush=True)

for count in (5, 10, 20, 35, 48):
    found = 0
    tokens = 0
    for trial in range(4):
        lines = filler(count)
        lines[count // 2] = "[8 May 2023] Priya: My sister is allergic to walnuts."
        reply = ask(ANSWER_SYSTEM, lines, "What is Priya's sister allergic to?")
        found += "walnut" in reply.text.lower()
        tokens = reply.prompt_tokens
    print(f"C filler={count} tokens~{tokens}: found {found}/4 ttft={reply.ttft_s:.1f}s", flush=True)
