"""Check whether the model really reads the whole prompt, wherever a fact sits in it.

Plants one fact at the start, middle or end of about 1,200 tokens of filler, then asks for it.
Also reports time to first token, to reveal changes in prompt processing speed.

Usage: python scripts/context_probe.py <base_url> [rounds]
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inferx.context import ANSWER_SYSTEM, build_prompt  # noqa: E402
from inferx.llm import LocalLLM  # noqa: E402

url = sys.argv[1]
rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
tokenizer = Path.home() / ".foundry/cache/models/Microsoft/qwen2.5-1.5b-instruct-qnn-npu-3/v3/tokenizer.json"
llm = LocalLLM(url, "qwen2.5-1.5b-instruct-qnn-npu:3", tokenizer)
rng = random.Random(0)
NAMES = ["Priya", "Tomas", "Lena", "Omar", "Mei", "Jonas"]
ITEMS = ["walnuts", "shellfish", "peanuts", "kiwi", "soy", "sesame"]
TOPICS = ["river", "market", "garden", "harbor", "library", "station", "museum", "bakery"]

for round_number in range(rounds):
    for position in ("start", "middle", "end"):
        name, item = rng.choice(NAMES), rng.choice(ITEMS)
        lines = [
            f"[{rng.randint(1, 28)} May 2023] Sam: We walked by the {rng.choice(TOPICS)} and talked about "
            f"plan {rng.randint(100, 999)} for next month, which went fine."
            for _ in range(48)
        ]
        index = {"start": 0, "middle": len(lines) // 2, "end": len(lines) - 1}[position]
        lines[index] = f"[8 May 2023] {name}: My sister is allergic to {item}."
        prompt = build_prompt(f"What is {name}'s sister allergic to?", [], lines, [])
        reply = llm.complete(ANSWER_SYSTEM, prompt, max_tokens=100)
        print(
            f"round {round_number + 1} {position:<6} tokens={reply.prompt_tokens} ttft={reply.ttft_s:.1f}s "
            f"found={item in reply.text.lower()} reply={reply.text[:50]!r}",
            flush=True,
        )
