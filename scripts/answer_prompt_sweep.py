"""Compare answer prompts on planted facts: present (must answer) and absent (must decline).

Uses synthetic text plus varied wording, so no benchmark data influences the prompt choice.

Usage: python scripts/answer_prompt_sweep.py <base_url>
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inferx.context import build_prompt  # noqa: E402
from inferx.llm import LLMError, LocalLLM  # noqa: E402

url = sys.argv[1]
tokenizer = Path.home() / ".foundry/cache/models/Microsoft/qwen2.5-1.5b-instruct-qnn-npu-3/v3/tokenizer.json"
llm = LocalLLM(url, "qwen2.5-1.5b-instruct-qnn-npu:3", tokenizer)
TOPICS = ["river", "market", "garden", "harbor", "library", "station", "museum", "bakery"]
CASES = [
    ("Priya", "My sister is allergic to walnuts.", "What is Priya's sister allergic to?", "walnut"),
    ("Tomas", "I adopted a greyhound named Bolt last week.", "What is the name of Tomas's dog?", "bolt"),
    ("Lena", "I started a new job as a nurse in Lisbon.", "Where does Lena work now?", "lisbon"),
    ("Omar", "My favorite band is The Comets.", "What is Omar's favorite band?", "comets"),
    ("Mei", "We moved to Denver in March.", "Where did Mei move to?", "denver"),
    ("Jonas", "I play the cello every Sunday.", "What instrument does Jonas play?", "cello"),
]
CANDIDATES = {
    "A_no_decline": "Answer the question using only the memories. Reply with a short phrase.",
    "B_rule_only": (
        "Answer the question using only the memories. Reply with a short phrase. "
        "If no memory mentions it, reply: Not mentioned"
    ),
    "C_look_first": (
        "Read the memories carefully and find the one that answers the question. "
        "Reply with the answer as a short phrase. "
        "Only if no memory is about the question at all, reply: Not mentioned"
    ),
    "D_balanced_examples": (
        "Answer the question using only the memories, as a short phrase. "
        "If no memory is about the question, reply: Not mentioned\n"
        "Example memories: Ana: I visited Rome in May. Ben: I have two cats.\n"
        "Question: What pets does Ben have? Answer: two cats\n"
        "Question: Where did Ana go? Answer: Rome\n"
        "Question: What is Ben's job? Answer: Not mentioned"
    ),
}


def filler(rng, count):
    return [
        f"[{rng.randint(1, 28)} May 2023] Sam: We walked by the {rng.choice(TOPICS)} and talked about "
        f"plan {rng.randint(100, 999)} for next month, which went fine."
        for _ in range(count)
    ]


for name, system in CANDIDATES.items():
    rng = random.Random(0)
    found = declined_present = declined_absent = total = 0
    examples = []
    for case_name, statement, question, word in CASES:
        for count in (10, 35):
            lines = filler(rng, count)
            present = list(lines)
            present[rng.randrange(count)] = f"[8 May 2023] {case_name}: {statement}"
            try:
                reply = llm.complete(system, build_prompt(question, [], present, []), max_tokens=80).text
                absent = llm.complete(system, build_prompt(question, [], lines, []), max_tokens=80).text
            except LLMError as error:
                print(f"{name}: LLM error {error}", flush=True)
                continue
            total += 1
            found += word in reply.lower()
            declined_present += "not mentioned" in reply.lower()
            declined_absent += "not mentioned" in absent.lower()
            if len(examples) < 2:
                examples.append((reply[:50], absent[:50]))
    print(f"{name}: present found {found}/{total} (wrongly declined {declined_present}), "
          f"absent declined {declined_absent}/{total}, examples={examples}", flush=True)
