"""Score the controller's rules against the simulator's labels, with no LLM.

This is the number a trained classifier has to beat.

Usage: python scripts/eval_controller_labels.py [--split val]
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inferx.controller import core_facts, may_replace, worth_remembering  # noqa: E402
from inferx.locomo import sessions  # noqa: E402


def f1(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(score, 3), "n": tp + fn}


parser = argparse.ArgumentParser()
parser.add_argument("--split", default="val", choices=["train", "val"])
args = parser.parse_args()
samples = json.loads((ROOT / f"data/assistant/{args.split}.json").read_text(encoding="utf-8"))

store, core, replace = Counter(), Counter(), Counter()
sensitive_kept = sensitive_total = 0
by_source = {}
for sample in samples:
    messages = {message.id: message for _, _, group in sessions(sample) for message in group}
    for message_id, label in sample["labels"].items():
        message = messages[message_id]
        kept = worth_remembering(message)
        store[("tp" if label["store"] else "fp") if kept else ("fn" if label["store"] else "tn")] += 1
        tally = by_source.setdefault(label["source"], Counter())
        tally["correct" if kept == label["store"] else "wrong"] += 1
        if label["sensitive"]:
            sensitive_total += 1
            sensitive_kept += kept
        found = bool(core_facts(message))
        wanted = bool(label["core"])
        core[("tp" if wanted else "fp") if found else ("fn" if wanted else "tn")] += 1
        if label["replaces"]:
            replace["allowed" if may_replace(message.text, messages[label["replaces"]].text) else "blocked"] += 1

print("worth remembering:", f1(store["tp"], store["fp"], store["fn"]))
print("  correct by source:", {key: round(value["correct"] / sum(value.values()), 3) for key, value in by_source.items()})
print(f"  one-time codes kept: {sensitive_kept} of {sensitive_total}")
print("stable (core) facts:", f1(core["tp"], core["fp"], core["fn"]))
print("real updates the rule allows:", dict(replace))
