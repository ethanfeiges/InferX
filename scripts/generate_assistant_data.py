"""Write the simulated personal-assistant data to data/assistant/.

Usage: python scripts/generate_assistant_data.py [--long]

--long writes only data/assistant/val_long.json, with four months of history per user.
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inferx.assistant_sim import make_dataset, make_long_users  # noqa: E402

out = ROOT / "data/assistant"
out.mkdir(parents=True, exist_ok=True)
data = {"val_long": make_long_users()} if "--long" in sys.argv else make_dataset()
for split, samples in data.items():
    (out / f"{split}.json").write_text(json.dumps(samples, indent=1), encoding="utf-8")
    labels = [label for sample in samples for label in sample["labels"].values()]
    questions = [item for sample in samples for item in sample["qa"]]
    print(split, "users", len(samples), "items", len(labels), "questions", len(questions))
    print("  op", dict(Counter(label["op"] for label in labels)))
    print("  source", dict(Counter(label["source"] for label in labels)))
    print("  question type", dict(Counter(item["qtype"] for item in questions)))
