"""Train the assistant's memory classifiers on simulated data and score them against the rules.

Never reads the test split. handwritten.json tunes the thresholds, so its scores are optimistic.
handwritten_check.json is scored once, after tuning, and nothing is tuned on it.

Usage: python scripts/train_memory_classifier.py
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inferx.controller import core_facts, may_replace, worth_remembering  # noqa: E402
from inferx.embed import Embedder  # noqa: E402
from inferx.interface import Message  # noqa: E402
from inferx.locomo import sessions  # noqa: E402
from inferx.memory_classifier import (  # noqa: E402
    Item, LinearHead, MemoryClassifier, evaluate, pair_features, render, summarize,
)

CANDIDATES = 5
embed_model = Embedder(ROOT / "models/minilm")
cache: dict[str, np.ndarray] = {}


def embed(text: str) -> np.ndarray:
    if text not in cache:
        cache[text] = embed_model(text)
    return cache[text]


def timeline(sample: dict) -> list[dict]:
    rows = []
    for _, _, group in sessions(sample):
        for message in group:
            label = sample["labels"][message.id]
            rows.append({"id": message.id, "speaker": message.speaker, "text": message.text, "store": label["store"],
                         "kind": label["kind"], "replaces": label["replaces"], "sensitive": label["sensitive"]})
    return rows


def pair_rows(rows: list[dict]):
    """Yield a pair for the true target and for the other nearby stored messages."""
    active: list[Item] = []
    for row in rows:
        vector = embed(render(row["speaker"], row["text"]))
        if row["store"]:
            ranked = sorted(active, key=lambda item: -float(vector @ item.vector))
            chosen = ranked[:CANDIDATES]
            if row["replaces"] and all(item.id != row["replaces"] for item in chosen):
                chosen += [item for item in active if item.id == row["replaces"]]
            for item in chosen:
                yield pair_features(vector, item.vector, row["text"], item.text, row["speaker"], item.speaker), \
                    "yes" if item.id == row["replaces"] else "no"
            if row["replaces"]:
                active = [item for item in active if item.id != row["replaces"]]
            active.append(Item(row["id"], row["speaker"], row["text"], vector))


def rules_baseline(rows: list[dict]) -> dict:
    """Score the current rules with the same replay and the same definitions."""
    counts = Counter()
    active: list[Item] = []
    for row in rows:
        message = Message(row["id"], row["speaker"], row["text"])
        kept = worth_remembering(message) or bool(core_facts(message))
        vector = embed(render(row["speaker"], row["text"]))
        if row["store"] and kept:
            counts["store_tp"] += 1
            counts["kind_total"] += 1
            guess = "core" if core_facts(message) else "state"
            counts["kind_right"] += guess == row["kind"]
        elif kept:
            counts["store_fp"] += 1
        elif row["store"]:
            counts["store_fn"] += 1
        if row["sensitive"]:
            counts["sensitive_total"] += 1
            counts["sensitive_kept"] += kept
        target = None
        if kept and active:
            nearest = max(active, key=lambda item: float(vector @ item.vector))
            if float(vector @ nearest.vector) >= 0.85 and may_replace(row["text"], nearest.text):
                target = nearest.id
        if row["replaces"] and target == row["replaces"]:
            counts["update_tp"] += 1
        elif row["replaces"]:
            counts["update_fn"] += 1
        elif target:
            counts["update_fp"] += 1
        if kept:
            if target:
                active = [item for item in active if item.id != target]
            active.append(Item(row["id"], row["speaker"], row["text"], vector))
    for name in ("store_tp", "store_fp", "store_fn", "kind_right", "kind_total", "update_tp", "update_fp", "update_fn",
                 "sensitive_total", "sensitive_kept", "false_sensitive"):
        counts.setdefault(name, 0)
    return summarize(counts)


def merged(results: list[dict]) -> dict:
    total = Counter()
    for result in results:
        total.update(result["counts"])
    return summarize(total)


data = ROOT / "data/assistant"
train = [timeline(sample) for sample in json.loads((data / "train.json").read_text(encoding="utf-8"))]
val = [timeline(sample) for sample in json.loads((data / "val.json").read_text(encoding="utf-8"))]
handwritten = json.loads((data / "handwritten.json").read_text(encoding="utf-8"))

start = time.perf_counter()
flat = [row for rows in train for row in rows]
x = np.stack([embed(render(row["speaker"], row["text"])) for row in flat])
print(f"embedded {len(flat)} training messages in {time.perf_counter() - start:.0f}s", flush=True)

start = time.perf_counter()
store = LinearHead(["no", "yes"]).fit(x, ["yes" if row["store"] else "no" for row in flat])
sensitive = LinearHead(["no", "yes"]).fit(x, ["yes" if row["sensitive"] else "no" for row in flat])
kept = [index for index, row in enumerate(flat) if row["store"]]
kind = LinearHead(["state", "event", "core"]).fit(x[kept], [flat[index]["kind"] for index in kept])
pairs = [pair for rows in train for pair in pair_rows(rows)]
pair_x, pair_y = np.stack([features for features, _ in pairs]), [label for _, label in pairs]
replaces = LinearHead(["no", "yes"]).fit(pair_x, pair_y)
print(f"trained on {len(pairs)} pairs ({pair_y.count('yes')} real updates) in {time.perf_counter() - start:.0f}s", flush=True)

classifier = MemoryClassifier(embed, store, kind, sensitive, replaces, candidates=CANDIDATES)
check = json.loads((data / "handwritten_check.json").read_text(encoding="utf-8"))
best = (-1.0, None)
for sensitive_threshold in (0.5, 0.9, 0.99, 0.999):
    for store_threshold in (0.2, 0.3, 0.4, 0.5):
        for replace_threshold in (0.3, 0.5, 0.7, 0.9):
            classifier.sensitive_threshold = sensitive_threshold
            classifier.store_threshold = store_threshold
            classifier.replace_threshold = replace_threshold
            result = evaluate(classifier, handwritten)
            # A kept secret is a privacy failure, so a setting that keeps any is never chosen.
            score = -1.0 if result["counts"]["sensitive_kept"] else result["store"]["f1"] + result["update"]["f1"]
            if score > best[0]:
                best = (score, (sensitive_threshold, store_threshold, replace_threshold))
classifier.sensitive_threshold, classifier.store_threshold, classifier.replace_threshold = best[1]
print("chosen thresholds (sensitive, store, replace):", best[1], flush=True)


def without_counts(result):
    return {key: value for key, value in result.items() if key != "counts"}


results = {
    "thresholds": {"sensitive": best[1][0], "store": best[1][1], "replace": best[1][2]},
    "training_messages": len(flat),
    "training_pairs": len(pairs),
    "val": {"classifier": merged([evaluate(classifier, rows) for rows in val]),
            "rules": rules_baseline([row for rows in val for row in rows])},
    "tuning_set": {"classifier": without_counts(evaluate(classifier, handwritten)), "rules": rules_baseline(handwritten)},
    "fresh_check": {"classifier": without_counts(evaluate(classifier, check)), "rules": rules_baseline(check)},
}
out = ROOT / "models/assistant_classifier"
out.mkdir(parents=True, exist_ok=True)
classifier.save(out / "heads.npz")
(out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps(results, indent=1))
