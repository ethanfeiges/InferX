"""Make update decisions stricter without retraining, then score once on the unseen check set.

The threshold is chosen on handwritten.json with F0.5, which counts a wrong overwrite as worse than
a missed update. handwritten_check.json has already been seen, so it is reported as optimistic.
handwritten_check2.json is the only unseen set, and nothing is tuned on it.

Usage: python scripts/tune_memory_thresholds.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inferx.embed import Embedder  # noqa: E402
from inferx.locomo import sessions  # noqa: E402
from inferx.memory_classifier import MemoryClassifier, evaluate, render, summarize  # noqa: E402

GRID = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99)
model = Embedder(ROOT / "models/minilm")
cache = {}


def embed(text):
    if text not in cache:
        cache[text] = model(text)
    return cache[text]


def load(name):
    return json.loads((ROOT / "data/assistant" / name).read_text(encoding="utf-8"))


def timeline(sample):
    rows = []
    for _, _, group in sessions(sample):
        for message in group:
            label = sample["labels"][message.id]
            rows.append({"id": message.id, "speaker": message.speaker, "text": message.text, "store": label["store"],
                         "kind": label["kind"], "replaces": label["replaces"], "sensitive": label["sensitive"]})
    return rows


def f_half(update):
    p, r = update["precision"], update["recall"]
    return 1.25 * p * r / (0.25 * p + r) if p + r else 0.0


def report(result):
    return {key: result[key] for key in ("store", "update", "kind_accuracy", "sensitive_kept", "false_sensitive")} | {
        "wrong_overwrites": result["counts"]["update_fp"], "errors": result["errors"]}


path = ROOT / "models/assistant_classifier/heads.npz"
classifier = MemoryClassifier.load(path, embed)
tuning, seen = load("handwritten.json"), load("handwritten_check.json")
unseen = load("handwritten_check2.json")
val = [timeline(sample) for sample in load("val.json")]

best = (-1.0, None)
for threshold in GRID:
    classifier.replace_threshold = threshold
    update = evaluate(classifier, tuning)["update"]
    score = f_half(update)
    print(f"threshold {threshold}: tuning precision {update['precision']} recall {update['recall']} F0.5 {score:.3f}", flush=True)
    if score >= best[0]:  # ties go to the stricter setting
        best = (score, threshold)
classifier.replace_threshold = best[1]
print("chosen replace threshold:", best[1], flush=True)

total = Counter()
for rows in val:
    total.update(evaluate(classifier, rows)["counts"])
results = {
    "thresholds": {"sensitive": classifier.sensitive_threshold, "store": classifier.store_threshold, "replace": best[1]},
    "val": {**summarize(total), "wrong_overwrites": total["update_fp"]},
    "tuning_set": report(evaluate(classifier, tuning)),
    "seen_check": report(evaluate(classifier, seen)),
    "unseen_check": report(evaluate(classifier, unseen)),
}
classifier.save(path)
(ROOT / "models/assistant_classifier/metrics_strict.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print("RESULTS")
for name in ("val", "tuning_set", "seen_check", "unseen_check"):
    print(name, {key: value for key, value in results[name].items() if key != "errors"})
print("unseen errors:")
for error in results["unseen_check"]["errors"]:
    print(" ", error["id"], "store", error["wanted_store"], "->", error["got_store"], "| replaces", error["wanted_replaces"], "->",
          error["got_replaces"], "|", error["text"][:60])
