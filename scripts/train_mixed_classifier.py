"""Retrain the memory controller on the simulator plus timelines written by outside models.

Data:
  - simulator: data/assistant/train.json (as in train_memory_classifier.py)
  - outside pool: data/external/<batch>/person_*.json, written through opencode by Nemotron 3.5 Super,
    MiMo 2.6 Flash, Big Pickle and HY4 Flash (LongCat never wrote a file). One person per writer is held out
    for choosing settings (outside validation); the rest train. Batches in EXCLUDED are left out.
  - held-out test: data/external/20261118-195649 (written by DeepSeek). Never read here; scored once
    afterwards with scripts/eval_external.py.

InferX never stores secrets, so outside items marked sensitive train as "do not store, replaces nothing".

Choices made on validation only (handwritten.json plus the outside validation people):
  - how much to repeat the outside training rows (1x, 2x or 3x), since the simulator has more of them
  - the three thresholds. As in the running system (inferx/systems.py), a message the secret rules catch is never
    stored, so those secrets are taken out of the tuning timelines; a secret counts as kept only when the rules miss it
    and the controller stores it. The setting that keeps the fewest secrets wins, then the best store F1 + update F1.
    (The original training ruled out any setting that kept a secret; on outside-written validation no setting
    reaches zero, so the count is minimised instead.)

Writes models/assistant_classifier_mixed/heads.npz and metrics.json. The original controller in
models/assistant_classifier is left as it is.

Usage: python scripts/train_mixed_classifier.py
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inferx.embed import Embedder  # noqa: E402
from inferx.locomo import sessions  # noqa: E402
from inferx.sensitive import secret_kind  # noqa: E402
from inferx.memory_classifier import Item, LinearHead, MemoryClassifier, evaluate, pair_features, render, summarize  # noqa: E402

CANDIDATES = 5
TEST_RUN = "20261118-195649"
# Batches left out, with the reason. Decided from label statistics before any training.
EXCLUDED = {
    "20261122-112926_nemotron-3.5-super-vl-free":
        "repeats the same shift entry many times and labels each copy as an update (28-80 per person)",
}
embed_model = Embedder(ROOT / "models/minilm")
cache: dict[str, np.ndarray] = {}


def embed(text: str) -> np.ndarray:
    if text not in cache:
        cache[text] = embed_model(text)
    return cache[text]


def simulator(name: str) -> list[list[dict]]:
    samples = json.loads((ROOT / f"data/assistant/{name}.json").read_text(encoding="utf-8"))
    return [[{"id": m.id, "speaker": m.speaker, "text": m.text,
              **{k: s["labels"][m.id][k] for k in ("store", "kind", "replaces", "sensitive")}}
             for _, _, group in sessions(s) for m in group] for s in samples]


def outside(path: Path) -> list[dict]:
    rows = []
    for item in json.loads(path.read_text(encoding="utf-8"))["items"]:
        store, replaces = bool(item["store"]), item.get("replaces") or None
        if item.get("sensitive"):
            store, replaces = False, None
        kind = item.get("kind") if store else None
        rows.append({"id": item["id"], "speaker": item["speaker"], "text": item["text"], "store": store,
                     "kind": kind if kind in ("state", "event", "core") else ("state" if store else None),
                     "replaces": replaces, "sensitive": bool(item.get("sensitive"))})
    return rows


def pair_rows(rows: list[dict]):
    """A pair for the true target and for the other nearby stored messages (as in train_memory_classifier.py)."""
    active: list[Item] = []
    for row in rows:
        vector = embed(render(row["speaker"], row["text"]))
        if row["store"]:
            chosen = sorted(active, key=lambda item: -float(vector @ item.vector))[:CANDIDATES]
            if row["replaces"] and all(item.id != row["replaces"] for item in chosen):
                chosen += [item for item in active if item.id == row["replaces"]]
            for item in chosen:
                yield pair_features(vector, item.vector, row["text"], item.text, row["speaker"], item.speaker), \
                    "yes" if item.id == row["replaces"] else "no"
            if row["replaces"]:
                active = [item for item in active if item.id != row["replaces"]]
            active.append(Item(row["id"], row["speaker"], row["text"], vector))


def merged(classifier, timelines) -> dict:
    total = Counter()
    for rows in timelines:
        total.update(evaluate(classifier, rows)["counts"])
    return {**summarize(total), "counts": dict(total)}


def train(timelines: list[list[dict]]) -> MemoryClassifier:
    flat = [row for rows in timelines for row in rows]
    x = np.stack([embed(render(row["speaker"], row["text"])) for row in flat])
    store = LinearHead(["no", "yes"]).fit(x, ["yes" if row["store"] else "no" for row in flat])
    sensitive = LinearHead(["no", "yes"]).fit(x, ["yes" if row["sensitive"] else "no" for row in flat])
    kept = [i for i, row in enumerate(flat) if row["store"]]
    kind = LinearHead(["state", "event", "core"]).fit(x[kept], [flat[i]["kind"] for i in kept])
    pairs = [pair for rows in timelines for pair in pair_rows(rows)]
    replaces = LinearHead(["no", "yes"]).fit(np.stack([f for f, _ in pairs]), [label for _, label in pairs])
    return MemoryClassifier(embed, store, kind, sensitive, replaces, candidates=CANDIDATES)


def without_caught_secrets(rows: list[dict]) -> list[dict]:
    """Drop the secrets the rules catch: the running system never stores them, whatever the controller says."""
    return [row for row in rows if not (row["sensitive"] and secret_kind(row["text"]))]


def tune(classifier, tuning: list[list[dict]]) -> tuple[tuple, tuple]:
    tuning = [without_caught_secrets(rows) for rows in tuning]
    best = ((float("-inf"), 0.0), None)
    for sensitive_threshold in (0.5, 0.9, 0.99, 0.999):
        for store_threshold in (0.2, 0.3, 0.4, 0.5):
            for replace_threshold in (0.3, 0.5, 0.7, 0.9):
                classifier.sensitive_threshold, classifier.store_threshold = sensitive_threshold, store_threshold
                classifier.replace_threshold = replace_threshold
                result = merged(classifier, tuning)
                score = (-result["counts"].get("sensitive_kept", 0), result["store"]["f1"] + result["update"]["f1"])
                if score > best[0]:
                    best = (score, (sensitive_threshold, store_threshold, replace_threshold))
    classifier.sensitive_threshold, classifier.store_threshold, classifier.replace_threshold = best[1]
    return best


def main() -> None:
    batches = sorted(p for p in (ROOT / "data/external").iterdir()
                     if p.is_dir() and not p.name.startswith(TEST_RUN) and p.name not in EXCLUDED)
    pool_train, pool_val, used = [], [], {}
    for batch in batches:
        people = sorted(batch.glob("person_*.json"))
        if not people:
            continue
        used[batch.name] = [p.name for p in people]
        for path in people:
            # One validation person per writer: the one-person trial batches, and person 5 of a 5-person batch.
            held = len(people) == 1 or path.stem == "person_5"
            (pool_val if held else pool_train).append(outside(path))
    print(f"outside pool: {len(pool_train)} training people, {len(pool_val)} validation people, from {len(batches)} batches")
    sim_train = simulator("train")
    handwritten = json.loads((ROOT / "data/assistant/handwritten.json").read_text(encoding="utf-8"))
    tuning = [handwritten] + pool_val

    start = time.perf_counter()
    runs = {}
    for repeat in (1, 2, 3):
        classifier = train(sim_train + pool_train * repeat)
        score, thresholds = tune(classifier, tuning)
        runs[repeat] = (score, thresholds, classifier)
        print(f"outside rows x{repeat}: secrets kept {-score[0]}, store+update F1 {score[1]:.3f}, thresholds {thresholds} "
              f"({time.perf_counter() - start:.0f}s)", flush=True)
    repeat = max(runs, key=lambda k: runs[k][0])
    score, thresholds, classifier = runs[repeat]
    print(f"chosen: outside rows x{repeat}, thresholds (sensitive, store, replace) {thresholds}")

    def clean(result):
        return {k: v for k, v in result.items() if k != "counts"}

    check = json.loads((ROOT / "data/assistant/handwritten_check.json").read_text(encoding="utf-8"))
    results = {
        "outside_repeat": repeat, "tuning_secrets_kept": -score[0], "tuning_f1_sum": score[1],
        "thresholds": dict(zip(("sensitive", "store", "replace"), thresholds)),
        "batches": used, "excluded": EXCLUDED,
        "training_people": {"simulator": len(sim_train), "outside": len(pool_train)},
        "outside_val": clean(merged(classifier, pool_val)),
        "simulator_val": clean(merged(classifier, simulator("val"))),
        "simulator_val_long": clean(merged(classifier, simulator("val_long"))),
        "handwritten_check": clean(merged(classifier, [check])),
    }
    out = ROOT / "models/assistant_classifier_mixed"
    out.mkdir(parents=True, exist_ok=True)
    classifier.save(out / "heads.npz")
    (out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
