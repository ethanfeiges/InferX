"""Combine the two controllers: the original store, kind and update heads with the retrained secret head.

The mixed controller (scripts/train_mixed_classifier.py) kept far fewer secrets on outside-written validation
people, but its update head got worse on the simulator people (update F1 0.95 -> 0.75) and barely better on
outside people (0.27 -> 0.30). The secret head is a separate yes/no head on the same MiniLM vector, so it can be
swapped in on its own.

Only the secret threshold is tuned, on the same tuning set as train_mixed_classifier.py (handwritten.json plus the
outside validation people, with the secrets the rules catch taken out): fewest kept secrets, then the best
store F1 + update F1. The store and update thresholds stay at the original controller's values.

The held-out test (data/external/20261118-195649, DeepSeek) is not read here.

Writes models/assistant_classifier_hybrid/heads.npz and metrics.json.

Usage: python scripts/build_hybrid_classifier.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import train_mixed_classifier as mixed  # noqa: E402
from inferx.memory_classifier import MemoryClassifier  # noqa: E402


def validation_people() -> list[list[dict]]:
    people = []
    for batch in sorted((ROOT / "data/external").iterdir()):
        if not batch.is_dir() or batch.name.startswith(mixed.TEST_RUN) or batch.name in mixed.EXCLUDED:
            continue
        found = sorted(batch.glob("person_*.json"))
        people += [mixed.outside(p) for p in found if len(found) == 1 or p.stem == "person_5"]
    return people


def clean(result: dict) -> dict:
    return {k: v for k, v in result.items() if k != "counts"}


def main() -> None:
    old = MemoryClassifier.load(ROOT / "models/assistant_classifier/heads.npz", mixed.embed)
    new = MemoryClassifier.load(ROOT / "models/assistant_classifier_mixed/heads.npz", mixed.embed)
    hybrid = MemoryClassifier(mixed.embed, old.store, old.kind, new.sensitive, old.replaces, old.replace_threshold,
                              old.candidates, old.store_threshold, old.sensitive_threshold)

    outside_val = validation_people()
    handwritten = json.loads((ROOT / "data/assistant/handwritten.json").read_text(encoding="utf-8"))
    tuning = [mixed.without_caught_secrets(rows) for rows in [handwritten] + outside_val]
    best = ((float("-inf"), 0.0), None)
    for threshold in (0.5, 0.7, 0.9, 0.99, 0.999):
        hybrid.sensitive_threshold = threshold
        result = mixed.merged(hybrid, tuning)
        score = (-result["counts"].get("sensitive_kept", 0), result["store"]["f1"] + result["update"]["f1"])
        print(f"secret threshold {threshold}: secrets kept {-score[0]}, store+update F1 {score[1]:.3f}")
        if score > best[0]:
            best = (score, threshold)
    hybrid.sensitive_threshold = best[1]
    print(f"chosen secret threshold {best[1]}")

    check = json.loads((ROOT / "data/assistant/handwritten_check.json").read_text(encoding="utf-8"))
    sets = {"outside_val": outside_val,
            "outside_val_after_rules": [mixed.without_caught_secrets(rows) for rows in outside_val],
            "simulator_val": mixed.simulator("val"), "simulator_val_long": mixed.simulator("val_long"),
            "handwritten_check": [check]}
    results = {"heads": {"store": "original", "kind": "original", "replaces": "original", "sensitive": "mixed"},
               "thresholds": {"sensitive": hybrid.sensitive_threshold, "store": hybrid.store_threshold,
                              "replace": hybrid.replace_threshold},
               "compared": {}}
    for name, classifier in (("original", old), ("mixed", new), ("hybrid", hybrid)):
        results["compared"][name] = {label: clean(mixed.merged(classifier, rows)) for label, rows in sets.items()}
    out = ROOT / "models/assistant_classifier_hybrid"
    out.mkdir(parents=True, exist_ok=True)
    hybrid.save(out / "heads.npz")
    (out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    for label in sets:
        print(label)
        for name, scores in results["compared"].items():
            r = scores[label]
            print(f"  {name:8} store F1 {r['store']['f1']:.3f}  update F1 {r['update']['f1']:.3f}  "
                  f"secrets kept {r['sensitive_kept']:10}  false secret alarms {r['false_sensitive']}")


if __name__ == "__main__":
    main()
