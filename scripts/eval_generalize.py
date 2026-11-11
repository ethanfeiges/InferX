"""Measure the generalization additions on labelled data, with no LLM.

  - secret rules: secrets caught and false alarms, on the simulator's labels and on LoCoMo chat
  - owner guard: update precision and recall of the trained controller, with and without it
  - unsure band: how many update decisions phi would be asked about, and how many errors they hold

Never reads the test split. The secret rules were written while looking at misses in these same sets,
so their scores here are optimistic; messages written by someone else are the real check.

Usage: python scripts/eval_generalize.py [--band 0.2]
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inferx.embed import Embedder  # noqa: E402
from inferx.locomo import sessions  # noqa: E402
from inferx.memory_classifier import Item, MemoryClassifier, different_item, different_owner, evaluate, summarize  # noqa: E402
from inferx.sensitive import secret_kind  # noqa: E402

DATA = ROOT / "data/assistant"


def timeline(sample: dict) -> list[dict]:
    return [{"id": m.id, "speaker": m.speaker, "text": m.text,
             **{key: sample["labels"][m.id][key] for key in ("store", "kind", "replaces", "sensitive")}}
            for _, _, group in sessions(sample) for m in group]


def load_sets() -> dict[str, list[list[dict]]]:
    sets = {name: [timeline(sample) for sample in json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))]
            for name in ("val", "val_long")}
    for name in ("handwritten_check", "handwritten_check2"):
        sets[name] = [json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))]
    return sets


def secrets(sets: dict) -> None:
    print("Secret rules on their own (without the trained head)")
    for name, timelines in sets.items():
        rows = [row for rows in timelines for row in rows]
        caught = sum(1 for row in rows if row["sensitive"] and secret_kind(row["text"]))
        total = sum(row["sensitive"] for row in rows)
        alarms = sum(1 for row in rows if not row["sensitive"] and secret_kind(row["text"]))
        print(f"  {name:20} caught {caught} of {total}, false alarms {alarms} of {len(rows) - total}")
    locomo = [m.text for sample in json.loads((ROOT / "data/locomo/locomo10.json").read_text(encoding="utf-8"))
              for _, _, group in sessions(sample) for m in group]
    print(f"  {'LoCoMo chat':20} false alarms {sum(1 for text in locomo if secret_kind(text))} of {len(locomo)}")


def guards(classifier: MemoryClassifier, sets: dict) -> None:
    print("Owner guard (real updates found by the trained controller)")
    for label, chosen in (("without", ()), ("with", (different_owner,))):
        classifier.guards = chosen
        for name, timelines in sets.items():
            total = Counter()
            for rows in timelines:
                total.update(evaluate(classifier, rows)["counts"])
            update = summarize(total)["update"]
            print(f"  {label:8} {name:20} precision {update['precision']:.3f} recall {update['recall']:.3f} "
                  f"(found {total['update_tp']}, wrong {total['update_fp']}, missed {total['update_fn']})")


def unsure(classifier: MemoryClassifier, sets: dict, band: float) -> None:
    print(f"Unsure updates (score within {band} of {classifier.replace_threshold}, or named things differ)")
    classifier.guards = (different_owner,)
    for name, timelines in sets.items():
        asked = held = errors = 0
        for rows in timelines:
            active: list[Item] = []
            for row in rows:
                decision, vector = classifier.decide(row["speaker"], row["text"], active)
                if not decision.store:
                    continue
                wrong = (decision.replaces or None) != (row["replaces"] or None)
                errors += wrong
                old = next((item for item in active if item.id == decision.candidate), None)
                near = decision.candidate and abs(decision.replace_score - classifier.replace_threshold) < band
                conflict = decision.replaces and old and different_item(row["speaker"], row["text"], old)
                if near or conflict:
                    asked += 1
                    held += wrong
                if decision.replaces:
                    active = [item for item in active if item.id != decision.replaces]
                active.append(Item(row["id"], row["speaker"], row["text"], vector, decision.kind))
        print(f"  {name:20} phi asked {asked} times; those cases hold {held} of the {errors} update errors")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--band", type=float, default=0.2)
    args = parser.parse_args()
    embedder, cache = Embedder(ROOT / "models/minilm"), {}

    def embed(text):
        if text not in cache:
            cache[text] = embedder(text)
        return cache[text]

    classifier = MemoryClassifier.load(ROOT / "models/assistant_classifier/heads.npz", embed)
    sets = load_sets()
    secrets(sets)
    guards(classifier, sets)
    unsure(classifier, sets, args.band)


if __name__ == "__main__":
    main()
