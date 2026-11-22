"""Score the memory controller and the secret rules on timelines written by an outside model.

The timelines in data/external/<run>/person_*.json were written by a different model through opencode
(scripts/external_writer.ps1), from data/external/brief.md, without access to this repo. Nothing here is
tuned on them: this runs once, with the trained controller and rules as they are.

One policy difference from the writer's labels: InferX never stores secrets, so items the writer marked
sensitive count as "do not store" and "replaces nothing" when scoring. The count of such items is printed.

With --galaxy the controller runs on the Galaxy phone's NPU (scripts/build_galaxy_models.py must have pushed
its heads). The report also counts secrets the full system stores: the rules and the controller both miss them.

Usage: python scripts/eval_external.py data/external/<run> [--heads <heads.npz>] [--out eval.json] [--galaxy]
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inferx.embed import Embedder  # noqa: E402
from inferx.memory_classifier import MemoryClassifier, different_owner, evaluate, summarize  # noqa: E402
from inferx.sensitive import secret_kind  # noqa: E402


def rows(person: dict) -> tuple[list[dict], int]:
    out, aligned = [], 0
    for item in person["items"]:
        store, replaces = bool(item["store"]), item.get("replaces") or None
        if item.get("sensitive") and (store or replaces):
            aligned += 1
            store, replaces = False, None
        out.append({"id": item["id"], "speaker": item["speaker"], "text": item["text"], "store": store,
                    "kind": item.get("kind") if store else None, "replaces": replaces,
                    "sensitive": bool(item.get("sensitive"))})
    return out, aligned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--heads", type=Path, default=ROOT / "models/assistant_classifier/heads.npz",
                        help="the controller to score; models/assistant_classifier_mixed/heads.npz for the retrained one")
    parser.add_argument("--out", default="eval.json", help="file name for the report, inside the run folder")
    parser.add_argument("--galaxy", action="store_true", help="run the controller on the phone's NPU")
    args = parser.parse_args()
    people = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(args.run.glob("person_*.json"))}
    embedder, cache = Embedder(ROOT / "models/minilm"), {}

    def embed(text):
        if text not in cache:
            cache[text] = embedder(text)
        return cache[text]

    if args.galaxy:
        from inferx.galaxy import load_phone_controller
        controller = args.heads.parent.name
        prefix = "" if controller == "assistant_classifier" else controller.removeprefix("assistant_classifier_") + "_"
        classifier = load_phone_controller(args.heads, ROOT / "models/minilm_fp32/tokenizer.json", prefix)
    else:
        classifier = MemoryClassifier.load(args.heads, embed)
    report = {"run": str(args.run), "heads": str(args.heads), "device": "galaxy" if args.galaxy else "laptop cpu",
              "people": {}}
    leaked = Counter()
    totals = {"plain": Counter(), "owner_guard": Counter()}
    secret_rows = []
    for name, person in people.items():
        timeline, aligned = rows(person)
        secret_rows += timeline
        if args.galaxy:
            classifier.prefetch([SimpleNamespace(speaker=r["speaker"], text=r["text"]) for r in timeline])
        # The running system never stores what the secret rules catch, so leaks are counted without those.
        filtered = [r for r in timeline if not (r["sensitive"] and secret_kind(r["text"]))]
        entry = {"owner": person["owner"], "items": len(timeline), "secrets_set_to_not_stored": aligned}
        for label, guards in (("plain", ()), ("owner_guard", (different_owner,))):
            classifier.guards = guards
            result = evaluate(classifier, timeline)
            totals[label].update(result["counts"])
            leaked[label] += evaluate(classifier, filtered)["counts"]["sensitive_kept"]
            entry[label] = {k: v for k, v in result.items() if k != "counts"}
        report["people"][name] = entry
        print(f"{name} ({person['owner']}): update {entry['plain']['update']}, store {entry['plain']['store']}, "
              f"secrets kept {entry['plain']['sensitive_kept']}")
    report["trained"] = {label: summarize(counts) for label, counts in totals.items()}
    secrets = [r for r in secret_rows if r["sensitive"]]
    report["secrets_stored_by_system"] = {label: f"{n} of {len(secrets)}" for label, n in leaked.items()}
    caught = [r for r in secrets if secret_kind(r["text"])]
    alarms = [r for r in secret_rows if not r["sensitive"] and secret_kind(r["text"])]
    report["secret_rules"] = {"caught": f"{len(caught)} of {len(secrets)}",
                              "missed": [r["text"] for r in secrets if not secret_kind(r["text"])],
                              "false_alarms": f"{len(alarms)} of {len(secret_rows) - len(secrets)}",
                              "false_alarm_texts": [r["text"] for r in alarms]}
    print(json.dumps({k: report[k] for k in ("trained", "secret_rules", "secrets_stored_by_system")}, indent=1, ensure_ascii=False))
    (args.run / args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print("saved", args.run / args.out)


if __name__ == "__main__":
    main()
