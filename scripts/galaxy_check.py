"""Check that the controller makes the same memory decisions on the Galaxy phone as on the laptop.

No LLM: replays validation users' messages through the trained controller twice, once with MiniLM and its
4 heads on the laptop's NPU (as in the final run) and once on the phone's NPU. Both see the same stored
items at every step (the laptop's), so any difference is the device, not a different history. Also scores
each device against the labels with evaluate(), letting each follow its own decisions.

Never reads the test split.

Usage: python scripts/galaxy_check.py [--sets val,val_long] [--users 2]
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inferx.galaxy import load_phone_controller  # noqa: E402
from inferx.locomo import sessions  # noqa: E402
from inferx.memory_classifier import Item, MemoryClassifier, evaluate, summarize  # noqa: E402
from inferx.npu_classifier import NpuEmbedder, move_to_npu  # noqa: E402

HEADS = ROOT / "models/assistant_classifier/heads.npz"


def timeline(sample: dict) -> list[dict]:
    return [{"id": m.id, "speaker": m.speaker, "text": m.text, "message": m,
             **{key: sample["labels"][m.id][key] for key in ("store", "kind", "replaces", "sensitive")}}
            for _, _, group in sessions(sample) for m in group]


def lockstep(laptop: MemoryClassifier, phone, rows: list[dict]) -> tuple[Counter, list[str]]:
    """Both controllers decide on the same active list, which follows the laptop's decisions."""
    counts, differences, active = Counter(), [], []
    for row in rows:
        a, vector = laptop.decide(row["speaker"], row["text"], active)
        b, _ = phone.decide(row["speaker"], row["text"], active)
        counts["messages"] += 1
        same = (a.store, a.kind, a.sensitive, a.replaces) == (b.store, b.kind, b.sensitive, b.replaces)
        counts["same"] += same
        for field in ("store", "kind", "sensitive", "replaces"):
            counts[f"{field}_differs"] += getattr(a, field) != getattr(b, field)
        counts["max_store_score_gap"] = max(counts["max_store_score_gap"], abs(a.store_score - b.store_score))
        if not same:
            differences.append(f"{row['text'][:70]!r}: laptop {(a.store, a.kind, a.sensitive, a.replaces)} "
                               f"(store {a.store_score:.3f}), phone {(b.store, b.kind, b.sensitive, b.replaces)} "
                               f"(store {b.store_score:.3f})")
        if a.store:
            if a.replaces:
                active = [item for item in active if item.id != a.replaces]
            active.append(Item(row["id"], row["speaker"], row["text"], vector, a.kind))
    return counts, differences


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sets", default="val,val_long")
    parser.add_argument("--users", type=int, default=2, help="users per set")
    args = parser.parse_args()
    laptop = move_to_npu(MemoryClassifier.load(HEADS, NpuEmbedder(ROOT / "models/minilm_fp32")),
                         ROOT / "models/assistant_classifier/npu")
    phone = load_phone_controller(HEADS, ROOT / "models/minilm_fp32/tokenizer.json")
    report = {}
    for name in args.sets.split(","):
        samples = json.loads((ROOT / f"data/assistant/{name}.json").read_text(encoding="utf-8"))[: args.users]
        total, scored = Counter(), {"laptop": Counter(), "phone": Counter()}
        examples = []
        for user, sample in enumerate(samples):
            rows = timeline(sample)
            start = time.perf_counter()
            phone.prefetch([row["message"] for row in rows])
            counts, differences = lockstep(laptop, phone, rows)
            total.update({k: v for k, v in counts.items() if k != "max_store_score_gap"})
            total["max_store_score_gap"] = max(total["max_store_score_gap"], counts["max_store_score_gap"])
            examples += differences
            scored["laptop"].update(evaluate(laptop, rows)["counts"])
            scored["phone"].update(evaluate(phone, rows)["counts"])
            print(f"{name} user {user}: {counts['same']} of {counts['messages']} decisions the same "
                  f"({time.perf_counter() - start:.0f}s)", flush=True)
        report[name] = {"lockstep": dict(total), "differences": examples,
                        **{device: summarize(scored[device]) for device in scored}}
        print(json.dumps({k: v for k, v in report[name].items() if k != "differences"}, indent=1, default=float))
        print("\n".join(examples[:10]))
    calls = {"embedder": phone.embed.phone_calls,
             **{n: getattr(phone, n).phone_calls for n in ("store", "kind", "sensitive", "replaces")}}
    report["phone_calls"] = calls
    out = ROOT / "runs/galaxy_check.json"
    out.write_text(json.dumps(report, indent=1, default=float))
    print("phone calls:", calls, "\nsaved", out)


if __name__ == "__main__":
    main()
