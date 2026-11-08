"""Compare systems across runs on the simulated assistant data.

Pass runs oldest first. A later run replaces an earlier one for the same system and user.

Usage: python scripts/compare_assistant_runs.py runs/<a> runs/<b> ...
"""
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inferx.assistant_score import correct, out_of_date  # noqa: E402

SYSTEMS = ["plain_retrieval", "decoupled", "trained"]


def interval(differences):
    rng = random.Random(0)
    means = sorted(statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(10000))
    return round(statistics.mean(differences), 3), round(means[250], 3), round(means[9750], 3)


by_user = defaultdict(dict)
idle_by_user = defaultdict(dict)
for run in map(Path, sys.argv[1:]):
    summary = json.loads((run / "summary.json").read_text())
    user = json.loads((run / "config.json").read_text())["conversation"]
    for system in SYSTEMS:
        path = run / f"{system}.jsonl"
        if path.exists() and system in summary:
            lines = path.read_text(encoding="utf-8").splitlines()
            by_user[system][user] = [json.loads(line) for line in lines]
            idle_by_user[system][user] = summary[system].get("idle", {})

rows = {}
for system in SYSTEMS:
    rows[system] = [row for user in sorted(by_user[system]) for row in by_user[system][user]]
    for row in rows[system]:
        row["correct"] = float(not row["error"] and correct(row, row["text"]))
        row["stale"] = out_of_date(row, row["text"])

for system in SYSTEMS:
    if not rows[system]:
        continue
    group_rows = rows[system]
    print(f"\n{system}: {len(group_rows)} questions, {sum(bool(r['error']) for r in group_rows)} errors, "
          f"correct {statistics.mean(r['correct'] for r in group_rows):.3f}, F1 {statistics.mean(r['score'] or 0 for r in group_rows):.3f}, "
          f"evidence recall {statistics.mean(r['evidence_recall'] for r in group_rows if r['evidence_recall'] is not None):.3f}")
    groups = defaultdict(list)
    for row in group_rows:
        groups[row["qtype"]].append(row)
    for name, group in sorted(groups.items()):
        recalls = [r["evidence_recall"] for r in group if r["evidence_recall"] is not None]
        recall = f"{statistics.mean(recalls):.3f}" if recalls else "n/a"
        print(f"  {name}: n={len(group)} correct {statistics.mean(r['correct'] for r in group):.3f}, evidence recall {recall}")
    changed = [row for row in group_rows if row["stale_answer"]]
    print(f"  about something that changed: {len(changed)}, correct {sum(r['correct'] for r in changed):.0f}, "
          f"out-of-date replies {sum(r['stale'] for r in changed)}")
    totals = defaultdict(float)
    for stats in idle_by_user[system].values():
        for key, value in stats.items():
            totals[key] += value
    print("  idle:", {key: round(value, 1) for key, value in totals.items()})

for a, b in (("trained", "plain_retrieval"), ("trained", "decoupled"), ("decoupled", "plain_retrieval")):
    if rows[a] and rows[b]:
        assert all(x["question"] == y["question"] for x, y in zip(rows[a], rows[b]))
        print(f"\n{a} minus {b}: correct", interval([x["correct"] - y["correct"] for x, y in zip(rows[a], rows[b])]))
        changed = [(x, y) for x, y in zip(rows[a], rows[b]) if x["stale_answer"]]
        print("  about something that changed:", interval([x["correct"] - y["correct"] for x, y in changed]))
