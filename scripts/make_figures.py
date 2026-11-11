"""Draw the README figures from the validation results.

The numbers come from scripts/compare_assistant_runs.py on the runs named beside each figure.
Trained and rules were run before the date-rule fix (commit f8d6568); reruns are pending.

Usage: python scripts/make_figures.py   (writes docs/figures/*.png)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "figures"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#b4b2a9"  # reference palette slots 1-2, neutral
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": "#d6d4cc", "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def bars(ax, labels, values, highlight, unit="", total=None, title=""):
    """Horizontal bars, one highlighted, value at the end of each bar."""
    y = range(len(labels))[::-1]
    colors = [BLUE if label == highlight else GRAY for label in labels]
    ax.barh(list(y), values, color=colors, height=0.6)
    top = total or max(values)
    for yi, value in zip(y, values):
        ax.text(value + top * 0.015, yi, f"{value:g}{unit}", va="center", color=INK, fontsize=9)
    ax.set_yticks(list(y), labels)
    ax.set_xlim(0, top * 1.15)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#ebe9e2", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8)


def long_history():
    # Runs 20261108-145846, 161136, 174426: 3 users, 112 days, 248 questions, cloud rule on.
    systems = ["Trained", "Plain retrieval", "Local memory", "Rules", "Full context"]
    correct = [75.8, 69.4, 69.4, 68.5, 8.9]
    changed_ok = [56, 43, 43, 44, 7]
    stale = [4, 21, 21, 23, 1]
    cloud = [45, 87, 248, 115, 142]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2), gridspec_kw={"wspace": 0.12})
    bars(axes[0], systems, correct, "Trained", "%", 100, "Answers correct (%)")
    bars(axes[1], systems, cloud, "Trained", "", 248, "Questions sent to the cloud (of 248)")
    axes[1].set_yticklabels([])
    fig.suptitle("Long-history validation users (3 users, 112 days each, 248 questions)",
                 x=0.01, y=1.06, ha="left", fontsize=12, color=INK)
    fig.savefig(OUT / "long_history.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return changed_ok


def changed_facts():
    # Same runs: questions about something that changed (83).
    systems = ["Trained", "Rules", "Plain retrieval", "Full context"]
    ok, stale = [56, 44, 43, 7], [4, 23, 21, 1]
    y = range(len(systems))[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    ax.barh([v + 0.17 for v in y], ok, height=0.32, color=BLUE, label="Current value (correct)")
    ax.barh([v - 0.17 for v in y], stale, height=0.32, color=ORANGE, label="Old value (wrong)")
    for yi, a, b in zip(y, ok, stale):
        ax.text(a + 1, yi + 0.17, str(a), va="center", fontsize=9, color=INK)
        ax.text(b + 1, yi - 0.17, str(b), va="center", fontsize=9, color=INK)
    ax.set_yticks(list(y), systems)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 83)
    ax.grid(axis="x", color="#ebe9e2", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.set_title("83 questions about something that changed (long users)", loc="left", fontsize=10.5, color=INK)
    fig.savefig(OUT / "changed_facts.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def mem0():
    # Short validation users 0 and 1 (111 questions): the users mem0 cloud has finished.
    systems = ["Trained", "Plain retrieval", "mem0 cloud", "mem0 2025 on phi"]
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    bars(ax, systems, [84.7, 78.4, 74.8, 31.5], "Trained", "%", 100,
         "Answers correct (%), short users 0 and 1")
    fig.savefig(OUT / "mem0.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fact_audit():
    # All 288 facts phi wrote for the trained system on the 3 long users, checked by hand.
    kinds = ["Correct", "Date wrongly removed by the date rule (bug, fixed)", "Two people merged",
             "Meaning reversed or garbled", "Wrong date or time", "Wrong item treated as an update"]
    counts = [254, 17, 6, 5, 3, 3]
    fig, ax = plt.subplots(figsize=(6.5, 2.9))
    bars(ax, kinds, counts, "Correct", "", 288, "What phi's 288 rewritten facts look like (long users)")
    fig.savefig(OUT / "fact_audit.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    long_history()
    changed_facts()
    mem0()
    fact_audit()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
