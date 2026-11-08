"""Context budget and prompt construction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

ANSWER_SYSTEM = (
    "Answer the question using only the memories. Reply with a short phrase. "
    "If no memory mentions it, reply: Not mentioned"
)


@dataclass(frozen=True)
class Budget:
    """Token budget. Defaults follow the proposal's split of a 4,096-token limit."""

    system: int = 200
    core: int = 300
    retrieved: int = 1200
    recent: int = 1400
    reply: int = 800


def fit(items: list[str], limit: int, count: Callable[[str], int]) -> list[str]:
    """Keep items in order until the token limit is reached."""
    kept, used = [], 0
    for item in items:
        cost = count(item) + 1
        if used + cost > limit:
            break
        kept.append(item)
        used += cost
    return kept


def build_prompt(question: str, core: list[str], retrieved: list[str], recent: list[str]) -> str:
    parts = []
    if core:
        parts.append("Known facts:\n" + "\n".join(core))
    if retrieved:
        parts.append("Memories:\n" + "\n".join(retrieved))
    if recent:
        parts.append("Recent messages:\n" + "\n".join(recent))
    parts.append(f"Question: {question} Answer:")
    return "\n\n".join(parts)
