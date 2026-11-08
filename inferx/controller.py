"""The memory controller: the only component that writes facts or core memory.

Version 1 uses rules. worth_remembering() and core_facts() stand in for the small
NPU classifiers in the proposal until labeled training data exists.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from .interface import Message

GREETING = re.compile(
    r"^\s*(hi|hey|hello|thanks|thank you|bye|goodbye|see you|take care|good to see|"
    r"no problem|sure|ok|okay|wow|cool|nice|awesome|great|congrats)\b[^.?!]{0,40}[.!?]*\s*$",
    re.I,
)
SIGNAL = re.compile(
    r"\b(i|i'm|i've|i'll|my|we|our|me)\b|\b(yesterday|today|tomorrow|last|next|ago)\b|\d",
    re.I,
)
CORE_PATTERNS = [
    ("name", re.compile(r"\bmy name is ([A-Z][\w'-]+)")),
    ("allergy", re.compile(r"\bi(?:'m| am) allergic to ([^.,!?]{2,40})", re.I)),
    ("home", re.compile(r"\bi live in ([A-Z][^.,!?]{1,40})")),
    ("job", re.compile(r"\bi work as (?:an? )?([^.,!?]{2,40})", re.I)),
]
# Months count too, or facts dated 16 March and 16 June would look like duplicates.
NUMBER = re.compile(
    r"\d+|\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b",
    re.I,
)
PRONOUN = re.compile(r"\b(i|i'm|i've|i'll|me|my|mine|we|our|us|you|your)\b", re.I)
# Lasting states can change, so a newer one may replace an older one. Everything else is
# treated as an event, and a later similar event does not make an earlier one untrue.
LASTING = re.compile(
    r"\b(lives? in|works? (?:as|at|for)|is (?:an?|the) \w+|is allergic to|is married to|"
    r"owns?|favou?rite|has (?:an?|\d+|\w+ (?:cats?|dogs?|kids?|children|siblings?))\b)",
    re.I,
)


class Decision(Enum):
    ADD = "add"
    UPDATE = "update"
    IGNORE = "ignore"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class Thresholds:
    duplicate: float = 0.95
    related: float = 0.85


def worth_remembering(message: Message) -> bool:
    text = message.text.strip()
    if len(text.split()) < 4 or GREETING.match(text):
        return False
    return bool(SIGNAL.search(text))


def core_facts(message: Message) -> list[tuple[str, str]]:
    """Return (key, fact) pairs for stable facts stated in a message."""
    facts = []
    for key, pattern in CORE_PATTERNS:
        match = pattern.search(message.text)
        if match:
            facts.append((f"{message.speaker}.{key}", f"{message.speaker} {key}: {match.group(1).strip()}"))
    return facts


def valid_fact(fact: str, speaker: str) -> bool:
    """Reject rewrites that are not standalone. The raw message stays searchable either way."""
    if PRONOUN.search(fact) or fact.startswith((f"{speaker}:", "Speaker")):
        return False
    # Phone sources are named like "Maya's calendar", and a fact about them starts with "Maya".
    subject = speaker.split("'s ")[0]
    return subject.lower() in fact.lower()


def may_replace(fact: str, stored: str) -> bool:
    """Only a lasting state may replace another lasting state."""
    return bool(LASTING.search(fact) and LASTING.search(stored))


def decide(fact: str, nearest_text: str | None, nearest_score: float | None, t: Thresholds = Thresholds()) -> Decision:
    """Compare a new fact with its most similar stored fact."""
    if nearest_text is None or nearest_score is None or nearest_score < t.related:
        return Decision.ADD
    if nearest_score >= t.duplicate:
        new = {word.lower() for word in NUMBER.findall(fact)}
        old = {word.lower() for word in NUMBER.findall(nearest_text)}
        if not (new and old and new != old):
            return Decision.IGNORE
        return Decision.UPDATE if may_replace(fact, nearest_text) else Decision.ADD
    return Decision.AMBIGUOUS
