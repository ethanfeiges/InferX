"""Date handling for fact writing, done by rules so the LLM never does date arithmetic."""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
COUNTS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
_COUNT = r"\d+|an?|one|two|three|four|five|six"
RELATIVE = re.compile(
    r"\b(?:(?P<simple>yesterday|today|tonight|tomorrow|last night)"
    r"|(?P<count>" + _COUNT + r") (?P<unit>day|week)s? ago"
    r"|(?P<side>last|next) (?P<span>week|weekend|month|year|" + "|".join(WEEKDAYS) + r")"
    r"|this (?P<this>month|year))\b",
    re.I,
)
_MONTH = r"[A-Z][a-z]+"
FULL_DATE = re.compile(
    r"\b(?:(?P<day>\d{1,2})(?:st|nd|rd|th)? (?P<month>" + _MONTH + r"),? \d{4}"
    r"|(?P<month2>" + _MONTH + r") (?P<day2>\d{1,2})(?:st|nd|rd|th)?,? \d{4})\b"
)
_LEAD_IN = r"(?:,?\s*(?:specifically\s+)?(?:on|around|about|since|as of|by|at|in|approximately)\s+(?:approximately\s+|around\s+)?)?"


def parse_date(text: str) -> date | None:
    """Read a date such as '7:18 pm on 27 May, 2023' or '8 May 2023'."""
    cleaned = re.sub(r"^.*?\bon\s+", "", text).replace(",", "").strip()
    try:
        return datetime.strptime(cleaned, "%d %B %Y").date()
    except ValueError:
        return None


def _day(when: date) -> str:
    return f"{when.day} {when:%B %Y}"


def _month(when: date, offset: int) -> str:
    index = when.year * 12 + when.month - 1 + offset
    return f"{date(index // 12, index % 12 + 1, 1):%B %Y}"


def _resolve(match: re.Match, when: date) -> str:
    if match["simple"]:
        days = {"yesterday": -1, "last night": -1, "today": 0, "tonight": 0, "tomorrow": 1}
        return _day(when + timedelta(days=days[match["simple"].lower()]))
    if match["count"]:
        word = match["count"].lower()
        count = int(word) if word.isdigit() else COUNTS[word]
        if match["unit"].lower() == "day":
            return _day(when - timedelta(days=count))
        return f"{count} week{'s' if count > 1 else ''} before {_day(when)}"
    if match["this"]:
        return _month(when, 0) if match["this"].lower() == "month" else str(when.year)
    last = match["side"].lower() == "last"
    span = match["span"].lower()
    if span in ("week", "weekend"):
        return f"the {span} {'before' if last else 'after'} {_day(when)}"
    if span == "month":
        return _month(when, -1 if last else 1)
    if span == "year":
        return str(when.year + (-1 if last else 1))
    target = WEEKDAYS.index(span)
    if last:
        return _day(when - timedelta(days=(when.weekday() - target - 1) % 7 + 1))
    return _day(when + timedelta(days=(target - when.weekday() - 1) % 7 + 1))


def annotate(text: str, when: date | None) -> str:
    """Add the resolved date after each exact relative time, leaving vague ones as written."""
    if when is None:
        return text
    return RELATIVE.sub(lambda match: f"{match[0]} ({_resolve(match, when)})", text)


def remove_unsupported_dates(fact: str, source: str) -> tuple[str, int]:
    """Drop full dates from a fact unless the (annotated) source states them."""
    lowered = source.lower()

    def supported(match: re.Match) -> bool:
        day, month = match["day"] or match["day2"], (match["month"] or match["month2"]).lower()
        return bool(re.search(rf"\b{day} {month}\b|\b{month} {day}(?:st|nd|rd|th)?\b", lowered))

    pattern = re.compile(_LEAD_IN + FULL_DATE.pattern)
    removed = 0

    def replace(match: re.Match) -> str:
        nonlocal removed
        if supported(match):
            return match[0]
        removed += 1
        return ""

    cleaned = pattern.sub(replace, fact)
    cleaned = re.sub(r"\s+([.,;])", r"\1", cleaned)
    cleaned = re.sub(r",\s*\.", ".", cleaned)
    return re.sub(r"\s{2,}", " ", cleaned).strip(), removed
