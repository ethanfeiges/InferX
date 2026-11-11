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


MONTHS = {name: number for number, name in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_MONTH_WORD = r"(?P<{}>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?"
_ORDINAL = r"(?:st|nd|rd|th)?"
SOURCE_DATE = re.compile(
    r"\b(?P<d1>\d{1,2})" + _ORDINAL + r"(?:\s+of)?\s+" + _MONTH_WORD.format("m1") + r"\b"
    r"|\b" + _MONTH_WORD.format("m2") + r"\s+(?P<d2>\d{1,2})" + _ORDINAL + r"\b"
    r"|\b(?P<d3>\d{1,2})/(?P<m3>\d{1,2})(?:/\d{2,4})?\b",  # numeric dates are day first, as in the data
    re.I,
)


def stated_days(text: str) -> set[tuple[int, int]]:
    """Every (day, month) a text states, in any common form: '21 April', 'Apr 21st', '21/4/2025'."""
    days = set()
    for match in SOURCE_DATE.finditer(text):
        if match["d3"]:
            day, month = int(match["d3"]), int(match["m3"])
        else:
            day = int(match["d1"] or match["d2"])
            month = MONTHS[(match["m1"] or match["m2"]).lower()[:3]]
        if 1 <= day <= 31 and 1 <= month <= 12:
            days.add((day, month))
    return days


def remove_unsupported_dates(fact: str, source: str) -> tuple[str, int]:
    """Drop full dates from a fact unless the (annotated) source states the same day and month.

    The source may write the date in any common form ('25/5/2025', 'May 25', '25 May'), so dates are
    compared as (day, month), not as text. Only real month names are checked; anything else is kept.
    """
    stated = stated_days(source)

    def supported(match: re.Match) -> bool:
        month = (match["month"] or match["month2"]).lower()[:3]
        if month not in MONTHS:
            return True
        return (int(match["day"] or match["day2"]), MONTHS[month]) in stated

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
