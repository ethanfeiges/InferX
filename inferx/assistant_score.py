"""Fair scoring for the simulated assistant, where messages show dates in many formats.

LoCoMo keeps its official F1 scoring. Here a reply is correct when it contains the answer, so
"12/4/2025" matches "12 April 2025" and "$7 per month" matches "$7".
"""
from __future__ import annotations

import re

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november",
          "december"]
_MONTH = "|".join(MONTHS) + "|" + "|".join(month[:3] for month in MONTHS)
GOLD_DATE = re.compile(rf"^(\d{{1,2}}) ({_MONTH}) \d{{4}}$")
SLASH = re.compile(r"\b(\d{1,2})/(\d{1,2})(?:/\d{2,4})?\b")
DAY_MONTH = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?(?: of)? ({_MONTH})\b")
MONTH_DAY = re.compile(rf"\b({_MONTH}) (\d{{1,2}})(?:st|nd|rd|th)?\b")
DIGITS = re.compile(r"^[\d\s]+$")


def _dates(text: str) -> set[tuple[int, str]]:
    found = {(int(day), month[:3]) for day, month in DAY_MONTH.findall(text)}
    found |= {(int(day), month[:3]) for month, day in MONTH_DAY.findall(text)}
    found |= {(int(day), MONTHS[int(month) - 1][:3]) for day, month in SLASH.findall(text) if 1 <= int(month) <= 12}
    return found


def contains(reply: str, value: str) -> bool:
    """Whether the reply states this value, in any common format."""
    reply, value = reply.lower(), value.lower().strip()
    date = GOLD_DATE.match(value)
    if date:
        return (int(date[1]), date[2][:3]) in _dates(reply)
    if DIGITS.match(value):
        return value.replace(" ", "") in re.sub(r"[\s-]", "", reply)
    pattern = r"(?<![\w$])" + re.escape(value) + r"(?!\w)"
    if re.search(pattern, reply):
        return True
    # Word forms of a status count, such as "refund issued" for "refunded".
    stem = value[:6] if len(value) > 6 and " " not in value else None
    return bool(stem and re.search(rf"\b{re.escape(stem)}", reply))


def correct(qa: dict, reply: str) -> bool:
    if qa["category"] == 5:
        lowered = reply.lower()
        return "not mentioned" in lowered or "no information available" in lowered
    return contains(reply, str(qa["answer"]))


def out_of_date(qa: dict, reply: str) -> bool:
    """The reply gives the old value and not the current one."""
    stale = qa.get("stale_answer")
    return bool(stale) and contains(reply, stale) and not contains(reply, str(qa["answer"]))
