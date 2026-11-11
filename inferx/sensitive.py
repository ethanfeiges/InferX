"""Rules that spot secrets by their format, as a backstop to the trained 'sensitive' head.

The head only ever saw one-time codes. These rules cover formats that look the same for every user, so
they need no training data: card numbers (checked with the Luhn sum), IBANs (checked with mod 97),
passwords and PINs given in a message, one-time codes, US social security numbers, UK National
Insurance numbers, and API keys. Health details are left out on purpose: an allergy is a fact the
assistant should remember.
"""
from __future__ import annotations

import re

# "Area code" and "dress code" are not secrets.
NOT_SECRET = r"(?<!area )(?<!dress )(?<!zip )(?<!post )(?<!country )(?<!bar )(?<!colour )(?<!color )"
SECRET_WORD = (rf"(?:{NOT_SECRET}code|otp|passcode|pin|one[- ]time|verification|authentication|authori[sz]ation|"
               r"confirmation code|2fa|mfa)")
ACTION = r"(?:verify|confirm|authori[sz]e|approve|sign(?:ing)? in|log(?:ging)? in|enter|use)"
# 3 to 8 digits that are not part of a date, time, word or longer number ("12/05", "10:30", "1920s").
DIGITS = r"(?<![\d/.:-])\d{3,8}(?![\d/:-]|\.\d|[A-Za-z])"
CODE = re.compile(
    # "Security code: 118557", "PIN for this login: 8105", "authorisation number is 660418"
    rf"\b{SECRET_WORD}\b(?:\s+[\w-]+){{0,3}}?\s*(?::|=|\bis\b)\s*{DIGITS}"
    # "334442 is your authentication number", "G-208841 is your account confirmation code"
    rf"|(?:\b[A-Z]{{1,2}}-\d{{3,8}}\b|{DIGITS})[^.\d]{{0,25}}\b(?:is|as) your\b[^.\d]{{0,30}}\b{SECRET_WORD}"
    rf"|\b{ACTION}\b[^.\d]{{0,40}}\b(?:with|using|code)\b\s*:?\s*{DIGITS}"  # "confirm your payment with 7674"
    rf"|\b(?:use|enter)\s+{DIGITS}\s+to\b",  # "enter 804 to finish signing in"
    re.I)
PASSWORD = re.compile(
    r"\b(?:password|passwd|pwd|passphrase|pin(?: number| code)?|wifi key|wi-fi key|cvv|cvc)\b"
    r"[^.\n]{0,30}?(?:\s(?:is|was|=|-)|[:=])\s*\S+", re.I)
SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
NI_NUMBER = re.compile(r"\b[A-CEGHJ-PR-TW-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b")
API_KEY = re.compile(r"\b(?:sk|pk|rk)[-_](?:live|test|proj)?[-_]?[A-Za-z0-9]{16,}\b|\bAKIA[0-9A-Z]{16}\b"
                     r"|\bgh[pousr]_[A-Za-z0-9]{30,}\b|\bxox[abpr]-[A-Za-z0-9-]{10,}\b")
CARD = re.compile(r"\b\d(?:[ -]?\d){12,18}\b")
IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]){11,30}\b")


def luhn(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2:
            value = value * 2 - 9 if value > 4 else value * 2
        total += value
    return total % 10 == 0


def iban_valid(text: str) -> bool:
    compact = text.replace(" ", "")
    rearranged = compact[4:] + compact[:4]
    number = "".join(str(int(char, 36)) for char in rearranged)
    return 15 <= len(compact) <= 34 and int(number) % 97 == 1


def secret_kind(text: str) -> str | None:
    """Name the kind of secret a message contains, or None."""
    for match in CARD.finditer(text):
        digits = re.sub(r"\D", "", match.group())
        if 13 <= len(digits) <= 19 and luhn(digits):
            return "card number"
    if any(iban_valid(match.group()) for match in IBAN.finditer(text)):
        return "bank account"
    for kind, pattern in (("password or PIN", PASSWORD), ("one-time code", CODE), ("social security number", SSN),
                          ("national insurance number", NI_NUMBER), ("API key", API_KEY)):
        if pattern.search(text):
            return kind
    return None
