"""Phone number and guest name clean-up."""

from __future__ import annotations

import re

MIN_PHONE_DIGITS = 10
MAX_PHONE_DIGITS = 15
MAX_NAME_LENGTH = 100


def normalize_phone(raw: str | None) -> str | None:
    """Keep the digits (and a leading '+'). Returns None unless there are 10–15 digits."""
    if not raw:
        return None
    raw = raw.strip()
    digits = re.sub(r"\D", "", raw)
    if not MIN_PHONE_DIGITS <= len(digits) <= MAX_PHONE_DIGITS:
        return None
    return ("+" if raw.startswith("+") else "") + digits


def speak_phone(phone: str) -> str:
    """Digit by digit, in groups, so text-to-speech reads it clearly:
    '5551234567' -> '5 5 5, 1 2 3, 4 5 6 7'."""
    plus = phone.startswith("+")
    digits = phone.lstrip("+")
    if len(digits) == 10:
        groups = [digits[:3], digits[3:6], digits[6:]]
    else:
        groups = [digits[i : i + 3] for i in range(0, len(digits), 3)]
    spoken = ", ".join(" ".join(g) for g in groups)
    return f"plus {spoken}" if plus else spoken


def clean_name(raw: str | None) -> str | None:
    """Collapse whitespace and fix all-lower/all-upper casing. None if it isn't a usable name."""
    if not raw:
        return None
    name = " ".join(raw.split())
    if not any(c.isalpha() for c in name) or len(name) > MAX_NAME_LENGTH:
        return None
    if name.islower() or name.isupper():
        name = name.title()
    return name
