"""Date validation and speech-friendly formatting.

The LLM turns phrases like "next Friday for 3 nights" into ISO dates; this module checks
the result. ``today`` is always passed in, so tests never depend on the real clock.
"""

from __future__ import annotations

from datetime import date

MAX_NIGHTS = 30
MAX_DAYS_AHEAD = 365


def parse_iso(value: str | None) -> date | None:
    """Parse ``YYYY-MM-DD``; returns None for empty or malformed input."""
    if not value:
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return None


def validate_check_in(check_in: date, today: date) -> list[str]:
    if check_in < today:
        return [f"{speak_date(check_in, today)} has already passed."]
    if (check_in - today).days > MAX_DAYS_AHEAD:
        return ["We can only take bookings up to a year ahead."]
    return []


def validate_range(check_in: date, check_out: date, today: date) -> list[str]:
    """User-facing problems with a stay; an empty list means the dates are fine."""
    errors = validate_check_in(check_in, today)
    nights = (check_out - check_in).days
    if nights <= 0:
        errors.append("Check-out has to be after check-in.")
    elif nights > MAX_NIGHTS:
        errors.append(f"We can only take bookings of up to {MAX_NIGHTS} nights.")
    return errors


def _ordinal(n: int) -> str:
    suffix = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def speak_date(d: date, today: date | None = None) -> str:
    """'Friday, November 27th' (year added when it isn't the current one)."""
    text = f"{d:%A}, {d:%B} {_ordinal(d.day)}"
    if today is not None and d.year != today.year:
        text += f", {d.year}"
    return text
