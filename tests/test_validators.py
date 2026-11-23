import pytest

from backend.app.utils.validators import clean_name, normalize_phone, speak_phone


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("555-123-4567", "5551234567"),
        ("(555) 123 4567", "5551234567"),
        ("+1 555 123 4567", "+15551234567"),
        ("+44 20 7946 0958", "+442079460958"),
        ("12345", None),  # too short
        ("1234567890123456", None),  # too long
        ("", None),
        (None, None),
    ],
)
def test_normalize_phone(raw, expected):
    assert normalize_phone(raw) == expected


def test_speak_phone_groups_digits():
    assert speak_phone("5551234567") == "5 5 5, 1 2 3, 4 5 6 7"
    assert speak_phone("+15551234567") == "plus 1 5 5, 5 1 2, 3 4 5, 6 7"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("  ada   lovelace ", "Ada Lovelace"),
        ("GRACE HOPPER", "Grace Hopper"),
        ("Mary-Jane McDonald", "Mary-Jane McDonald"),  # mixed case is left alone
        ("123", None),
        ("", None),
        ("x" * 101, None),
    ],
)
def test_clean_name(raw, expected):
    assert clean_name(raw) == expected
