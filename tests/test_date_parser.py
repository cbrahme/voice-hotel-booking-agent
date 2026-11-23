from datetime import date

import pytest

from backend.app.utils.date_parser import parse_iso, speak_date, validate_check_in, validate_range
from tests.conftest import TODAY, day


def test_tc03_valid_range_has_no_errors():
    assert validate_range(day(1), day(3), TODAY) == []
    assert validate_range(TODAY, day(30), TODAY) == []  # same-day check-in, 30 nights is the limit


@pytest.mark.parametrize(
    "ci,co,expected",
    [
        (-1, 2, "has already passed"),
        (5, 5, "Check-out has to be after check-in"),
        (5, 3, "Check-out has to be after check-in"),
        (1, 32, "up to 30 nights"),
        (366, 368, "up to a year ahead"),
    ],
)
def test_tc04_invalid_ranges(ci, co, expected):
    errors = validate_range(day(ci), day(co), TODAY)
    assert len(errors) == 1 and expected in errors[0]


def test_past_check_in_names_the_date():
    assert validate_check_in(day(-1), TODAY) == ["Thursday, November 19th has already passed."]


@pytest.mark.parametrize(
    "d,expected",
    [
        (date(2026, 11, 21), "Saturday, November 21st"),
        (date(2026, 11, 22), "Sunday, November 22nd"),
        (date(2026, 11, 23), "Monday, November 23rd"),
        (date(2026, 11, 11), "Wednesday, November 11th"),
        (date(2026, 12, 12), "Saturday, December 12th"),
        (date(2026, 12, 31), "Thursday, December 31st"),
    ],
)
def test_speak_date(d, expected):
    assert speak_date(d, TODAY) == expected


def test_speak_date_adds_year_when_not_this_year():
    assert speak_date(date(2027, 1, 2), TODAY) == "Saturday, January 2nd, 2027"


@pytest.mark.parametrize("value,expected", [("2026-11-27", date(2026, 11, 27)), (" 2026-11-27 ", date(2026, 11, 27)),
                                            ("next friday", None), ("", None), (None, None), ("2026-02-30", None)])
def test_parse_iso(value, expected):
    assert parse_iso(value) == expected
