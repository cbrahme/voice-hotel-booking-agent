"""Offline checks of the prompt sent to the LLM."""

from datetime import date

from backend.app.agent.extractor import ExtractionContext, build_messages, calendar_lines


def test_calendar_lists_weekdays_from_today():
    lines = calendar_lines(date(2026, 11, 23)).splitlines()
    assert len(lines) == 21
    assert lines[0] == "Mon 2026-11-23 (today)"
    assert lines[4] == "Fri 2026-11-27"
    assert lines[-1] == "Sun 2026-12-13"


def test_prompt_contains_context():
    ctx = ExtractionContext(
        today=date(2026, 11, 23),
        step="room",
        known={"check_in_date": "2026-11-27", "num_guests": 2, "room_type": None},
        last_reply='Say "Deluxe" or "Suite"',
    )
    system, human = build_messages("the deluxe", ctx)
    assert human.content == "the deluxe"
    text = system.content
    assert "Today is 2026-11-23 (Monday)." in text
    assert "Fri 2026-11-27" in text
    assert "Current step: room" in text
    assert '{"check_in_date": "2026-11-27", "num_guests": 2}' in text  # unknown values left out
    assert "Say 'Deluxe' or 'Suite'" in text  # quotes can't break the prompt
