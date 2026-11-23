"""TC-16: the real OpenAI extractor on a few realistic utterances.

Excluded by default; run with `pytest -m live` (uses a small number of tokens).
"""

import pytest

from backend.app.agent.extractor import ExtractionContext, OpenAIExtractor
from backend.app.config import get_settings
from tests.conftest import TODAY, day

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def llm():
    settings = get_settings()
    if not settings.openai_api_key:
        pytest.skip("OPENAI_API_KEY not set")
    return OpenAIExtractor(settings.openai_model, settings.openai_api_key)


def ctx(step="dates", last_reply="Hi! I can book a room for you. When would you like to stay?", **known):
    return ExtractionContext(today=TODAY, step=step, known=known, last_reply=last_reply)


def test_relative_dates_and_guests(llm):
    ext = llm.extract("I need a room next Friday for three nights for me and my wife", ctx())
    assert ext.check_in == day(7).isoformat()
    assert ext.nights == 3 or ext.check_out == day(10).isoformat()
    assert ext.num_guests == 2
    assert ext.intent == "provide"


def test_spoken_phone_and_name(llm):
    ext = llm.extract(
        "it's Ada Lovelace, my number is five five five, one two three, four five six seven",
        ctx(step="contact", last_reply="May I have your full name and phone number?"),
    )
    assert ext.guest_name == "Ada Lovelace"
    assert "".join(c for c in (ext.phone_number or "") if c.isdigit()) == "5551234567"


def test_yes_at_confirmation(llm):
    ext = llm.extract("yes please go ahead", ctx(step="awaiting_confirmation", last_reply="... Shall I confirm?"))
    assert ext.intent == "confirm"
    assert ext.check_in is None and ext.num_guests is None


def test_does_not_repeat_known_values(llm):
    ext = llm.extract(
        "the deluxe one",
        ctx(
            step="room",
            last_reply="For 2 guests on those dates, we have a Standard at $120 a night, a Deluxe at $180 a night, "
            "and a Suite at $280 a night. Which would you like?",
            check_in_date=day(7).isoformat(),
            num_guests=2,
        ),
    )
    assert ext.room_type == "Deluxe"
    assert ext.check_in is None and ext.num_guests is None


def test_garbled_speech_is_unclear(llm):
    ext = llm.extract("uh the the mmm blue", ctx(step="guests", last_reply="How many guests will be staying?"))
    assert ext.intent in ("unclear", "off_topic")
    assert ext.num_guests is None
