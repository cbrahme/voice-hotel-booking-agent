"""Conversation tests with a scripted fake LLM (no network, fixed 'today')."""

import pytest

from backend.app import crud
from backend.app.agent import prompts as P
from backend.app.agent.state import TurnExtraction
from tests.conftest import TODAY, day

SID = "session-1"


@pytest.fixture
def say(agent, extractor):
    """say(intent=..., **fields) -> TurnResult: script what the LLM 'hears', then send a turn."""

    def _say(intent="provide", text="(speech)", session=SID, **fields):
        for key in ("check_in", "check_out"):
            if isinstance(fields.get(key), int):
                fields[key] = day(fields[key]).isoformat()
        extractor.push(TurnExtraction(intent=intent, **fields))
        return agent.chat(session, text)

    return _say


def fill_to_confirmation(say, room="Deluxe", guests=2):
    say(check_in=7, nights=2)
    say(num_guests=guests)
    say(room_type=room)
    return say(guest_name="Ada Lovelace", phone_number="555 123 4567")


def reservations(session_factory):
    with session_factory() as db:
        return crud.list_reservations(db)


# --- happy path & one-shot -----------------------------------------------------------


def test_tc05_happy_path(say, session_factory):
    steps = [
        say(check_in=7, nights=2).current_step,
        say(num_guests=2).current_step,
        say(room_type="Deluxe").current_step,
        say(guest_name="Ada Lovelace", phone_number="555 123 4567").current_step,
    ]
    assert steps == ["guests", "room", "contact", "awaiting_confirmation"]

    result = say(intent="confirm")
    assert result.current_step == "confirmed" and result.is_complete
    assert "Ada" in result.reply and str(result.reservation_id) in result.reply

    [r] = reservations(session_factory)
    assert (r.reservation_id, r.room_type, r.num_guests, r.total_price, r.status) == (
        result.reservation_id, "Deluxe", 2, 360.0, "CONFIRMED")
    assert (r.check_in_date, r.check_out_date, r.phone_number) == (day(7), day(9), "5551234567")


def test_tc15_everything_in_one_sentence(say):
    result = say(check_in=7, check_out=10, num_guests=2, room_type="Suite",
                 guest_name="Grace Hopper", phone_number="+1 555 987 6543")
    assert result.current_step == "awaiting_confirmation"
    assert result.reply.startswith("Let me confirm: a Suite for 2 guests")
    assert "3 nights, for a total of $840" in result.reply
    assert "plus 1 5 5, 5 9 8, 7 6 5, 4 3" in result.reply


def test_replies_read_back_dates_and_prices(say):
    r = say(check_in=7, nights=2)
    assert r.reply == "Got it: Friday, November 27th to Sunday, November 29th, 2 nights. How many guests will be staying?"
    r = say(num_guests=2)
    assert r.reply == ("For 2 guests on those dates, we have a Standard at $120 a night, a Deluxe at $180 a night, "
                       "and a Suite at $280 a night. Which would you like?")
    r = say(room_type="Standard")
    assert r.reply == "The Standard for 2 nights comes to $240. May I have your full name and phone number?"
    assert r.booking["total_price"] == 240.0 and r.booking["nights"] == 2


# --- dates -----------------------------------------------------------------------------


def test_check_in_then_nights_in_next_turn(say):
    r = say(check_in=7)
    assert r.current_step == "dates"
    assert r.reply == "Checking in Friday, November 27th. How many nights will you stay?"
    r = say(nights=3)
    assert r.current_step == "guests" and r.booking["check_out_date"] == day(10).isoformat()


def test_tc04_past_date_is_rejected(say):
    r = say(check_in=-1, nights=2)
    assert r.current_step == "dates" and r.booking["check_in_date"] is None
    assert r.reply.startswith("Thursday, November 19th has already passed.")


def test_unparseable_date_from_llm(say):
    r = say(check_in="next friday")
    assert r.current_step == "dates" and r.reply.startswith(P.BAD_DATE)


def test_moving_check_in_past_old_check_out_keeps_stay_length(say):
    say(check_in=7, nights=2)
    r = say(check_in=10)
    assert (r.booking["check_in_date"], r.booking["check_out_date"]) == (day(10).isoformat(), day(12).isoformat())


def test_moving_check_in_earlier_keeps_check_out(say):
    say(check_in=7, nights=2)
    r = say(check_in=6)
    assert (r.booking["check_in_date"], r.booking["check_out_date"]) == (day(6).isoformat(), day(9).isoformat())


# --- guests & capacity -----------------------------------------------------------------


def test_tc06_too_many_guests(say):
    say(check_in=7, nights=2)
    r = say(num_guests=5)
    assert r.current_step == "guests" and r.booking["num_guests"] is None
    assert r.reply == "Our largest room, the Suite, fits up to 4 guests. How many guests will be staying?"


def test_single_room_option_yes_picks_it(say):
    say(check_in=7, nights=2)
    r = say(num_guests=4)
    assert "we have a Suite at $280 a night. Would you like it?" in r.reply
    r = say(intent="confirm")
    assert r.current_step == "contact" and r.booking["room_type"] == "Suite"


# --- backtracking ----------------------------------------------------------------------


def test_tc07_guest_change_at_room_step_refilters_rooms(say, session_factory):
    say(check_in=7, nights=2)
    say(num_guests=2)
    r = say(num_guests=3)
    assert r.current_step == "room"
    assert "Standard" not in r.reply and "Deluxe" in r.reply and "Suite" in r.reply
    assert reservations(session_factory) == []


def test_guest_change_clears_room_that_no_longer_fits(say):
    say(check_in=7, nights=2)
    say(num_guests=2)
    say(room_type="Standard")
    r = say(num_guests=3)
    assert r.current_step == "room" and r.booking["room_type"] is None and r.booking["total_price"] is None
    assert r.reply.startswith("The Standard fits up to 2 guests. For 3 guests on those dates")


def test_date_change_keeps_room_and_reprices(say):
    fill_to_confirmation(say, room="Deluxe")
    r = say(check_in=7, nights=4)
    assert r.current_step == "awaiting_confirmation" and r.booking["room_type"] == "Deluxe"
    assert r.booking["total_price"] == 720.0 and "for a total of $720" in r.reply
    assert r.reply.startswith("Let me confirm")  # dates aren't acknowledged twice


def test_date_change_to_sold_out_dates_clears_room(say, session_factory):
    for _ in range(2):
        with session_factory() as db:
            crud.create_reservation(db, "Other Guest", "5550000000", "Suite", day(20), day(22), 2)
    say(check_in=7, nights=2)
    say(num_guests=2)
    say(room_type="Suite")
    r = say(check_in=20, nights=2)
    assert r.current_step == "room" and r.booking["room_type"] is None
    assert r.reply.startswith("Sorry, the Suite isn't available for those dates.")
    assert "Suite at" not in r.reply


def test_room_named_before_dates_is_checked_later(say):
    say(room_type="Standard")
    say(check_in=7, nights=2)
    r = say(num_guests=3)
    assert r.booking["room_type"] is None and r.reply.startswith("The Standard fits up to 2 guests.")


# --- confirmation ----------------------------------------------------------------------


def test_tc12_deny_then_fix_phone_then_confirm(say, session_factory):
    fill_to_confirmation(say)
    r = say(intent="deny")
    assert r.current_step == "awaiting_confirmation" and r.reply == P.ASK_WHAT_TO_CHANGE
    r = say(phone_number="555 987 6543")
    assert r.reply.startswith("Let me confirm") and "9 8 7, 6 5 4 3" in r.reply
    r = say(intent="confirm")
    assert r.is_complete
    [res] = reservations(session_factory)
    assert res.phone_number == "5559876543"


def test_unclear_answer_at_confirmation_asks_yes_or_no(say, session_factory):
    fill_to_confirmation(say)
    r = say(intent="unclear")
    assert r.reply == f"{P.DIDNT_CATCH} {P.ASK_YES_NO}" and r.current_step == "awaiting_confirmation"
    assert reservations(session_factory) == []


def test_invalid_phone_is_not_stored(say):
    say(check_in=7, nights=2)
    say(num_guests=2)
    say(room_type="Deluxe")
    r = say(guest_name="Ada Lovelace", phone_number="555 123")  # user paused mid-number
    assert r.current_step == "contact" and r.booking["phone_number"] is None
    assert r.reply == (
        "I only caught 6 digits: 5 5 5, 1 2 3. Please say the whole number, including the area code. "
        "Thanks, Ada. What's the best phone number to reach you?"
    )
    r = say(phone_number="5551234567")
    assert r.current_step == "awaiting_confirmation" and r.booking["phone_number"] == "5551234567"


@pytest.mark.parametrize(
    "raw,expected",
    [("1234567890123456", P.PHONE_TOO_LONG), ("call me", P.BAD_PHONE)],
)
def test_other_phone_problems(say, raw, expected):
    say(check_in=7, nights=2)
    say(num_guests=2)
    say(room_type="Deluxe")
    r = say(guest_name="Ada Lovelace", phone_number=raw)
    assert r.reply.startswith(expected)


def test_tc13_room_taken_before_yes(say, session_factory):
    """Someone else books the last Suites while we wait for "yes": the per-turn check catches it."""
    fill_to_confirmation(say, room="Suite")
    for _ in range(2):
        with session_factory() as db:
            crud.create_reservation(db, "Other Guest", "5550000000", "Suite", day(7), day(9), 2)

    r = say(intent="confirm")
    assert r.current_step == "room" and not r.is_complete and r.booking["room_type"] is None
    assert r.reply.startswith("Sorry, the Suite isn't available for those dates. For 2 guests")
    assert "Suite at" not in r.reply
    assert len(reservations(session_factory)) == 2

    r = say(room_type="Deluxe")  # name & phone are kept, so straight back to the read-back
    assert r.current_step == "awaiting_confirmation"
    assert say(intent="confirm").is_complete
    assert len(reservations(session_factory)) == 3


def test_tc13_room_taken_at_the_moment_of_booking(say, session_factory, monkeypatch):
    """The room goes in the instant between the check and the insert: create_reservation refuses."""
    fill_to_confirmation(say, room="Suite")

    def sold_out(*args, **kwargs):
        raise crud.RoomUnavailableError("gone")

    monkeypatch.setattr(crud, "create_reservation", sold_out)
    r = say(intent="confirm")
    assert r.current_step == "room" and not r.is_complete and r.booking["room_type"] is None
    assert r.reply.startswith("I'm sorry, the Suite was just booked by someone else. For 2 guests")
    assert reservations(session_factory) == []


def test_tc14_duplicate_yes_books_once(say, agent, extractor, session_factory):
    fill_to_confirmation(say)
    first = say(intent="confirm")
    calls_before = len(extractor.calls)
    second = agent.chat(SID, "yes")  # nothing scripted: the LLM must not be called
    assert len(extractor.calls) == calls_before
    assert second.reservation_id == first.reservation_id
    assert second.reply.startswith(f"Your booking is already confirmed, reference {first.reservation_id}")
    assert len(reservations(session_factory)) == 1


# --- unclear input & failures ----------------------------------------------------------


def test_tc11_unclear_input_changes_nothing(say):
    say(check_in=7, nights=2)
    r = say(intent="unclear")
    assert r.current_step == "guests" and r.booking["num_guests"] is None
    assert r.reply == f"{P.DIDNT_CATCH} {P.ASK_GUESTS}"


def test_greeting_gets_a_friendly_prompt(say):
    r = say(intent="off_topic", text="hello")
    assert r.reply == f"{P.OFF_TOPIC} {P.ASK_DATES}"


def test_type_hint_after_three_unclear_turns(say):
    replies = [say(intent="unclear").reply for _ in range(3)]
    assert P.TYPE_HINT not in replies[1] and replies[2].endswith(P.TYPE_HINT)
    r = say(check_in=7, nights=2)  # progress resets the counter
    assert P.TYPE_HINT not in r.reply


def test_llm_failure_keeps_state(say, agent, extractor):
    say(check_in=7, nights=2)
    extractor.push(TimeoutError("OpenAI timed out"))
    r = agent.chat(SID, "two people")
    assert r.reply == f"{P.ERROR_RETRY} {P.ASK_GUESTS}"
    assert r.current_step == "guests" and r.booking["check_in_date"] == day(7).isoformat()


# --- sessions & LLM context ------------------------------------------------------------


def test_sessions_are_isolated_and_resettable(say, agent):
    say(check_in=7, nights=2, session="a")
    r = say(intent="unclear", session="b")
    assert r.current_step == "dates"
    assert agent.state("a")["check_in_date"] == day(7).isoformat()

    agent.reset("a")
    assert agent.state("a") == {}


def test_llm_gets_today_step_known_values_and_last_reply(say, extractor):
    first = say(check_in=7, nights=2, text="next friday for two nights")
    say(num_guests=2, text="two of us")

    message, ctx = extractor.calls[1]
    assert message == "two of us"
    assert ctx.today == TODAY and ctx.step == "guests"
    assert ctx.known["check_in_date"] == day(7).isoformat() and ctx.known["num_guests"] is None
    assert ctx.last_reply == first.reply
    assert extractor.calls[0][1].last_reply == P.GREETING
