"""FastAPI endpoint tests (TC-08), using the scripted fake LLM and a temporary database."""

import pytest
from fastapi.testclient import TestClient

from backend.app import crud, main
from backend.app.agent import prompts as P
from backend.app.agent.state import TurnExtraction
from backend.app.config import Settings
from tests.conftest import day


@pytest.fixture
def client(agent, session_factory, tmp_path):
    app = main.create_app(agent=agent, session_factory=session_factory, frontend_dir=None)
    with TestClient(app) as c:
        yield c


def chat(client, extractor, message="(speech)", session_id="s1", **extraction):
    extraction.setdefault("intent", "provide")
    extractor.push(TurnExtraction(**extraction))
    return client.post("/api/chat", json={"session_id": session_id, "user_message": message})


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


# --- /api/chat -------------------------------------------------------------------------


def test_tc08_chat_response_shape(client, extractor):
    r = chat(client, extractor, check_in=day(7).isoformat(), nights=2)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"reply_text", "current_step", "booking_summary", "is_complete", "reservation_id"}
    assert body["current_step"] == "guests" and body["is_complete"] is False and body["reservation_id"] is None
    assert body["booking_summary"]["check_in_date"] == day(7).isoformat()
    assert body["booking_summary"]["nights"] == 2
    assert body["reply_text"].startswith("Got it: Friday, November 27th")
    timing = dict(part.split(";dur=") for part in r.headers["Server-Timing"].split(", "))
    assert set(timing) == {"llm", "total"} and float(timing["llm"]) <= float(timing["total"])


def test_no_llm_timing_once_booked(client, extractor):
    chat(client, extractor, check_in=day(7).isoformat(), nights=2, num_guests=2, room_type="Suite",
         guest_name="Ada Lovelace", phone_number="5551234567")
    chat(client, extractor, intent="confirm")
    r = client.post("/api/chat", json={"session_id": "s1", "user_message": "thanks"})  # no LLM call
    assert r.json()["is_complete"] and r.headers["Server-Timing"].startswith("total;dur=")


def test_full_booking_over_http(client, extractor, session_factory):
    chat(client, extractor, check_in=day(7).isoformat(), nights=2)
    chat(client, extractor, num_guests=2)
    chat(client, extractor, room_type="Suite")
    chat(client, extractor, guest_name="Ada Lovelace", phone_number="5551234567")
    body = chat(client, extractor, intent="confirm").json()

    assert body["is_complete"] and body["current_step"] == "confirmed"
    assert body["booking_summary"]["total_price"] == 560.0

    listed = client.get("/api/reservations").json()
    assert [r["reservation_id"] for r in listed] == [body["reservation_id"]]
    assert listed[0]["check_in_date"] == day(7).isoformat() and listed[0]["guest_name"] == "Ada Lovelace"


def test_sessions_are_separate(client, extractor):
    chat(client, extractor, session_id="a", check_in=day(7).isoformat(), nights=2)
    body = chat(client, extractor, session_id="b", intent="off_topic").json()
    assert body["current_step"] == "dates" and body["booking_summary"]["check_in_date"] is None


@pytest.mark.parametrize(
    "payload",
    [
        {"session_id": "s1", "user_message": "   "},
        {"session_id": "s1", "user_message": ""},
        {"session_id": "", "user_message": "hi"},
        {"session_id": "s1", "user_message": "x" * 1001},
        {"user_message": "hi"},
    ],
)
def test_chat_rejects_bad_payloads(client, extractor, payload):
    assert client.post("/api/chat", json=payload).status_code == 422
    assert extractor.calls == []  # the LLM is never called for invalid input


def test_chat_strips_whitespace_before_the_llm(client, extractor):
    chat(client, extractor, message="  two people  ", intent="unclear")
    assert extractor.calls[0][0] == "two people"


# --- /api/reset ------------------------------------------------------------------------


def test_reset_starts_over(client, extractor):
    chat(client, extractor, check_in=day(7).isoformat(), nights=2)
    r = client.post("/api/reset", json={"session_id": "s1"})
    assert r.json() == {"session_id": "s1", "reply_text": P.GREETING}

    body = chat(client, extractor, intent="unclear").json()
    assert body["current_step"] == "dates" and body["booking_summary"]["check_in_date"] is None


def test_reset_unknown_session_is_fine(client):
    assert client.post("/api/reset", json={"session_id": "never-used"}).status_code == 200


# --- /api/inventory --------------------------------------------------------------------


def test_inventory_catalog(client):
    rows = client.get("/api/inventory").json()
    assert [(r["room_type"], r["price_per_night"], r["max_guests"], r["available_count"]) for r in rows] == [
        ("Standard", 120.0, 2, None),
        ("Deluxe", 180.0, 3, None),
        ("Suite", 280.0, 4, None),
    ]
    assert [r["room_type"] for r in client.get("/api/inventory?num_guests=3").json()] == ["Deluxe", "Suite"]


def test_inventory_for_dates_includes_sold_out(client, session_factory):
    for _ in range(2):
        with session_factory() as db:
            crud.create_reservation(db, "Other Guest", "5550000000", "Suite", day(7), day(9), 2)
    rows = client.get("/api/inventory", params={"check_in": day(8).isoformat(), "check_out": day(10).isoformat()}).json()
    assert {r["room_type"]: r["available_count"] for r in rows} == {"Standard": 5, "Deluxe": 3, "Suite": 0}


@pytest.mark.parametrize(
    "params,status",
    [
        ({"check_in": "2026-12-01"}, 400),  # only one date
        ({"check_in": "2026-12-05", "check_out": "2026-12-01"}, 400),  # inverted
        ({"check_in": "next friday", "check_out": "2026-12-01"}, 422),  # not a date
        ({"num_guests": 0}, 422),
    ],
)
def test_inventory_bad_params(client, params, status):
    assert client.get("/api/inventory", params=params).status_code == status


# --- /api/reservations -----------------------------------------------------------------


def test_reservations_hide_cancelled_by_default(client, session_factory):
    with session_factory() as db:
        r = crud.create_reservation(db, "Ada Lovelace", "5551234567", "Standard", day(7), day(9), 1)
    with session_factory() as db:
        db.get(crud.Reservation, r.reservation_id).status = "CANCELLED"
        db.commit()
    assert client.get("/api/reservations").json() == []
    assert len(client.get("/api/reservations?include_cancelled=true").json()) == 1


# --- startup & static files ------------------------------------------------------------


def test_startup_fails_clearly_without_api_key(monkeypatch, session_factory):
    monkeypatch.setattr(main, "get_settings", lambda: Settings(None, "gpt-4o-mini", 8000, "sqlite://", "UTC"))
    app = main.create_app(session_factory=session_factory, frontend_dir=None)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        with TestClient(app):
            pass


def test_serves_the_real_frontend(agent, session_factory):
    app = main.create_app(agent=agent, session_factory=session_factory)  # default frontend folder
    with TestClient(app) as c:
        page = c.get("/")
        assert page.status_code == 200 and 'id="mic"' in page.text
        for asset, kind in (("app.js", "javascript"), ("styles.css", "text/css")):
            r = c.get(f"/{asset}")
            assert r.status_code == 200 and kind in r.headers["content-type"]


def test_serves_frontend_without_shadowing_api(agent, session_factory, tmp_path):
    (tmp_path / "index.html").write_text("<h1>voice booking</h1>")
    app = main.create_app(agent=agent, session_factory=session_factory, frontend_dir=tmp_path)
    with TestClient(app) as c:
        assert "voice booking" in c.get("/").text
        assert c.get("/api/health").json() == {"status": "ok"}
