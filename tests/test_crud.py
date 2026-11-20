import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from backend.app import crud
from backend.app.database import SEED_ROOMS, init_db, make_engine, seed_rooms, write_transaction
from backend.app.models import STATUS_CANCELLED, Reservation, Room
from tests.conftest import day


def book(session_factory, room_type="Standard", ci=10, co=14, guests=2, name="Ada Lovelace", phone="+15551234567"):
    """Create a reservation in its own session (create_reservation needs a fresh transaction)."""
    with session_factory() as s:
        return crud.create_reservation(s, name, phone, room_type, day(ci), day(co), guests)


def by_type(rows):
    return {r.room_type: r for r in rows}


# --- Seeding -----------------------------------------------------------------


def test_seed_creates_catalog(db):
    rooms = {r.room_type: r for r in db.scalars(select(Room))}
    assert set(rooms) == {"Standard", "Deluxe", "Suite"}
    assert (rooms["Suite"].base_price_per_night, rooms["Suite"].total_inventory, rooms["Suite"].max_guests) == (280.0, 2, 4)


def test_seed_is_idempotent(engine, db):
    assert seed_rooms(db) is False
    init_db(engine)
    assert db.scalar(select(func.count()).select_from(Room)) == len(SEED_ROOMS)


# --- Availability ------------------------------------------------------------


def test_tc01_all_rooms_available_when_no_bookings(db):
    rows = by_type(crud.check_availability(db, day(10), day(14)))
    assert {k: (v.available_count, v.price_per_night) for k, v in rows.items()} == {
        "Standard": (5, 120.0),
        "Deluxe": (3, 180.0),
        "Suite": (2, 280.0),
    }


def test_tc02_sold_out_room_type_is_omitted(session_factory, db):
    for _ in range(5):
        book(session_factory, "Standard", 10, 14)

    available = by_type(crud.check_availability(db, day(11), day(13)))
    assert "Standard" not in available
    assert available["Deluxe"].available_count == 3
    assert available["Suite"].available_count == 2

    everything = by_type(crud.check_availability(db, day(11), day(13), only_available=False))
    assert everything["Standard"].available_count == 0


def test_back_to_back_stays_do_not_overlap(session_factory, db):
    for _ in range(2):
        book(session_factory, "Suite", 10, 14)
    # Checking in on the day the others check out is fine…
    assert "Suite" in by_type(crud.check_availability(db, day(14), day(16)))
    # …but overlapping by one night is not.
    assert "Suite" not in by_type(crud.check_availability(db, day(13), day(15)))


def test_cancelled_bookings_free_the_room(session_factory, db):
    ids = [book(session_factory, "Suite", 10, 14).reservation_id for _ in range(2)]
    db.get(Reservation, ids[0]).status = STATUS_CANCELLED
    db.commit()
    assert by_type(crud.check_availability(db, day(10), day(14)))["Suite"].available_count == 1


def test_filters_by_guest_capacity(db):
    assert set(by_type(crud.check_availability(db, day(1), day(2), num_guests=3))) == {"Deluxe", "Suite"}
    assert set(by_type(crud.check_availability(db, day(1), day(2), num_guests=4))) == {"Suite"}
    assert crud.check_availability(db, day(1), day(2), num_guests=5) == []
    assert crud.max_room_capacity(db) == 4


def test_filter_by_room_type_and_iso_strings(db):
    rows = crud.check_availability(db, day(1).isoformat(), day(3).isoformat(), room_type="Deluxe")
    assert [r.room_type for r in rows] == ["Deluxe"]


@pytest.mark.parametrize("ci,co", [(5, 5), (5, 3)])
def test_availability_rejects_bad_date_order(db, ci, co):
    with pytest.raises(crud.InvalidDatesError):
        crud.check_availability(db, day(ci), day(co))


def test_rejects_malformed_date_string(db):
    with pytest.raises(crud.InvalidDatesError):
        crud.check_availability(db, "next friday", day(3))


# --- Pricing -----------------------------------------------------------------


def test_quote_price_is_nights_times_rate(db):
    assert crud.quote_price(db, "Deluxe", day(10), day(13)) == 540.0
    assert crud.nights_between(day(10), day(13)) == 3


def test_quote_unknown_room_type(db):
    with pytest.raises(crud.UnknownRoomTypeError):
        crud.quote_price(db, "Penthouse", day(1), day(2))


# --- Creating reservations ---------------------------------------------------


def test_create_reservation_happy_path(session_factory, db):
    r = book(session_factory, "Deluxe", 10, 12, guests=3, name="  Grace Hopper ")
    assert r.reservation_id is not None
    assert (r.guest_name, r.total_price, r.status) == ("Grace Hopper", 360.0, "CONFIRMED")

    stored = db.get(Reservation, r.reservation_id)
    assert stored.check_in_date == day(10) and stored.check_out_date == day(12)


def test_create_reservation_rejects_over_capacity(session_factory, db):
    with pytest.raises(crud.CapacityError):
        book(session_factory, "Standard", guests=3)
    assert crud.list_reservations(db) == []


def test_create_reservation_rechecks_availability(session_factory, db):
    """TC-13 (DB part): the last Suite is booked between offer and commit."""
    offered = crud.check_availability(db, day(10), day(14), room_type="Suite")
    assert offered and offered[0].available_count == 2
    book(session_factory, "Suite")
    book(session_factory, "Suite")

    with pytest.raises(crud.RoomUnavailableError):
        book(session_factory, "Suite")
    db.rollback()  # end db's read snapshot so it sees the other sessions' commits
    assert len(crud.list_reservations(db)) == 2


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"name": " "}, crud.BookingError),
        ({"phone": ""}, crud.BookingError),
        ({"guests": 0}, crud.CapacityError),
        ({"ci": 14, "co": 10}, crud.InvalidDatesError),
        ({"room_type": "Penthouse"}, crud.UnknownRoomTypeError),
    ],
)
def test_create_reservation_validation(session_factory, db, kwargs, error):
    with pytest.raises(error):
        book(session_factory, **kwargs)
    assert crud.list_reservations(db) == []


def test_list_reservations_filters_status(session_factory, db):
    a = book(session_factory, ci=20, co=21)
    b = book(session_factory, ci=10, co=11)
    db.get(Reservation, a.reservation_id).status = STATUS_CANCELLED
    db.commit()
    assert [r.reservation_id for r in crud.list_reservations(db)] == [b.reservation_id]
    assert len(crud.list_reservations(db, status=None)) == 2


# --- Database constraints & locking -----------------------------------------


def test_db_rejects_inverted_dates(db):
    db.add(Reservation(guest_name="X", phone_number="1", room_type="Suite",
                       check_in_date=day(5), check_out_date=day(4), num_guests=1, total_price=1))
    with pytest.raises(IntegrityError):
        db.commit()


def test_db_enforces_foreign_key(db):
    db.add(Reservation(guest_name="X", phone_number="1", room_type="Penthouse",
                       check_in_date=day(4), check_out_date=day(5), num_guests=1, total_price=1))
    with pytest.raises(IntegrityError):
        db.commit()


def test_dates_stored_as_iso_text(session_factory, db):
    book(session_factory, ci=1, co=2)
    raw = db.connection().exec_driver_sql("SELECT check_in_date, typeof(check_in_date) FROM reservations").one()
    assert tuple(raw) == (day(1).isoformat(), "text")


def test_write_transaction_holds_the_write_lock(tmp_path):
    """While one write transaction is open, a second writer can't start, but readers can."""
    url = f"sqlite:///{tmp_path / 'lock.db'}"
    eng_a = make_engine(url)
    eng_b = make_engine(url, timeout=0)  # fail immediately instead of waiting for the lock
    init_db(eng_a)
    try:
        with Session(eng_a) as a, write_transaction(a):
            a.scalar(select(func.count()).select_from(Reservation))

            with Session(eng_b) as reader:
                assert reader.scalar(select(func.count()).select_from(Room)) == 3

            with Session(eng_b) as writer, pytest.raises(OperationalError, match="locked"):
                with write_transaction(writer):
                    pass
    finally:
        eng_a.dispose()
        eng_b.dispose()
