"""Availability, pricing and reservation logic (plan.md §4.3).

Prices are always computed here from the database, never taken from LLM output.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .database import write_transaction
from .models import STATUS_CONFIRMED, Reservation, Room


class BookingError(ValueError):
    """Base class for booking rule violations. Messages are safe to show to the user."""


class InvalidDatesError(BookingError):
    pass


class UnknownRoomTypeError(BookingError):
    pass


class CapacityError(BookingError):
    pass


class RoomUnavailableError(BookingError):
    pass


@dataclass(frozen=True)
class RoomAvailability:
    room_type: str
    price_per_night: float
    max_guests: int
    total_inventory: int
    available_count: int

    @property
    def is_available(self) -> bool:
        return self.available_count > 0


def to_date(value: date | str) -> date:
    """Accept a ``date`` or an ISO ``YYYY-MM-DD`` string."""
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise InvalidDatesError(f"Invalid date {value!r}; expected YYYY-MM-DD.") from exc


def nights_between(check_in: date | str, check_out: date | str) -> int:
    """Number of nights; raises InvalidDatesError unless check_out is after check_in."""
    ci, co = to_date(check_in), to_date(check_out)
    nights = (co - ci).days
    if nights <= 0:
        raise InvalidDatesError("Check-out must be after check-in.")
    return nights


def _booked_counts(db: Session, check_in: date, check_out: date, room_type: str | None = None) -> dict[str, int]:
    """Confirmed bookings per room type that overlap [check_in, check_out)."""
    stmt = (
        select(Reservation.room_type, func.count())
        .where(
            Reservation.status == STATUS_CONFIRMED,
            Reservation.check_in_date < check_out,
            Reservation.check_out_date > check_in,
        )
        .group_by(Reservation.room_type)
    )
    if room_type is not None:
        stmt = stmt.where(Reservation.room_type == room_type)
    return {rt: count for rt, count in db.execute(stmt)}


def check_availability(
    db: Session,
    check_in: date | str,
    check_out: date | str,
    num_guests: int | None = None,
    room_type: str | None = None,
    only_available: bool = True,
) -> list[RoomAvailability]:
    """Room types for the stay, cheapest first.

    Filters out room types too small for ``num_guests`` (when given). With
    ``only_available=False`` sold-out types are included with ``available_count == 0``.
    """
    ci, co = to_date(check_in), to_date(check_out)
    nights_between(ci, co)

    stmt = select(Room).order_by(Room.base_price_per_night)
    if num_guests is not None:
        stmt = stmt.where(Room.max_guests >= num_guests)
    if room_type is not None:
        stmt = stmt.where(Room.room_type == room_type)
    rooms = db.scalars(stmt).all()

    booked = _booked_counts(db, ci, co, room_type)
    result = [
        RoomAvailability(
            room_type=r.room_type,
            price_per_night=r.base_price_per_night,
            max_guests=r.max_guests,
            total_inventory=r.total_inventory,
            available_count=max(r.total_inventory - booked.get(r.room_type, 0), 0),
        )
        for r in rooms
    ]
    return [r for r in result if r.is_available] if only_available else result


def get_room(db: Session, room_type: str) -> Room:
    room = db.scalar(select(Room).where(Room.room_type == room_type))
    if room is None:
        raise UnknownRoomTypeError(f"We don't have a room type called {room_type!r}.")
    return room


def max_room_capacity(db: Session) -> int:
    return db.scalar(select(func.max(Room.max_guests))) or 0


def quote_price(db: Session, room_type: str, check_in: date | str, check_out: date | str) -> float:
    """Total price = nights × base price per night, from the database."""
    room = get_room(db, room_type)
    return round(nights_between(check_in, check_out) * room.base_price_per_night, 2)


def create_reservation(
    db: Session,
    guest_name: str,
    phone_number: str,
    room_type: str,
    check_in: date | str,
    check_out: date | str,
    num_guests: int,
) -> Reservation:
    """Re-check capacity and availability, compute the price and insert, all in one
    ``BEGIN IMMEDIATE`` transaction so the room can't be taken in between.

    ``db`` must not have an open transaction. Raises a :class:`BookingError` subclass
    if any rule fails; nothing is written in that case.
    """
    guest_name, phone_number = guest_name.strip(), phone_number.strip()
    if not guest_name or not phone_number:
        raise BookingError("Guest name and phone number are required.")
    if num_guests < 1:
        raise CapacityError("At least one guest is required.")
    ci, co = to_date(check_in), to_date(check_out)
    nights = nights_between(ci, co)

    with write_transaction(db):
        room = get_room(db, room_type)
        if num_guests > room.max_guests:
            raise CapacityError(f"The {room.room_type} room fits up to {room.max_guests} guests.")
        booked = _booked_counts(db, ci, co, room.room_type).get(room.room_type, 0)
        if booked >= room.total_inventory:
            raise RoomUnavailableError(f"Sorry, the {room.room_type} room is no longer available for those dates.")

        reservation = Reservation(
            guest_name=guest_name,
            phone_number=phone_number,
            room_type=room.room_type,
            check_in_date=ci,
            check_out_date=co,
            num_guests=num_guests,
            total_price=round(nights * room.base_price_per_night, 2),
            status=STATUS_CONFIRMED,
        )
        db.add(reservation)
        db.flush()  # assigns reservation_id inside the transaction
    return reservation


def list_reservations(db: Session, status: str | None = STATUS_CONFIRMED) -> list[Reservation]:
    stmt = select(Reservation).order_by(Reservation.check_in_date, Reservation.reservation_id)
    if status is not None:
        stmt = stmt.where(Reservation.status == status)
    return list(db.scalars(stmt).all())
