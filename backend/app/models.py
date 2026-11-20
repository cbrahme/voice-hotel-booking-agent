"""SQLAlchemy models for room inventory and reservations (schema: plan.md §4.1)."""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import CheckConstraint, Date, DateTime, Float, ForeignKey, Index, Integer, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

STATUS_CONFIRMED = "CONFIRMED"
STATUS_CANCELLED = "CANCELLED"


class Base(DeclarativeBase):
    pass


class Room(Base):
    __tablename__ = "rooms"
    __table_args__ = (
        CheckConstraint("base_price_per_night > 0", name="ck_rooms_price_positive"),
        CheckConstraint("total_inventory >= 0", name="ck_rooms_inventory_nonnegative"),
        CheckConstraint("max_guests >= 1", name="ck_rooms_max_guests_positive"),
        {"sqlite_autoincrement": True},
    )

    room_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    room_type: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    base_price_per_night: Mapped[float] = mapped_column(Float, nullable=False)
    total_inventory: Mapped[int] = mapped_column(Integer, nullable=False)
    max_guests: Mapped[int] = mapped_column(Integer, nullable=False, default=2, server_default="2")

    def __repr__(self) -> str:
        return f"Room({self.room_type!r}, ${self.base_price_per_night}/night, x{self.total_inventory}, max {self.max_guests})"


class Reservation(Base):
    __tablename__ = "reservations"
    __table_args__ = (
        CheckConstraint("check_out_date > check_in_date", name="ck_reservations_dates_ordered"),
        CheckConstraint("num_guests >= 1", name="ck_reservations_guests_positive"),
        CheckConstraint(
            f"status IN ('{STATUS_CONFIRMED}', '{STATUS_CANCELLED}')", name="ck_reservations_status"
        ),
        Index("idx_reservations_overlap", "room_type", "status", "check_in_date", "check_out_date"),
        {"sqlite_autoincrement": True},  # never reuse a reservation ID
    )

    reservation_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    guest_name: Mapped[str] = mapped_column(String, nullable=False)
    phone_number: Mapped[str] = mapped_column(String, nullable=False)
    room_type: Mapped[str] = mapped_column(ForeignKey("rooms.room_type"), nullable=False)
    # SQLAlchemy's Date type is stored by SQLite as TEXT 'YYYY-MM-DD', matching the spec.
    check_in_date: Mapped[date] = mapped_column(Date, nullable=False)
    check_out_date: Mapped[date] = mapped_column(Date, nullable=False)
    num_guests: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    total_price: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(
        String, nullable=False, default=STATUS_CONFIRMED, server_default=STATUS_CONFIRMED
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())

    def __repr__(self) -> str:
        return (
            f"Reservation(#{self.reservation_id} {self.room_type} {self.check_in_date}→{self.check_out_date} "
            f"{self.guest_name!r} {self.status})"
        )
