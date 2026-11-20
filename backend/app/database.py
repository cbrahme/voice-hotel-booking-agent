"""SQLite engine, session factory, write-transaction helper and idempotent seeding."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

from sqlalchemy import Engine, create_engine, event, select
from sqlalchemy.orm import Session, sessionmaker

from .config import get_settings
from .models import Base, Room

SEED_ROOMS: tuple[dict[str, Any], ...] = (
    {"room_type": "Standard", "base_price_per_night": 120.00, "total_inventory": 5, "max_guests": 2},
    {"room_type": "Deluxe", "base_price_per_night": 180.00, "total_inventory": 3, "max_guests": 3},
    {"room_type": "Suite", "base_price_per_night": 280.00, "total_inventory": 2, "max_guests": 4},
)

# Execution option read by the "begin" hook below to choose the SQLite BEGIN mode.
_BEGIN_MODE_OPTION = "sqlite_begin_mode"


def make_engine(url: str, **connect_args: Any) -> Engine:
    """Create an engine. For SQLite, also:

    - enables foreign keys;
    - uses WAL journaling, so open read transactions never block a writer's commit
      (and vice versa) — important with several concurrent sessions;
    - lets SQLAlchemy (not the sqlite3 driver) emit BEGIN, so a transaction can be
      opened as ``BEGIN IMMEDIATE`` (see :func:`write_transaction`).
    """
    is_sqlite = url.startswith("sqlite")
    if is_sqlite:
        connect_args.setdefault("check_same_thread", False)
    engine = create_engine(url, connect_args=connect_args)

    if is_sqlite:

        @event.listens_for(engine, "connect")
        def _on_connect(dbapi_conn, _record):
            dbapi_conn.isolation_level = None  # stop pysqlite from issuing its own BEGIN
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA journal_mode=WAL")
            cur.close()

        @event.listens_for(engine, "begin")
        def _on_begin(conn):
            mode = conn.get_execution_options().get(_BEGIN_MODE_OPTION, "DEFERRED")
            conn.exec_driver_sql(f"BEGIN {mode}")

    return engine


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def write_transaction(db: Session) -> Iterator[Session]:
    """Run a block in a transaction that takes SQLite's write lock up front (``BEGIN IMMEDIATE``).

    Any read inside the block is therefore consistent with the write that follows: no other
    writer can slip in between (e.g. booking the last room after we counted it as free).
    Commits on success, rolls back on error. ``db`` must not already be in a transaction.
    """
    with db.begin():
        db.connection(execution_options={_BEGIN_MODE_OPTION: "IMMEDIATE"})
        yield db


def seed_rooms(db: Session) -> bool:
    """Insert the default room catalog if the rooms table is empty. Returns True if it seeded."""
    if db.scalar(select(Room.room_id).limit(1)) is not None:
        return False
    db.add_all(Room(**row) for row in SEED_ROOMS)
    db.commit()
    return True


def init_db(engine: Engine) -> None:
    """Create tables (if missing) and seed the room catalog (if empty)."""
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        seed_rooms(db)


@lru_cache
def get_engine() -> Engine:
    return make_engine(get_settings().database_url)


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    return make_session_factory(get_engine())


def get_db() -> Iterator[Session]:
    """FastAPI dependency: one session per request."""
    with get_session_factory()() as db:
        yield db
