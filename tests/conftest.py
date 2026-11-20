from datetime import date, timedelta

import pytest

from backend.app.database import init_db, make_engine, make_session_factory

# Fixed "today" for every test (plan.md §6.1): never depend on the real clock.
TODAY = date(2026, 11, 20)  # a Friday


def day(n: int) -> date:
    """TODAY + n days."""
    return TODAY + timedelta(days=n)


@pytest.fixture
def engine(tmp_path):
    """Fresh, seeded SQLite file database per test (file-based, like production, so
    each session gets its own connection and real locking/WAL behaviour applies)."""
    eng = make_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def session_factory(engine):
    return make_session_factory(engine)


@pytest.fixture
def db(session_factory):
    with session_factory() as session:
        yield session
