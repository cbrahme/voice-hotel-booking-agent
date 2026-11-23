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


class ScriptedExtractor:
    """Stands in for the LLM: returns queued TurnExtractions in order and records
    every call, so agent tests run offline and deterministically."""

    def __init__(self):
        self.queue = []
        self.calls = []  # (message, ExtractionContext)

    def push(self, *extractions):
        self.queue.extend(extractions)

    def extract(self, message, context):
        self.calls.append((message, context))
        if not self.queue:
            raise AssertionError(f"no scripted extraction left for message {message!r}")
        item = self.queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def extractor():
    return ScriptedExtractor()


@pytest.fixture
def agent(extractor, session_factory):
    from backend.app.agent.graph import BookingAgent

    return BookingAgent(extractor, session_factory, today_fn=lambda: TODAY)
