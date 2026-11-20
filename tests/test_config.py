from zoneinfo import ZoneInfoNotFoundError

import pytest

from backend.app.config import get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_defaults(monkeypatch):
    for var in ("OPENAI_MODEL", "BACKEND_PORT", "DATABASE_URL", "HOTEL_TIMEZONE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("backend.app.config.load_dotenv", lambda: None)

    s = get_settings()
    assert s.openai_model == "gpt-4o-mini"
    assert s.backend_port == 8000
    assert s.database_url == "sqlite:///./hotel_booking.db"
    assert s.today()  # resolves in the hotel timezone


def test_env_overrides(monkeypatch):
    monkeypatch.setattr("backend.app.config.load_dotenv", lambda: None)
    monkeypatch.setenv("BACKEND_PORT", "9001")
    monkeypatch.setenv("HOTEL_TIMEZONE", "Asia/Kolkata")

    s = get_settings()
    assert s.backend_port == 9001
    assert s.tz.key == "Asia/Kolkata"


def test_invalid_timezone_fails_fast(monkeypatch):
    monkeypatch.setattr("backend.app.config.load_dotenv", lambda: None)
    monkeypatch.setenv("HOTEL_TIMEZONE", "Not/AZone")

    with pytest.raises(ZoneInfoNotFoundError):
        get_settings()
