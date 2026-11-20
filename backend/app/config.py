"""Application settings loaded from environment variables / `.env`."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from zoneinfo import ZoneInfo

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    openai_model: str
    backend_port: int
    database_url: str
    hotel_timezone: str

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.hotel_timezone)

    def today(self) -> date:
        """Today's date in the hotel's timezone (the anchor for date validation)."""
        return datetime.now(self.tz).date()


@lru_cache
def get_settings() -> Settings:
    load_dotenv()
    settings = Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        backend_port=int(os.getenv("BACKEND_PORT", "8000")),
        database_url=os.getenv("DATABASE_URL", "sqlite:///./hotel_booking.db"),
        hotel_timezone=os.getenv("HOTEL_TIMEZONE", "America/New_York"),
    )
    settings.tz  # fail fast on an invalid timezone name
    return settings
