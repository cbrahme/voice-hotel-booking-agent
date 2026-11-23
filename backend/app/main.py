"""FastAPI backend: chat endpoints for the booking agent, admin/inspection endpoints,
and the static frontend page.

Run:  uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session, sessionmaker

from . import crud
from .agent.graph import BookingAgent
from .agent.prompts import GREETING
from .config import get_settings
from .database import get_engine, get_session_factory, init_db

log = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"


# --------------------------------------------------------------------------- schemas


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    user_message: str = Field(max_length=1000)

    @field_validator("user_message")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("user_message must not be empty")
        return v


class BookingSummary(BaseModel):
    check_in_date: Optional[str]
    check_out_date: Optional[str]
    nights: Optional[int]
    num_guests: Optional[int]
    room_type: Optional[str]
    guest_name: Optional[str]
    phone_number: Optional[str]
    total_price: Optional[float]


class ChatResponse(BaseModel):
    reply_text: str
    current_step: str
    booking_summary: BookingSummary
    is_complete: bool
    reservation_id: Optional[int]


class ResetRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)


class ResetResponse(BaseModel):
    session_id: str
    reply_text: str  # the greeting to show/speak for the fresh conversation


class RoomInfo(BaseModel):
    room_type: str
    price_per_night: float
    max_guests: int
    total_inventory: int
    available_count: Optional[int] = None  # only when dates are given


class ReservationOut(BaseModel):
    reservation_id: int
    guest_name: str
    phone_number: str
    room_type: str
    check_in_date: date
    check_out_date: date
    num_guests: int
    total_price: float
    status: str


# --------------------------------------------------------------------------- app


class _SessionLocks:
    """One lock per session, so two requests for the same conversation can't interleave."""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._locks: dict[str, threading.Lock] = defaultdict(threading.Lock)

    def get(self, session_id: str) -> threading.Lock:
        with self._guard:
            return self._locks[session_id]

    def discard(self, session_id: str) -> None:
        with self._guard:
            self._locks.pop(session_id, None)


def create_app(
    agent: BookingAgent | None = None,
    session_factory: sessionmaker[Session] | None = None,
    frontend_dir: Path | None = FRONTEND_DIR,
) -> FastAPI:
    """Build the app. Anything not passed in is created at startup from settings
    (database from DATABASE_URL, agent backed by the real OpenAI model)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.session_factory is None:
            init_db(get_engine())
            app.state.session_factory = get_session_factory()
        if app.state.agent is None:
            from .agent.extractor import OpenAIExtractor

            settings = get_settings()
            if not settings.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
            extractor = OpenAIExtractor(settings.openai_model, settings.openai_api_key)
            app.state.agent = BookingAgent(extractor, app.state.session_factory, today_fn=settings.today)
        yield

    app = FastAPI(title="Voice Hotel Booking Agent", lifespan=lifespan)
    app.state.agent = agent
    app.state.session_factory = session_factory
    app.state.locks = _SessionLocks()

    # Plain `def` endpoints run in FastAPI's thread pool, so the blocking LLM call
    # in agent.chat doesn't stall other requests.

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/chat", response_model=ChatResponse)
    def chat(body: ChatRequest, request: Request, response: Response) -> ChatResponse:
        state = request.app.state
        started = time.perf_counter()
        with state.locks.get(body.session_id):
            result = state.agent.chat(body.session_id, body.user_message)
        total_ms = (time.perf_counter() - started) * 1000
        timing = [f"total;dur={total_ms:.0f}"]
        if result.llm_ms is not None:
            timing.insert(0, f"llm;dur={result.llm_ms:.0f}")
        response.headers["Server-Timing"] = ", ".join(timing)
        log.info("turn session=%s step=%s total_ms=%.0f llm_ms=%s", body.session_id, result.current_step,
                 total_ms, f"{result.llm_ms:.0f}" if result.llm_ms is not None else "-")
        return ChatResponse(
            reply_text=result.reply,
            current_step=result.current_step,
            booking_summary=BookingSummary(**result.booking),
            is_complete=result.is_complete,
            reservation_id=result.reservation_id,
        )

    @app.post("/api/reset", response_model=ResetResponse)
    def reset(body: ResetRequest, request: Request) -> ResetResponse:
        state = request.app.state
        with state.locks.get(body.session_id):
            state.agent.reset(body.session_id)
        state.locks.discard(body.session_id)
        return ResetResponse(session_id=body.session_id, reply_text=GREETING)

    @app.get("/api/inventory", response_model=list[RoomInfo])
    def inventory(
        request: Request,
        check_in: Optional[date] = Query(None, description="YYYY-MM-DD"),
        check_out: Optional[date] = Query(None, description="YYYY-MM-DD"),
        num_guests: Optional[int] = Query(None, ge=1),
    ) -> list[Any]:
        """Room catalog; with check_in and check_out, also how many of each are free."""
        if (check_in is None) != (check_out is None):
            raise HTTPException(400, "Give both check_in and check_out, or neither.")
        with request.app.state.session_factory() as db:
            if check_in is None:
                return [
                    RoomInfo(room_type=r.room_type, price_per_night=r.base_price_per_night,
                             max_guests=r.max_guests, total_inventory=r.total_inventory)
                    for r in crud.list_rooms(db)
                    if num_guests is None or r.max_guests >= num_guests
                ]
            try:
                rows = crud.check_availability(db, check_in, check_out, num_guests, only_available=False)
            except crud.BookingError as exc:
                raise HTTPException(400, str(exc)) from exc
            return [RoomInfo(**vars(r)) for r in rows]

    @app.get("/api/reservations", response_model=list[ReservationOut])
    def reservations(request: Request, include_cancelled: bool = False) -> list[Any]:
        """Bookings, for admin/testing. Contains guest contact details: local use only."""
        with request.app.state.session_factory() as db:
            rows = crud.list_reservations(db, status=None if include_cancelled else crud.STATUS_CONFIRMED)
            return [ReservationOut.model_validate(r, from_attributes=True) for r in rows]

    # The voice web page (Phase 5). Mounted last so /api/* routes take priority.
    if frontend_dir is not None and frontend_dir.is_dir():
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

    return app


app = create_app()
