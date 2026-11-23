"""The booking conversation as a LangGraph state machine.

Each user message runs one pass through the graph:

    extract ──► understand ──► collect_dates | collect_guests | collect_room_type |
    (1 LLM call)  (Python)      collect_contact | confirm_booking | book | ask_change |
                                clarify | done ──► END

- ``extract`` asks the LLM what the user said (a TurnExtraction). It is the only LLM call.
- ``understand`` validates the extracted values, merges them into the state, clears values
  that depend on anything that changed (backtracking), and picks the next node.
- Every other node writes the spoken reply from a template in ``prompts.py``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session, sessionmaker

from .. import crud
from ..utils.date_parser import parse_iso, speak_date, validate_check_in, validate_range
from ..utils.validators import clean_name, normalize_phone, speak_phone
from . import prompts as P
from .extractor import ExtractionContext, Extractor
from .state import BookingState, TurnExtraction

log = logging.getLogger(__name__)

SLOTS = ("check_in_date", "check_out_date", "num_guests", "room_type", "guest_name", "phone_number")

STEP_NODES = {
    "dates": "collect_dates",
    "guests": "collect_guests",
    "room": "collect_room_type",
    "contact": "collect_contact",
    "awaiting_confirmation": "confirm_booking",
    "confirmed": "done",
}

CLARIFY_HINT_AFTER = 3  # consecutive turns without progress before suggesting the text box


def step_for(state: dict) -> str:
    """The first step whose information is still missing."""
    if state.get("reservation_id"):
        return "confirmed"
    if not (state.get("check_in_date") and state.get("check_out_date")):
        return "dates"
    if not state.get("num_guests"):
        return "guests"
    if not state.get("room_type"):
        return "room"
    if not (state.get("guest_name") and state.get("phone_number")):
        return "contact"
    return "awaiting_confirmation"


# --------------------------------------------------------------------------- merging


def merge_turn(
    state: dict, ext: TurnExtraction, today: date, db: Session
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Apply one turn's extraction to the booking fields.

    Returns ``(fields, changed, notices)``: the new values of all booking fields plus
    ``total_price``, which kinds of fields changed, and sentences to say to the user
    (validation problems, a room that no longer fits, ...). Invalid values are never stored.
    """
    s: dict[str, Any] = {k: state.get(k) for k in SLOTS}
    changed: list[str] = []
    notices: list[str] = []

    # --- dates
    if ext.check_in or ext.check_out or ext.nights:
        old_ci, old_co = parse_iso(s["check_in_date"]), parse_iso(s["check_out_date"])
        new_ci, new_co = parse_iso(ext.check_in), parse_iso(ext.check_out)
        if (ext.check_in and not new_ci) or (ext.check_out and not new_co):
            notices.append(P.BAD_DATE)
        else:
            ci = new_ci or old_ci
            co = new_co
            if co is None and ext.nights is not None and ci:
                co = ci + timedelta(days=ext.nights)
            if co is None:
                co = old_co
                # Check-in moved on or past the old check-out: keep the same length of stay.
                if new_ci and old_ci and old_co and co <= new_ci:
                    co = new_ci + (old_co - old_ci)
            if ci and co:
                errors = validate_range(ci, co, today)
            elif ci:
                errors = validate_check_in(ci, today)
            elif co:
                errors = validate_check_in(co, today)
            else:
                errors = []
            if errors:
                notices.extend(errors)
            else:
                new_values = (ci.isoformat() if ci else None, co.isoformat() if co else None)
                if new_values != (s["check_in_date"], s["check_out_date"]):
                    s["check_in_date"], s["check_out_date"] = new_values
                    changed.append("dates")

    # --- guests
    if ext.num_guests is not None:
        largest = crud.largest_room(db)
        if ext.num_guests < 1:
            notices.append(P.BAD_GUEST_COUNT)
        elif largest and ext.num_guests > largest.max_guests:
            notices.append(P.TOO_MANY_GUESTS.format(room=largest.room_type, max_guests=largest.max_guests))
        elif ext.num_guests != s["num_guests"]:
            s["num_guests"] = ext.num_guests
            changed.append("guests")

    # --- room type
    if ext.room_type and ext.room_type != s["room_type"]:
        s["room_type"] = ext.room_type
        changed.append("room")

    # --- contact
    if ext.guest_name is not None:
        name = clean_name(ext.guest_name)
        if name is None:
            notices.append(P.BAD_NAME)
        elif name != s["guest_name"]:
            s["guest_name"] = name
            changed.append("name")
    if ext.phone_number is not None:
        phone = normalize_phone(ext.phone_number)
        if phone is None:
            notices.append(P.BAD_PHONE)
        elif phone != s["phone_number"]:
            s["phone_number"] = phone
            changed.append("phone")

    # --- the chosen room must still fit the party and be free on the dates
    ci, co, guests, room = s["check_in_date"], s["check_out_date"], s["num_guests"], s["room_type"]
    if room and ci and co and guests:
        if not crud.check_availability(db, ci, co, num_guests=guests, room_type=room):
            room_obj = crud.get_room(db, room)
            if room_obj.max_guests < guests:
                notices.append(P.ROOM_TOO_SMALL.format(room=room, max_guests=room_obj.max_guests))
            else:
                notices.append(P.ROOM_SOLD_OUT.format(room=room))
            s["room_type"] = None
            if "room" not in changed:
                changed.append("room")

    # --- price is always recomputed from the database
    s["total_price"] = crud.quote_price(db, s["room_type"], ci, co) if (s["room_type"] and ci and co) else None
    return s, changed, notices


# --------------------------------------------------------------------------- wording


def _nights(state: dict) -> int:
    return (parse_iso(state["check_out_date"]) - parse_iso(state["check_in_date"])).days


def _first_name(state: dict) -> str:
    return (state.get("guest_name") or "").split(" ")[0]


def _dates_ack(state: dict, today: date) -> Optional[str]:
    if "dates" not in (state.get("last_changed_slots") or []):
        return None
    if not (state.get("check_in_date") and state.get("check_out_date")):
        return None
    return P.DATES_ACK.format(
        check_in=speak_date(parse_iso(state["check_in_date"]), today),
        check_out=speak_date(parse_iso(state["check_out_date"]), today),
        nights=P.plural(_nights(state), "night"),
    )


def question_for(step: str, state: dict, today: date, db: Session) -> str:
    """The question to ask for a step, built from the current (validated) state."""
    if step == "dates":
        ci, co = parse_iso(state.get("check_in_date")), parse_iso(state.get("check_out_date"))
        if ci and not co:
            return P.ASK_CHECK_OUT.format(check_in=speak_date(ci, today))
        if co and not ci:
            return P.ASK_CHECK_IN.format(check_out=speak_date(co, today))
        return P.ASK_DATES
    if step == "guests":
        return P.ASK_GUESTS
    if step == "room":
        guests = P.plural(state["num_guests"], "guest")
        rooms = crud.check_availability(db, state["check_in_date"], state["check_out_date"], state["num_guests"])
        if not rooms:
            return P.NO_ROOMS.format(guests=guests)
        options = P.join_options(
            [P.ROOM_OPTION.format(room=r.room_type, price=P.money(r.price_per_night)) for r in rooms]
        )
        template = P.ROOM_SINGLE_OPTION if len(rooms) == 1 else P.ROOM_OPTIONS
        return template.format(guests=guests, options=options)
    if step == "contact":
        if not state.get("guest_name") and not state.get("phone_number"):
            return P.ASK_NAME_AND_PHONE.format(
                room=state["room_type"],
                nights=P.plural(_nights(state), "night"),
                price=P.money(state["total_price"]),
            )
        if not state.get("guest_name"):
            return P.ASK_NAME
        return P.ASK_PHONE.format(first_name=_first_name(state))
    if step == "awaiting_confirmation":
        return P.READ_BACK.format(
            room=state["room_type"],
            guests=P.plural(state["num_guests"], "guest"),
            check_in=speak_date(parse_iso(state["check_in_date"]), today),
            check_out=speak_date(parse_iso(state["check_out_date"]), today),
            nights=P.plural(_nights(state), "night"),
            price=P.money(state["total_price"]),
            name=state["guest_name"],
            phone=speak_phone(state["phone_number"]),
        )
    if step == "confirmed":
        return P.ALREADY_BOOKED.format(reservation_id=state["reservation_id"])
    raise ValueError(f"unknown step {step!r}")


def _say(state: dict, *parts: Optional[str]) -> dict:
    """Join notices + parts into the reply and append it to the conversation."""
    pieces = [*(state.get("notices") or []), *(p for p in parts if p)]
    if state.get("clarify_count", 0) >= CLARIFY_HINT_AFTER:
        pieces.append(P.TYPE_HINT)
    text = " ".join(pieces)
    return {"reply": text, "messages": [AIMessage(text)]}


# --------------------------------------------------------------------------- graph


def build_graph(
    extractor: Extractor,
    session_factory: sessionmaker[Session],
    today_fn: Callable[[], date],
) -> StateGraph:
    """Build the (uncompiled) booking graph. Dependencies are injected so tests can
    use a fake extractor, a temporary database and a fixed 'today'."""

    def extract(state: BookingState) -> dict:
        if state.get("reservation_id"):
            return {"extraction": None}  # booking finished: no need to call the LLM
        messages = state.get("messages") or []
        user_text = messages[-1].content if messages else ""
        last_reply = next((m.content for m in reversed(messages[:-1]) if isinstance(m, AIMessage)), P.GREETING)
        context = ExtractionContext(
            today=today_fn(),
            step=state.get("current_step") or "dates",
            known={k: state.get(k) for k in SLOTS},
            last_reply=last_reply,
        )
        try:
            return {"extraction": extractor.extract(user_text, context).model_dump()}
        except Exception:
            log.exception("extraction failed")
            return {"extraction": None}

    def understand(state: BookingState) -> dict:
        if state.get("reservation_id"):
            return {"route": "done", "notices": [], "last_changed_slots": []}
        if state.get("extraction") is None:
            return {"route": "clarify", "notices": [P.ERROR_RETRY], "last_changed_slots": []}

        ext = TurnExtraction.model_validate(state["extraction"])
        prev_step = state.get("current_step") or "dates"
        today = today_fn()
        with session_factory() as db:
            # "Yes" when only one room was offered means "that one".
            if prev_step == "room" and ext.intent == "confirm" and not ext.room_type:
                offered = crud.check_availability(
                    db, state["check_in_date"], state["check_out_date"], state["num_guests"]
                )
                if len(offered) == 1:
                    ext = ext.model_copy(update={"room_type": offered[0].room_type})
            fields, changed, notices = merge_turn(state, ext, today, db)

        new_step = step_for({**state, **fields})
        if changed or notices:
            route = STEP_NODES[new_step]
        elif prev_step == "awaiting_confirmation" and ext.intent == "confirm":
            route = "book"
        elif prev_step == "awaiting_confirmation" and ext.intent == "deny":
            route = "ask_change"
        else:
            route = "clarify"

        no_progress = not changed and route not in ("book", "ask_change")
        return {
            **fields,
            "current_step": new_step,
            "route": route,
            "notices": notices,
            "last_changed_slots": changed,
            "clarify_count": state.get("clarify_count", 0) + 1 if no_progress else 0,
        }

    def step_node(step: str):
        def node(state: BookingState) -> dict:
            today = today_fn()
            # The read-back already states the dates, so don't acknowledge them twice.
            ack = None if step == "awaiting_confirmation" else _dates_ack(state, today)
            with session_factory() as db:
                return _say(state, ack, question_for(step, state, today, db))

        return node

    def clarify(state: BookingState) -> dict:
        step = state.get("current_step") or "dates"
        intent = (state.get("extraction") or {}).get("intent")
        prefix = None
        if not state.get("notices"):
            prefix = P.OFF_TOPIC if intent == "off_topic" else P.DIDNT_CATCH
        if step == "awaiting_confirmation":
            return _say(state, prefix, P.ASK_YES_NO)
        today = today_fn()
        with session_factory() as db:
            return _say(state, prefix, question_for(step, state, today, db))

    def ask_change(state: BookingState) -> dict:
        return _say(state, P.ASK_WHAT_TO_CHANGE)

    def book(state: BookingState) -> dict:
        today = today_fn()
        with session_factory() as db:
            try:
                reservation = crud.create_reservation(
                    db,
                    guest_name=state["guest_name"],
                    phone_number=state["phone_number"],
                    room_type=state["room_type"],
                    check_in=state["check_in_date"],
                    check_out=state["check_out_date"],
                    num_guests=state["num_guests"],
                )
            except crud.RoomUnavailableError:
                cleared = {"room_type": None, "total_price": None, "current_step": "room"}
                retry_state = {**state, **cleared, "notices": [P.ROOM_JUST_TAKEN.format(room=state["room_type"])]}
                return {**cleared, **_say(retry_state, question_for("room", retry_state, today, db))}
            except crud.BookingError as exc:
                log.warning("booking rejected: %s", exc)
                return _say({**state, "notices": [str(exc)]}, P.ASK_WHAT_TO_CHANGE)

        return {
            "reservation_id": reservation.reservation_id,
            "total_price": reservation.total_price,
            "current_step": "confirmed",
            **_say(
                state,
                P.BOOKED.format(first_name=_first_name(state), reservation_id=reservation.reservation_id),
            ),
        }

    def done(state: BookingState) -> dict:
        return _say(state, P.ALREADY_BOOKED.format(reservation_id=state["reservation_id"]))

    builder = StateGraph(BookingState)
    builder.add_node("extract", extract)
    builder.add_node("understand", understand)
    for step, name in STEP_NODES.items():
        if step != "confirmed":
            builder.add_node(name, step_node(step))
    for name, fn in (("clarify", clarify), ("ask_change", ask_change), ("book", book), ("done", done)):
        builder.add_node(name, fn)

    builder.add_edge(START, "extract")
    builder.add_edge("extract", "understand")
    reply_nodes = [n for n in builder.nodes if n not in ("extract", "understand")]
    builder.add_conditional_edges("understand", lambda s: s["route"], {n: n for n in reply_nodes})
    for name in reply_nodes:
        builder.add_edge(name, END)
    return builder


# --------------------------------------------------------------------------- facade


@dataclass(frozen=True)
class TurnResult:
    reply: str
    current_step: str
    booking: dict[str, Any]
    is_complete: bool
    reservation_id: Optional[int]


def booking_summary(state: dict) -> dict[str, Any]:
    summary = {k: state.get(k) for k in SLOTS}
    summary["total_price"] = state.get("total_price")
    summary["nights"] = _nights(state) if state.get("check_in_date") and state.get("check_out_date") else None
    return summary


class BookingAgent:
    """One graph shared by all sessions; each session is a LangGraph thread."""

    def __init__(
        self,
        extractor: Extractor,
        session_factory: sessionmaker[Session],
        today_fn: Callable[[], date],
        checkpointer: BaseCheckpointSaver | None = None,
    ):
        self.checkpointer = checkpointer or InMemorySaver()
        self.graph = build_graph(extractor, session_factory, today_fn).compile(checkpointer=self.checkpointer)

    @staticmethod
    def _config(session_id: str) -> dict:
        return {"configurable": {"thread_id": session_id}}

    def chat(self, session_id: str, message: str) -> TurnResult:
        state = self.graph.invoke({"messages": [HumanMessage(message)]}, self._config(session_id))
        return TurnResult(
            reply=state["reply"],
            current_step=state["current_step"],
            booking=booking_summary(state),
            is_complete=state.get("reservation_id") is not None,
            reservation_id=state.get("reservation_id"),
        )

    def state(self, session_id: str) -> dict:
        return dict(self.graph.get_state(self._config(session_id)).values)

    def reset(self, session_id: str) -> None:
        self.checkpointer.delete_thread(session_id)
