"""Conversation state and the shape of what the LLM extracts each turn."""

from __future__ import annotations

from typing import Annotated, Literal, Optional, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

Step = Literal["dates", "guests", "room", "contact", "awaiting_confirmation", "confirmed"]
Intent = Literal["provide", "confirm", "deny", "unclear", "off_topic"]
RoomType = Literal["Standard", "Deluxe", "Suite"]


class BookingState(TypedDict, total=False):
    messages: Annotated[list, add_messages]

    # Booking fields (dates are ISO 'YYYY-MM-DD' strings)
    check_in_date: Optional[str]
    check_out_date: Optional[str]
    num_guests: Optional[int]
    room_type: Optional[str]
    guest_name: Optional[str]
    phone_number: Optional[str]
    total_price: Optional[float]  # always computed from the DB, never from the LLM
    reservation_id: Optional[int]

    current_step: Step
    clarify_count: int  # consecutive turns with no progress

    # Per-turn scratch values (overwritten every turn)
    extraction: Optional[dict]
    route: str
    notices: list[str]  # short sentences to say before the next question
    last_changed_slots: list[str]
    reply: str


class TurnExtraction(BaseModel):
    """What the user said in this one message. Fields they didn't mention stay null."""

    check_in: Optional[str] = Field(None, description="Check-in date as YYYY-MM-DD.")
    check_out: Optional[str] = Field(None, description="Check-out date as YYYY-MM-DD.")
    nights: Optional[int] = Field(None, description="Length of stay in nights, if stated.")
    num_guests: Optional[int] = Field(None, description="Total number of people staying.")
    room_type: Optional[RoomType] = Field(None, description="Room type the user chose.")
    guest_name: Optional[str] = Field(None, description="Full name for the booking.")
    phone_number: Optional[str] = Field(None, description="Phone number as digits, with leading + if given.")
    intent: Intent = Field(
        description=(
            "provide: gave booking details; confirm: said yes / agreed; deny: said no / wants a change; "
            "unclear: garbled or can't tell; off_topic: greeting or unrelated."
        )
    )
