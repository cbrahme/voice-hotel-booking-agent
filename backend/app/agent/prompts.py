"""The extraction prompt and every reply the agent speaks.

Replies are templates filled from validated state, so the agent can't invent prices,
dates or availability. Keep them short: they are read aloud.
"""

from __future__ import annotations

EXTRACTION_PROMPT = """\
You extract hotel booking details from ONE message a guest says to a voice booking assistant.
The message came from speech recognition, so it may contain recognition errors.

Today is {today} ({weekday}).
Current step: {step}
Already known: {known}
The assistant's last message: "{last_reply}"

Rules:
- Only fill a field if the guest stated it in THIS message. Never guess or copy known values.
- Dates: resolve to YYYY-MM-DD relative to today. A weekday on its own or "next <weekday>" means
  the first such day after today. A day of month without a month means the next one to come.
- If the guest gives a length of stay ("3 nights", "a week"), set nights.
- num_guests is the total number of people ("me and my wife" = 2, "just me" = 1).
- room_type must be Standard, Deluxe or Suite. Use the assistant's last message to resolve
  references like "the cheaper one" or "that one".
- phone_number: digits only (convert spoken numbers to digits); keep a leading + if a country code is said.
- intent: "provide" if any detail was given (even with a yes/no); otherwise "confirm" for yes/agreement,
  "deny" for no or a wish to change something, "off_topic" for greetings or unrelated talk,
  "unclear" if you can't tell what they mean.
"""

GREETING = "Hi! I can book a room for you. When would you like to stay?"

ERROR_RETRY = "Sorry, I had trouble there. Could you say that again?"
DIDNT_CATCH = "Sorry, I didn't quite catch that."
OFF_TOPIC = "I can help you book a room."
TYPE_HINT = "You can also type your answer in the box below."

ASK_DATES = "When would you like to stay? Tell me your check-in date and how many nights."
ASK_CHECK_OUT = "Checking in {check_in}. How many nights will you stay?"
ASK_CHECK_IN = "Checking out {check_out}. What date would you like to check in?"
DATES_ACK = "Got it: {check_in} to {check_out}, {nights}."
BAD_DATE = "I didn't catch the date."

ASK_GUESTS = "How many guests will be staying?"
TOO_MANY_GUESTS = "Our largest room, the {room}, fits up to {max_guests} guests."
BAD_GUEST_COUNT = "I need at least one guest."

ROOM_OPTIONS = "For {guests} on those dates, we have {options}. Which would you like?"
ROOM_SINGLE_OPTION = "For {guests} on those dates, we have {options}. Would you like it?"
NO_ROOMS = "Sorry, we have no rooms for {guests} on those dates. Would you like to try different dates?"
ROOM_OPTION = "a {room} at {price} a night"
ROOM_TOO_SMALL = "The {room} fits up to {max_guests} guests."
ROOM_SOLD_OUT = "Sorry, the {room} isn't available for those dates."
ROOM_JUST_TAKEN = "I'm sorry, the {room} was just booked by someone else."

ASK_NAME_AND_PHONE = "The {room} for {nights} comes to {price}. May I have your full name and phone number?"
ASK_NAME = "And what name should the booking be under?"
ASK_PHONE = "Thanks, {first_name}. What's the best phone number to reach you?"
BAD_PHONE = "That phone number doesn't look right."
BAD_NAME = "I didn't catch the name."

READ_BACK = (
    "Let me confirm: a {room} for {guests}, checking in {check_in} and checking out {check_out}, "
    "{nights}, for a total of {price}. The booking is under {name}, phone {phone}. Shall I confirm?"
)
ASK_YES_NO = "Shall I go ahead and confirm the booking? Please say yes or no."
ASK_WHAT_TO_CHANGE = "No problem. What would you like to change?"

BOOKED = "You're all set, {first_name}! Your booking reference is {reservation_id}. We look forward to seeing you."
ALREADY_BOOKED = (
    "Your booking is already confirmed, reference {reservation_id}. "
    "To make another booking, please start a new conversation."
)


def plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def money(amount: float) -> str:
    return f"${amount:,.0f}" if amount == int(amount) else f"${amount:,.2f}"


def join_options(parts: list[str]) -> str:
    if len(parts) <= 2:
        return " and ".join(parts)
    return ", ".join(parts[:-1]) + ", and " + parts[-1]
