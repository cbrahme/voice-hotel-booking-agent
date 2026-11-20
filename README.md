# Voice Hotel Booking Agent

Book a hotel room by talking instead of filling in a form.

You open a web page, click the mic and say something like *"I need a room next Friday for two nights for two people."* The assistant asks for anything it still needs: dates, number of guests, room type, your name and phone number. It then reads the booking back to you and saves it once you say yes. You can change your mind partway through ("actually, make it three guests") and it adjusts.

## How it works

- **Your browser** turns your speech into text and speaks the replies, using the browser's built-in speech features, so there's no paid speech service.
- **An AI model** (OpenAI `gpt-4o-mini`) picks out the details you mentioned, such as dates and the number of guests.
- **Python code** decides what to ask next, checks which rooms are free and works out the price. The AI never makes up prices or availability.
- **A local SQLite database** stores the rooms and bookings.

The hotel has three room types: Standard ($120/night, up to 2 guests), Deluxe ($180/night, up to 3) and Suite ($280/night, up to 4).

## Status

🚧 **Work in progress.** Done so far: project setup and the database layer (rooms, availability, pricing and safe booking). Still to build: the conversation logic, the API and the voice web page.

## Getting started

Requires Python 3.10+.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your OpenAI API key
pytest                 # run the tests
```

## Tech stack

Python · LangGraph · FastAPI · SQLAlchemy + SQLite · OpenAI API · Web Speech API
