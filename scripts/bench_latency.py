"""Latency benchmark (TC-10).

Two modes:

  extractor  Call the LLM extractor directly on realistic utterances and report speed
             AND accuracy. Use it to compare models / settings:
               python scripts/bench_latency.py extractor --model gpt-4o-mini --rounds 3
               python scripts/bench_latency.py extractor --model gpt-5.4-nano --reasoning none

  server     Run scripted booking conversations against a running server and report
             the backend time per turn (from the Server-Timing header):
               python scripts/bench_latency.py server --url http://127.0.0.1:8000 --conversations 3

Uses real API calls (a small number of tokens).
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
import uuid
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.agent.extractor import ExtractionContext, OpenAIExtractor  # noqa: E402
from backend.app.config import get_settings  # noqa: E402

TODAY = date(2026, 11, 23)  # a Monday
ROOMS = ("For 2 guests on those dates, we have a Standard at $120 a night, a Deluxe at $180 a night, "
         "and a Suite at $280 a night. Which would you like?")

# (utterance, step, last agent reply, known values, expected fields)
CASES = [
    ("I need a room next Friday for three nights", "dates", "When would you like to stay?", {},
     {"check_in": "2026-11-27", "stay": ("2026-11-30", 3)}),
    ("tomorrow until Sunday", "dates", "When would you like to stay?", {},
     {"check_in": "2026-11-24", "check_out": "2026-11-29"}),
    ("december 12th for a week", "dates", "When would you like to stay?", {},
     {"check_in": "2026-12-12", "stay": ("2026-12-19", 7)}),
    ("it'll be me and my wife", "guests", "How many guests will be staying?", {"check_in_date": "2026-11-27"},
     {"num_guests": 2}),
    ("the cheaper one", "room", ROOMS, {"num_guests": 2}, {"room_type": "Standard"}),
    ("Ada Lovelace, five five five, one two three, four five six seven", "contact",
     "May I have your full name and phone number?", {"room_type": "Deluxe"},
     {"guest_name": "Ada Lovelace", "phone_digits": "5551234567"}),
    ("yes please go ahead", "awaiting_confirmation", "... Shall I confirm?", {"room_type": "Deluxe"},
     {"intent": "confirm"}),
    ("no wait, make it three guests", "awaiting_confirmation", "... Shall I confirm?", {"num_guests": 2},
     {"num_guests": 3}),
    ("uh the the mmm blue", "guests", "How many guests will be staying?", {},
     {"intent_in": ("unclear", "off_topic"), "num_guests": None}),
]


def check(ext, expected) -> list[str]:
    """Return a list of mismatches (empty = correct)."""
    problems = []
    for key, want in expected.items():
        if key == "stay":
            out, nights = want
            if not (ext.check_out == out or ext.nights == nights):
                problems.append(f"stay: check_out={ext.check_out} nights={ext.nights}, want {out} / {nights}")
        elif key == "phone_digits":
            got = "".join(c for c in (ext.phone_number or "") if c.isdigit())
            if got != want:
                problems.append(f"phone {got!r} != {want!r}")
        elif key == "intent_in":
            if ext.intent not in want:
                problems.append(f"intent {ext.intent!r} not in {want}")
        elif getattr(ext, key) != want:
            problems.append(f"{key} {getattr(ext, key)!r} != {want!r}")
    return problems


def summarize(label: str, times_ms: list[float]) -> str:
    times = sorted(times_ms)
    p95 = times[min(len(times) - 1, round(0.95 * (len(times) - 1)))]
    return (f"{label}: n={len(times)}  mean={statistics.mean(times):.0f}ms  "
            f"median={statistics.median(times):.0f}ms  p95={p95:.0f}ms  max={times[-1]:.0f}ms")


def bench_extractor(args) -> None:
    settings = get_settings()
    extractor = OpenAIExtractor(args.model or settings.openai_model, settings.openai_api_key,
                                method=args.method, reasoning_effort=args.reasoning)
    print(f"model={extractor.model} method={extractor.method} reasoning={extractor.reasoning_effort}")
    extractor.extract("hello", ExtractionContext(today=TODAY, step="dates"))  # warm-up (connection, schema)

    times, failures = [], []
    for round_no in range(args.rounds):
        for text, step, last_reply, known, expected in CASES:
            ctx = ExtractionContext(today=TODAY, step=step, known=known, last_reply=last_reply)
            started = time.perf_counter()
            try:
                ext = extractor.extract(text, ctx)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"[round {round_no + 1}] {text!r}: ERROR {exc}")
                continue
            times.append((time.perf_counter() - started) * 1000)
            problems = check(ext, expected)
            if problems:
                failures.append(f"[round {round_no + 1}] {text!r}: " + "; ".join(problems))

    total = args.rounds * len(CASES)
    print(summarize("llm call", times))
    print(f"accuracy: {total - len(failures)}/{total}")
    for f in failures:
        print("  ✗", f)


CONVERSATION = [
    "I need a room next Friday for two nights",
    "it will be me and my wife",
    "the deluxe one please",
    "Ada Lovelace, 555 123 4567",
    "no",  # don't actually book: decline at the read-back
]


def bench_server(args) -> None:
    import httpx

    totals, llms = [], []
    with httpx.Client(base_url=args.url, timeout=30) as client:
        client.get("/api/health").raise_for_status()
        for _ in range(args.conversations):
            session = f"bench-{uuid.uuid4()}"
            for text in CONVERSATION:
                started = time.perf_counter()
                r = client.post("/api/chat", json={"session_id": session, "user_message": text})
                wall = (time.perf_counter() - started) * 1000
                r.raise_for_status()
                timing = dict(
                    part.split(";dur=") for part in r.headers.get("server-timing", "").replace(" ", "").split(",") if ";dur=" in part
                )
                totals.append(wall)
                if "llm" in timing:
                    llms.append(float(timing["llm"]))
            client.post("/api/reset", json={"session_id": session})
    print(summarize("request round trip", totals))
    if llms:
        print(summarize("  of which llm call", llms))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("extractor")
    p.add_argument("--model")
    p.add_argument("--method", default="function_calling", choices=["json_schema", "function_calling"])
    p.add_argument("--reasoning", help="reasoning_effort for reasoning models (none/minimal/low)")
    p.add_argument("--rounds", type=int, default=2)
    s = sub.add_parser("server")
    s.add_argument("--url", default="http://127.0.0.1:8000")
    s.add_argument("--conversations", type=int, default=3)
    args = parser.parse_args()
    bench_extractor(args) if args.mode == "extractor" else bench_server(args)


if __name__ == "__main__":
    main()
