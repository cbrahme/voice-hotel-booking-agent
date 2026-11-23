"""The one LLM call per turn: turn the guest's message into a TurnExtraction."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Protocol

from langchain_core.messages import HumanMessage, SystemMessage

from .prompts import EXTRACTION_PROMPT
from .state import TurnExtraction


@dataclass(frozen=True)
class ExtractionContext:
    today: date
    step: str
    known: dict[str, Any] = field(default_factory=dict)
    last_reply: str = ""


class Extractor(Protocol):
    def extract(self, message: str, context: ExtractionContext) -> TurnExtraction: ...


CALENDAR_DAYS = 21


def calendar_lines(today: date, days: int = CALENDAR_DAYS) -> str:
    """'Mon 2026-11-23 (today)', 'Tue 2026-11-24', ... — LLMs are unreliable at weekday
    arithmetic, so we give them a lookup table instead."""
    lines = []
    for i in range(days):
        d = today + timedelta(days=i)
        lines.append(f"{d:%a} {d.isoformat()}" + (" (today)" if i == 0 else ""))
    return "\n".join(lines)


def build_messages(message: str, context: ExtractionContext) -> list:
    known = {k: v for k, v in context.known.items() if v is not None}
    system = EXTRACTION_PROMPT.format(
        today=context.today.isoformat(),
        weekday=f"{context.today:%A}",
        calendar=calendar_lines(context.today),
        step=context.step,
        known=json.dumps(known) if known else "nothing yet",
        last_reply=context.last_reply.replace('"', "'"),
    )
    return [SystemMessage(system), HumanMessage(message)]


class OpenAIExtractor:
    """Calls OpenAI with structured output. Create once and reuse (keeps connections warm)."""

    def __init__(self, model: str, api_key: str | None = None, timeout: float = 10.0):
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=0,
            max_tokens=200,
            timeout=timeout,
            max_retries=1,
        )
        self._llm = llm.with_structured_output(TurnExtraction)

    def extract(self, message: str, context: ExtractionContext) -> TurnExtraction:
        return self._llm.invoke(build_messages(message, context))
