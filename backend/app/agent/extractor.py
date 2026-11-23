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


def weekday_example(today: date) -> str:
    """A worked example from the real calendar, e.g. on Monday 2026-11-23:
    '"next Friday" = 2026-11-27 (not 2026-12-04)'. Models follow concrete examples
    better than the abstract rule."""
    target = today + timedelta(days=4)
    return f'"next {target:%A}" = {target.isoformat()} (not {(target + timedelta(days=7)).isoformat()})'


def build_messages(message: str, context: ExtractionContext) -> list:
    known = {k: v for k, v in context.known.items() if v is not None}
    system = EXTRACTION_PROMPT.format(
        today=context.today.isoformat(),
        weekday=f"{context.today:%A}",
        calendar=calendar_lines(context.today),
        weekday_example=weekday_example(context.today),
        step=context.step,
        known=json.dumps(known) if known else "nothing yet",
        last_reply=context.last_reply.replace('"', "'"),
    )
    return [SystemMessage(system), HumanMessage(message)]


def is_reasoning_model(model: str) -> bool:
    """GPT-5.x and o-series models think before answering; they take reasoning_effort
    instead of temperature."""
    return model.startswith(("gpt-5", "o1", "o3", "o4"))


class OpenAIExtractor:
    """Calls OpenAI with structured output. Create once and reuse (keeps connections warm).

    ``method`` is how the schema is enforced. "function_calling" is the default: in
    benchmarks (scripts/bench_latency.py) it was ~30% faster than "json_schema" for
    gpt-4o-mini with equal or better accuracy.
    ``reasoning_effort`` applies to reasoning models only (e.g. "none", "minimal", "low").
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        method: str = "function_calling",
        reasoning_effort: str | None = None,
        timeout: float = 5.0,
    ):
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {"model": model, "api_key": api_key, "timeout": timeout, "max_retries": 1}
        if is_reasoning_model(model):
            kwargs["reasoning_effort"] = reasoning_effort or "low"
            kwargs["max_tokens"] = 2000  # includes reasoning tokens
        else:
            kwargs["temperature"] = 0
            kwargs["max_tokens"] = 200
        self.model, self.method = model, method
        self.reasoning_effort = kwargs.get("reasoning_effort")
        self._llm = ChatOpenAI(**kwargs).with_structured_output(TurnExtraction, method=method)

    def extract(self, message: str, context: ExtractionContext) -> TurnExtraction:
        return self._llm.invoke(build_messages(message, context))
