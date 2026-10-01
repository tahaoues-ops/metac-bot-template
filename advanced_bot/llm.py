"""
Thin async LLM layer over LiteLLM.

Every call is recorded in the RunContext trace (role, model, cost, seconds,
prompt, output) so we get per-question cost/time logs and can export
prompt/output pairs for later RL training.

Setting ADV_BOT_FAKE_LLM=1 (or RunContext.fake=True) routes every call to the
deterministic offline backend in advanced_bot/testing/fake.py — used by the
test suite and for trying the pipeline without API keys.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LlmResult:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    cost: float = 0.0
    model: str = ""
    seconds: float = 0.0
    raw_message: dict | None = None  # assistant message to append when continuing a tool loop


@dataclass
class CallRecord:
    role: str
    model: str
    cost: float
    seconds: float
    messages: list[dict]
    output: str
    tool_calls: list[dict]


@dataclass
class RunContext:
    """Per-question state: the knowledge cutoff, cost/time trace, and flags."""

    cutoff: datetime                      # "today" for the bot; in eval = question open time
    backtest: bool = False                # True when forecasting already-resolved questions
    fake: bool = field(default_factory=lambda: os.getenv("ADV_BOT_FAKE_LLM") == "1")
    calls: list[CallRecord] = field(default_factory=list)
    tool_log: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    started: float = field(default_factory=time.time)
    seed: int = 0                         # used by the fake backend only

    @property
    def total_cost(self) -> float:
        return sum(c.cost for c in self.calls)

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.started

    @property
    def today(self) -> str:
        return self.cutoff.astimezone(timezone.utc).strftime("%Y-%m-%d")

    def models_used(self) -> dict[str, list[str]]:
        out: dict[str, set[str]] = {}
        for c in self.calls:
            out.setdefault(c.role, set()).add(c.model)
        return {k: sorted(v) for k, v in out.items()}

    def cost_by_model(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for c in self.calls:
            out[c.model] = out.get(c.model, 0.0) + c.cost
        return out


async def complete(
    ctx: RunContext,
    role: str,
    model: str,
    messages: list[dict],
    *,
    temperature: float = 0.3,
    tools: list[dict] | None = None,
    timeout: float = 240,
    max_retries: int = 2,
    meta: dict | None = None,
) -> LlmResult:
    """One chat completion. `role` labels the call in logs (researcher, forecaster, critic...)."""
    start = time.time()
    if ctx.fake:
        from advanced_bot.testing.fake import fake_complete

        result = fake_complete(ctx, role, model, messages, tools=tools, meta=meta or {})
    else:
        result = await _litellm_complete(model, messages, temperature, tools, timeout, max_retries)
    result.seconds = time.time() - start
    result.model = model
    ctx.calls.append(
        CallRecord(
            role=role,
            model=model,
            cost=result.cost,
            seconds=result.seconds,
            messages=_strip_for_log(messages),
            output=result.text,
            tool_calls=[{"name": t.name, "arguments": t.arguments} for t in result.tool_calls],
        )
    )
    return result


async def _litellm_complete(
    model: str,
    messages: list[dict],
    temperature: float,
    tools: list[dict] | None,
    timeout: float,
    max_retries: int,
) -> LlmResult:
    import litellm

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "timeout": timeout,
        "drop_params": True,  # some reasoning models reject temperature
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = await litellm.acompletion(**kwargs)
            break
        except Exception as e:  # network / rate limit / provider errors
            last_error = e
            wait = 2 ** (attempt + 1)
            logger.warning(f"LLM call to {model} failed ({type(e).__name__}); retry in {wait}s")
            await asyncio.sleep(wait)
    else:
        raise RuntimeError(f"LLM call to {model} failed after retries: {last_error}")

    message = response.choices[0].message
    tool_calls = []
    for tc in getattr(message, "tool_calls", None) or []:
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {"_unparsed": tc.function.arguments}
        tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
    try:
        cost = float(litellm.completion_cost(completion_response=response) or 0.0)
    except Exception:
        cost = 0.0  # model not in LiteLLM's price table
    raw = message.model_dump() if hasattr(message, "model_dump") else dict(message)
    return LlmResult(text=message.content or "", tool_calls=tool_calls, cost=cost, raw_message=raw)


def _strip_for_log(messages: list[dict]) -> list[dict]:
    """Keep logs readable: drop provider-specific fields, keep role/content/tool info."""
    out = []
    for m in messages:
        entry = {"role": m.get("role"), "content": m.get("content")}
        if m.get("tool_calls"):
            entry["tool_calls"] = m["tool_calls"]
        if m.get("name"):
            entry["name"] = m["name"]
        out.append(entry)
    return out


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
