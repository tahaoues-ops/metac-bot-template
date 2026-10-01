"""
Deterministic offline backend (ADV_BOT_FAKE_LLM=1). No network, no keys, no cost.

It is NOT a forecaster: it exists to exercise every code path (tool loop,
citations, format retries, critique rounds, checks, aggregation, eval metrics).
To make eval/ensemble plumbing testable, fake forecasters peek at the known
resolution and add member-specific noise, so members differ in "skill":
claude < gpt < gemini noise. Never interpret fake scores as real performance.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from datetime import timedelta

from advanced_bot.llm import LlmResult, RunContext, ToolCall
from advanced_bot.question_view import PERCENTILES, QuestionView
from advanced_bot.tools import SourceItem, TimeWindow

NOISE = {"claude": 0.6, "gpt": 0.9, "gemini": 1.3, "grok": 1.0, "deepseek": 1.1}
TOOL_ORDER = ["search_news", "web_search", "wikipedia", "futuresearch"]


def _rng(*parts) -> random.Random:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return random.Random(int(h[:12], 16))


def fake_search(ctx: RunContext, key: str, query: str, w: TimeWindow) -> list[SourceItem]:
    rng = _rng(key, query, ctx.seed)
    items = []
    for i in range(2):
        pub = w.end - timedelta(days=rng.randint(1, max(2, int((w.end - w.start).days))))
        slug = hashlib.md5(f"{key}{query}{i}".encode()).hexdigest()[:8]
        items.append(SourceItem(
            title=f"{key.title()} result {i + 1} for '{query[:40]}'",
            url=f"https://example.org/{key}/{slug}",
            snippet=f"Synthetic {key} snippet {i + 1} about {query[:60]}. Officials reported 3 of 10 talks succeeded (30%).",
            published=pub, provider=key,
        ))
    return items


def fake_complete(ctx: RunContext, role: str, model: str, messages: list[dict], tools=None, meta=None) -> LlmResult:
    meta = meta or {}
    if role == "researcher":
        return _researcher(ctx, messages, tools, meta)
    if role == "summarizer":
        return LlmResult(text="The research describes the current situation and recent data [1].\n\n"
                              "Base rates suggest such events are uncommon, and gaps remain [2].")
    if role == "critic":
        return LlmResult(text="- The draft may underweight the base rate.\n- Check the timeline in the resolution criteria.")
    if role == "leak_judge":
        last = messages[-1]["content"]
        leaked = "RESOLVED AS" in last.upper().split("RESEARCH REPORT:", 1)[-1]
        return LlmResult(text=json.dumps({"leaked": leaked, "evidence": ["mentions the final result"] if leaked else []}))
    if role == "verifier":
        return LlmResult(text='{"supported": false, "url": ""}')
    if role.startswith("forecaster"):
        return _forecaster(ctx, role, model, messages, meta)
    return LlmResult(text="ok")


def _researcher(ctx: RunContext, messages, tools, meta) -> LlmResult:
    rnd = meta.get("round", 0)
    leak = meta["question"].key.split("-")[0].endswith("5")  # every 10th synthetic question leaks
    if meta.get("repair"):
        return LlmResult(text=_report(leak))
    offered = [t["function"]["name"] for t in tools or []]
    if offered and rnd < meta.get("min", 4):
        name = offered[rnd % len(offered)]
        q = meta["question"].q.question_text[:50]
        args = {"topic": q} if name == "wikipedia" else {"query": f"{q} angle {rnd}"}
        return LlmResult(text="", tool_calls=[ToolCall(id=f"call_{rnd}", name=name, arguments=args)])
    return LlmResult(text="<think>internal notes that must be stripped</think>\n" + _report(leak))


def _report(leak: bool = False) -> str:
    return (
        "## Current situation\n- Talks are ongoing according to officials [1].\n"
        + ("- Later coverage says the question was RESOLVED AS the final outcome [2].\n" if leak else "") + "\n"
        "## Latest data\n- 3 of 10 recent rounds of talks ended with an agreement (30%) [2].\n\n"
        "## Prediction markets\n- No liquid market was found [3].\n\n"
        "## Models and experts\n- Analysts are divided on the outcome [1][99].\n\n"
        "## Key mechanisms\n- Domestic politics constrain both sides [2].\n\n"
        "## Historical base rates\n- Similar negotiations succeeded 4 times in 20 years (20% per year) [3].\n\n"
        "## Information gaps\n- No data on private negotiations.\n\n"
        "## Sources\n[1] should be removed by code"
    )


def _skill_signal(qv: QuestionView, member: str, ctx: RunContext, sample: int) -> float:
    """Logit-scale signal: truth plus member-specific noise."""
    rng = _rng(qv.key, member, sample, ctx.seed)
    outcome = qv.outcome()
    noise = NOISE.get(member, 1.0)
    base = 0.0
    if outcome and "yes" in outcome:
        base = 1.2 if outcome["yes"] else -1.2
    return base + rng.gauss(0, noise)


def _forecaster(ctx: RunContext, role: str, model: str, messages, meta) -> LlmResult:
    qv: QuestionView = meta["question"]
    slot = meta["slot"]
    cfg = meta["cfg"]
    attempt = meta.get("attempt", 0)
    rng = _rng(qv.key, slot.member, slot.sample, role, ctx.seed)
    # Exercise the retry path: the first draft of member #2 is malformed once.
    if role == "forecaster_draft" and slot.sample == 1 and slot.member == "gpt" and attempt == 0:
        return LlmResult(text="(a) ... I forgot the JSON block.")
    sig = _skill_signal(qv, slot.member, ctx, slot.sample)
    step = int(role.rsplit("_", 1)[-1]) if role.startswith("forecaster_response") else 0
    reasoning = (
        "(a) Time until resolution: several months.\n(b) Outside view: 3 of 10 similar cases (30%) [2].\n"
        "(c) Status quo: no agreement.\n(d) Trend: slow progress [1].\n(e) Experts: divided [1].\n"
        "(f) Low scenario: talks collapse.\n(g) High scenario: breakthrough at a summit.\n(h) Forecast below.\n"
    )
    if step:
        reasoning = ("## Accepted criticisms\n- Base rate deserves more weight.\n## Defense\n- Timeline is fine.\n"
                     "## New insights\n- None.\n## Updated forecast\nSlightly adjusted.\n")

    if qv.kind == "binary":
        p = 1 / (1 + math.exp(-(sig * (1 + 0.1 * step))))
        p = min(max(p, 0.02), 0.98)
        base_rate = 30
        obj = {"base_rate": base_rate, "evidence_direction": "up" if p * 100 >= base_rate else "down",
               "probability": round(p * 100, 1)}
        if step:
            obj["update_direction"] = "none"
        if cfg.get_path("mixture.binary_time_to_event"):
            days = max((qv.resolution_date - ctx.cutoff).days, 1) if qv.resolution_date else 180
            obj = {"base_rate": base_rate, "evidence_direction": obj["evidence_direction"],
                   "time_to_event": {"p_never": round(max(0.05, 1 - p) * 0.5, 3), "scenarios": [
                       {"name": "main", "weight": 1, "distribution": "lognormal",
                        "params": {"median": days * (0.6 if p > 0.5 else 2.0), "sigma": 0.8}}]}}
    elif qv.kind == "multiple_choice":
        outcome = qv.outcome() or {}
        raw = {o: math.exp(rng.gauss(0, 1) * NOISE.get(slot.member, 1) + (1.5 if outcome.get("option") == o else 0))
               for o in qv.options}
        s = sum(raw.values())
        pct = {o: max(1.0, round(v / s * 100, 1)) for o, v in raw.items()}
        diff = 100 - sum(pct.values())
        top = max(pct, key=pct.get)
        pct[top] = round(pct[top] + diff, 1)
        obj = {"probabilities": pct}
    else:
        lo, hi = qv.lower, qv.upper
        outcome = qv.outcome()
        center = outcome["value"] if outcome else (lo + hi) / 2
        rngw = hi - lo
        center = center + rng.gauss(0, 0.15 * NOISE.get(slot.member, 1)) * rngw
        spread = 0.12 * rngw * NOISE.get(slot.member, 1)
        if cfg.get_path("mixture.numeric"):
            to_unit = (lambda v: (v - ctx.cutoff.timestamp()) / 86400) if qv.kind == "date" else (lambda v: v)
            obj = {"mixture": [
                {"name": "base", "weight": 0.7, "distribution": "normal",
                 "params": {"mean": to_unit(center), "sd": spread / (86400 if qv.kind == "date" else 1)}},
                {"name": "tail", "weight": 0.3, "distribution": "student_t",
                 "params": {"loc": to_unit(center), "scale": 2 * spread / (86400 if qv.kind == "date" else 1), "df": 3}},
            ]}
        else:
            z = {0.1: -3.09, 1: -2.33, 5: -1.64, 10: -1.28, 20: -0.84, 30: -0.52, 40: -0.25, 50: 0, 60: 0.25,
                 70: 0.52, 80: 0.84, 90: 1.28, 95: 1.64, 99: 2.33, 99.9: 3.09}
            vals = {p: center + z[p] * spread for p in PERCENTILES}
            if not qv.open_lower:
                vals = {p: max(v, lo + (p / 1000) * rngw * 0.01) for p, v in vals.items()}
            if not qv.open_upper:
                vals = {p: min(v, hi - ((100 - p) / 1000) * rngw * 0.01) for p, v in vals.items()}
            if qv.kind == "date":
                from datetime import datetime, timezone
                fmt = {p: datetime.fromtimestamp(v, tz=timezone.utc).strftime("%Y-%m-%d") for p, v in vals.items()}
                # dates have day resolution: keep them strictly increasing
                obj = {"percentiles": fmt}
            else:
                obj = {"percentiles": {f"{p:g}": round(v, 6) for p, v in vals.items()}}
        obj["status_quo_value"] = None
        obj.pop("status_quo_value")
    return LlmResult(text=reasoning + "\n```json\n" + json.dumps(obj) + "\n```", cost=0.0)
