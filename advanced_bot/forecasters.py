"""
Forecaster stage: N forecasters (spread across model families) run in parallel.
Each one:
  1. writes a draft (structured (a)-(h) template, or a simple prompt),
  2. passes code checks: format (with retries), arithmetic, unverified facts, consistency,
  3. goes through `critique.rounds` rounds of: critic review -> structured response.
"""
from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field

from advanced_bot import llm
from advanced_bot.checks import (
    FormatError,
    arithmetic_issues,
    consistency_issue,
    extract_json,
    parse_and_validate,
    unverified_claims,
    validate_percentile_values,
)
from advanced_bot.config import Config
from advanced_bot.llm import RunContext
from advanced_bot.mixture import mixture_percentiles, mixture_summary, parse_mixture, time_to_event_probability
from advanced_bot.question_view import PERCENTILES, QuestionView

logger = logging.getLogger(__name__)


@dataclass
class ForecasterSlot:
    member: str
    model: str
    sample: int


@dataclass
class MemberRun:
    member: str
    model: str
    sample: int
    prediction: dict
    reasoning: str
    stages: list[dict] = field(default_factory=list)   # [{stage, text, prediction}]
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def forecaster_slots(cfg: Config) -> list[ForecasterSlot]:
    """total_forecasters spread round-robin over members: 5 over 3 members -> 2,2,1."""
    members = cfg.get_path("ensemble.members")
    total = int(cfg.get_path("ensemble.total_forecasters"))
    slots, counts = [], {}
    for i in range(total):
        m = members[i % len(members)]
        counts[m["name"]] = counts.get(m["name"], 0) + 1
        slots.append(ForecasterSlot(member=m["name"], model=m["model"], sample=counts[m["name"]]))
    return slots


# ============================================================================ prompts


def _answer_format(qv: QuestionView, cfg: Config, structured: bool) -> str:
    mix_numeric = cfg.get_path("mixture.numeric") and qv.is_continuous
    tte = cfg.get_path("mixture.binary_time_to_event") and qv.kind == "binary"
    anchor = ""
    if structured and qv.kind == "binary":
        anchor = ('  "base_rate": <your outside-view probability, 0-100>,\n'
                  '  "evidence_direction": "up" | "down" | "neutral",   // how the specific evidence moves you from the base rate\n')
    elif structured and qv.is_continuous:
        sq = "YYYY-MM-DD" if qv.kind == "date" else "<number>"
        anchor = (f'  "status_quo_value": {sq},   // outcome if nothing changes\n'
                  '  "evidence_direction": "up" | "down" | "neutral",\n')

    if qv.kind == "binary" and tte:
        return (
            "End with a JSON block:\n```json\n{\n" + anchor +
            '  "time_to_event": {\n'
            '    "p_never": <probability (0-1) the event never happens in the foreseeable future>,\n'
            '    "scenarios": [ {"name": "...", "weight": <w>, "distribution": "lognormal"|"normal"|"student_t",\n'
            '                    "params": {...}} ]   // 1-4 scenarios for WHEN it happens, in DAYS FROM TODAY\n'
            "  }\n}\n```\n"
            "Params: normal {mean, sd}; lognormal {median, sigma}; student_t {loc, scale, df}. "
            "Code computes P(event before the resolution date) from this."
        )
    if qv.kind == "binary":
        return "End with a JSON block:\n```json\n{\n" + anchor + '  "probability": <0-100>\n}\n```'
    if qv.kind == "multiple_choice":
        opts = ", ".join(f'"{o}": <pct>' for o in qv.options)
        return ("End with a JSON block (percentages for EVERY option, each at least 1, summing to 100):\n"
                "```json\n{\n" + anchor + f'  "probabilities": {{{opts}}}\n}}\n```')
    unit = "days from today" if qv.kind == "date" else "question units"
    if mix_numeric:
        return (
            "End with a JSON block describing 1-4 scenarios; code turns it into percentiles:\n```json\n{\n" + anchor +
            '  "mixture": [ {"name": "...", "weight": <w>, "distribution": "normal"|"lognormal"|"student_t",\n'
            '                "params": {...}} ]\n}\n```\n'
            f"Params are in {unit}: normal {{mean, sd}}; lognormal {{median, sigma}} (sigma = sd of log); "
            "student_t {loc, scale, df}. Use wide tails: unknown unknowns are common."
        )
    keys = ", ".join(f'"{p:g}": <value>' for p in PERCENTILES)
    fmt = "YYYY-MM-DD dates" if qv.kind == "date" else "plain numbers (no units, no scientific notation)"
    return (
        "End with a JSON block with all 15 percentiles as " + fmt + ", strictly increasing:\n"
        "```json\n{\n" + anchor + f'  "percentiles": {{{keys}}}\n}}\n```\n'
        "Respect the bounds: a CLOSED bound can never be crossed. Set wide 0.1/99.9 tails."
    )


def draft_prompt(qv: QuestionView, cfg: Config, research: str, today: str, notes: str) -> str:
    structured = cfg.get_path("prompt_style") == "structured"
    header = qv.prompt_block(today)
    if not structured:
        return (
            f"You are an expert forecaster.\n\n{header}\n\nResearch:\n{research}\n\n{notes}"
            f"Give a short rationale, then your forecast.\n{_answer_format(qv, cfg, False)}"
        )
    low_high = ("(f) A scenario that leads to NO.\n(g) A scenario that leads to YES."
                if qv.kind == "binary" else
                "(f) A plausible scenario producing a LOW outcome.\n(g) A plausible scenario producing a HIGH outcome."
                if qv.is_continuous else
                "(f) A scenario producing an unexpected option.\n(g) A scenario that strongly favors the current leader.")
    return f"""You are a professional superforecaster. Forecast the question below.

{header}

Research report (cite facts by their [n] numbers; do not introduce facts that are not in the research,
and if you rely on your own background knowledge, label it clearly as "(background knowledge)"):
{research}

{notes}Write your reasoning with these headed sections:
(a) Time until resolution.
(b) Outside view: the historical base rate for this kind of event (show the reference class and the count).
(c) Status quo: what happens if nothing changes.
(d) Trend continuation: what happens if current trends continue.
(e) Experts and markets: what forecasters, prediction markets and experts expect.
{low_high}
(h) Forecast: weigh the above, starting from the outside view and adjusting for specific evidence.
Good forecasters respect the status quo, avoid overconfidence, and leave probability for surprises.

{_answer_format(qv, cfg, True)}"""


CRITIC_PROMPT = """You are a demanding forecasting reviewer. Review the forecaster's work below.

{header}

Research report:
{research}

Forecaster's current reasoning and forecast:
{draft}

{notes}List the most important problems, most important first, as bullets. Check for:
- misreading the resolution criteria or the time window,
- ignoring or misjudging the base rate / status quo,
- overconfidence or underconfidence,
- claims not supported by the research (and arithmetic errors),
- important evidence in the research that was ignored or double-counted.
Do not give your own forecast number. Only state facts that are in the research; if you
mention something else, label it "(unverified)"."""


RESPONSE_PROMPT = """A reviewer critiqued your forecast:

{critique}

{notes}Respond with these headed sections:
## Accepted criticisms
## Defense (criticisms you reject, and why)
## New insights
## Updated forecast

Then the JSON block, with one extra field "update_direction": "up" | "down" | "none" (how your
central forecast moved relative to your previous one).
{fmt}"""


# ============================================================================ running one forecaster


async def run_forecaster(
    ctx: RunContext, cfg: Config, qv: QuestionView, slot: ForecasterSlot, research: str
) -> MemberRun:
    r = cfg.get_path
    research_body = research.split("## Sources")[0]
    corpus = qv.corpus()
    flags: list[str] = []
    stages: list[dict] = []

    base_notes = _arith_note(arithmetic_issues(research_body))
    messages = [{"role": "user", "content": draft_prompt(qv, cfg, research, ctx.today, base_notes)}]
    text, obj, pred = await _ask_and_validate(ctx, cfg, qv, slot, messages, "forecaster_draft")
    pred, text, obj = await _consistency_pass(ctx, cfg, qv, slot, messages, text, obj, pred, None, flags)
    stages.append({"stage": "draft", "text": text, "prediction": pred})

    if r("critique.enabled"):
        for round_no in range(1, int(r("critique.rounds", 0)) + 1):
            draft_notes = _check_notes(cfg, text, research_body, corpus, "your previous answer")
            critic_res = await llm.complete(
                ctx, "critic", r("models.critic"),
                [{"role": "user", "content": CRITIC_PROMPT.format(
                    header=qv.prompt_block(ctx.today), research=research, draft=text,
                    notes=draft_notes)}],
                temperature=0.3, timeout=r("llm.timeout_seconds", 240),
                meta={"question": qv, "round": round_no},
            )
            critique = critic_res.text
            notes = _check_notes(cfg, critique, research_body, corpus, "the reviewer's critique")
            if notes and r("checks.verification") == "verify":
                notes += await _verify_quickly(ctx, cfg, qv, critique, research_body, corpus)
            stages.append({"stage": f"critique_{round_no}", "text": critique, "prediction": None})
            previous = pred
            messages.append({"role": "user", "content": RESPONSE_PROMPT.format(
                critique=critique, notes=notes, fmt=_answer_format(qv, cfg, r("prompt_style") == "structured"))})
            text, obj, pred = await _ask_and_validate(ctx, cfg, qv, slot, messages, f"forecaster_response_{round_no}")
            pred, text, obj = await _consistency_pass(ctx, cfg, qv, slot, messages, text, obj, pred, previous, flags)
            stages.append({"stage": f"response_{round_no}", "text": text, "prediction": pred})

    unverified = unverified_claims(text, research_body, corpus) if r("checks.verification") != "off" else []
    if unverified:
        flags.append("Unverified claims in final answer (not used as evidence): " + "; ".join(unverified))
    reasoning = _render_reasoning(slot, stages, flags)
    return MemberRun(member=slot.member, model=slot.model, sample=slot.sample, prediction=pred,
                     reasoning=reasoning, stages=stages, flags=flags)


async def _ask_and_validate(
    ctx: RunContext, cfg: Config, qv: QuestionView, slot: ForecasterSlot, messages: list[dict], role: str
) -> tuple[str, dict, dict]:
    """Call the model; validate strictly; on failure send the error back (max_format_retries)."""
    r = cfg.get_path
    retries = int(r("checks.max_format_retries", 2))
    last_error = ""
    for attempt in range(retries + 1):
        res = await llm.complete(
            ctx, role, slot.model, messages, temperature=r("llm.forecaster_temperature", 0.7),
            timeout=r("llm.timeout_seconds", 240), max_retries=r("llm.max_retries", 2),
            meta={"question": qv, "slot": slot, "attempt": attempt, "cfg": cfg},
        )
        messages.append({"role": "assistant", "content": res.text})
        try:
            obj = extract_json(res.text)
            pred = to_prediction(ctx, cfg, qv, obj)
            text = res.text + (f"\n\n{obj['_mixture_note']}" if obj.get("_mixture_note") else "")
            return text, obj, pred
        except FormatError as e:
            last_error = str(e)
            logger.info(f"[{slot.member}#{slot.sample}] format error (attempt {attempt + 1}): {e}")
            messages.append({"role": "user", "content": f"Your answer failed validation: {e}\n"
                                                        "Reply with the corrected JSON block (you may keep reasoning brief)."})
    # last resort for continuous questions: repair ordering/bounds in code
    if qv.is_continuous:
        try:
            obj = extract_json(messages[-2]["content"])
            raw = obj.get("percentiles", {})
            vals = [float(str(raw[k]).replace(",", "")) if qv.kind != "date" else
                    _date_ts(raw[k]) for k in (f"{p:g}" for p in PERCENTILES)]
            pred = {"percentiles": validate_percentile_values(qv, vals, strict=False)}
            ctx.warnings.append(f"{slot.member}#{slot.sample}: percentiles repaired in code after retries ({last_error})")
            return messages[-2]["content"], obj, pred
        except Exception:
            pass
    raise FormatError(f"{slot.member}#{slot.sample}: output still invalid after {retries} retries: {last_error}")


def _date_ts(v) -> float:
    from datetime import datetime, timezone
    d = datetime.fromisoformat(str(v))
    return (d if d.tzinfo else d.replace(tzinfo=timezone.utc)).timestamp()


def to_prediction(ctx: RunContext, cfg: Config, qv: QuestionView, obj: dict) -> dict:
    """Model JSON -> normalized prediction (handles the mixture / time-to-event formats)."""
    out_cfg = cfg.get("output", {})
    if qv.kind == "binary" and "time_to_event" in obj and cfg.get_path("mixture.binary_time_to_event"):
        if not qv.resolution_date:
            raise FormatError("time_to_event needs a resolution date; give \"probability\" instead.")
        days = (qv.resolution_date - ctx.cutoff).total_seconds() / 86400
        p, scenarios, p_never = time_to_event_probability(obj["time_to_event"], days)
        lo, hi = out_cfg.get("min_probability", 0.01), out_cfg.get("max_probability", 0.99)
        obj["_mixture_note"] = mixture_summary(qv, scenarios) + f"\n- p_never: {p_never:.0%}\n- P(before resolution) = {p:.1%}"
        return {"p": min(max(p, lo), hi)}
    if qv.is_continuous and "mixture" in obj and cfg.get_path("mixture.numeric"):
        scenarios = parse_mixture(obj["mixture"])
        values = mixture_percentiles(qv, scenarios, ctx.cutoff.timestamp())
        obj["_mixture_note"] = mixture_summary(qv, scenarios)
        return {"percentiles": validate_percentile_values(qv, values, strict=False)}
    return parse_and_validate(qv, obj, out_cfg)


async def _consistency_pass(ctx, cfg, qv, slot, messages, text, obj, pred, previous, flags):
    """If direction and number disagree, ask once for clarification."""
    if not cfg.get_path("checks.consistency"):
        return pred, text, obj
    issue = consistency_issue(qv, obj, pred, previous)
    if not issue:
        return pred, text, obj
    flags.append(f"Consistency check: {issue}")
    messages.append({"role": "user", "content": f"Consistency check (automatic): {issue}\n"
                                                "Either correct the number or correct the stated direction, "
                                                "explain in 2-3 sentences, and give the full JSON block again."})
    try:
        text2, obj2, pred2 = await _ask_and_validate(ctx, cfg, qv, slot, messages, "forecaster_clarify")
    except FormatError:
        return pred, text, obj
    if consistency_issue(qv, obj2, pred2, previous):
        flags.append("Consistency issue persisted after clarification.")
    return pred2, text + "\n\n**Clarification after consistency check:**\n" + text2, obj2


def _arith_note(issues: list[str]) -> str:
    if not issues:
        return ""
    return ("ARITHMETIC CHECK (computed by code; trust these over the text):\n"
            + "\n".join(f"- {i}" for i in issues) + "\n\n")


def _check_notes(cfg: Config, text: str, research: str, corpus: str, who: str) -> str:
    notes = ""
    if cfg.get_path("checks.arithmetic"):
        notes += _arith_note(arithmetic_issues(text))
    if cfg.get_path("checks.verification") != "off":
        claims = unverified_claims(text, research, corpus)
        if claims:
            notes += (f"UNVERIFIED (appear in {who} but NOT in the research; do not build on them):\n"
                      + "\n".join(f"- {c}" for c in claims) + "\n\n")
    return notes


async def _verify_quickly(ctx, cfg, qv, text, research, corpus) -> str:
    """verification = verify: one quick, date-bounded search per claim (max 3) + a verifier verdict."""
    from advanced_bot.tools import TimeWindow, available_tools, run_tool

    claims = unverified_claims(text, research, corpus)[:3]
    tools = [t for t in (available_tools(cfg) if not ctx.fake else cfg.get_path("research.tools")) if t in ("asknews", "web")]
    if not claims or not tools:
        return ""
    tool_name = {"asknews": "search_news", "web": "web_search"}[tools[0]]
    window = TimeWindow.for_question(cfg, ctx.cutoff, qv.resolution_date)
    lines = []
    for claim in claims:
        try:
            items = await run_tool(ctx, cfg, window, tool_name, {"query": f"{qv.q.question_text[:80]} {claim[:80]}"})
        except Exception as e:
            ctx.warnings.append(f"verification search failed: {e}")
            continue
        evidence = "\n".join(f"- {it.title} ({it.url}): {it.snippet[:400]}" for it in items[:4])
        res = await llm.complete(ctx, "verifier", cfg.get_path("models.verifier"), [{"role": "user", "content":
            f"Claim: {claim}\nSearch results:\n{evidence or '(none)'}\n"
            'Is the claim supported by these results? Answer JSON {"supported": true|false, "url": "<supporting url or empty>"}'}],
            temperature=0.0, meta={"question": qv, "claim": claim})
        try:
            verdict = extract_json(res.text)
        except FormatError:
            continue
        if verdict.get("supported"):
            lines.append(f"- VERIFIED by quick search: {claim} — {verdict.get('url', '')}")
    return ("QUICK VERIFICATION RESULTS:\n" + "\n".join(lines) + "\n\n") if lines else ""


def _render_reasoning(slot: ForecasterSlot, stages: list[dict], flags: list[str]) -> str:
    parts = [f"### Forecaster {slot.member} #{slot.sample} ({slot.model})"]
    for st in stages:
        title = st["stage"].replace("_", " ").capitalize()
        parts.append(f"#### {title}\n{st['text'].strip()}")
    if flags:
        parts.append("#### Automatic checks\n" + "\n".join(f"- {f}" for f in flags))
    return "\n\n".join(parts)


async def run_all_forecasters(ctx: RunContext, cfg: Config, qv: QuestionView, research: str) -> list[MemberRun]:
    slots = forecaster_slots(cfg)
    results = await asyncio.gather(*(run_forecaster(ctx, cfg, qv, s, research) for s in slots),
                                   return_exceptions=True)
    runs = []
    for slot, res in zip(slots, results):
        if isinstance(res, BaseException):
            ctx.warnings.append(f"Forecaster {slot.member}#{slot.sample} failed: {type(res).__name__}: {res}")
            logger.warning(f"Forecaster {slot.member}#{slot.sample} failed: {res}")
        else:
            runs.append(res)
    needed = math.ceil(len(slots) / 2)
    if len(runs) < needed:
        raise RuntimeError(f"Only {len(runs)}/{len(slots)} forecasters succeeded (need {needed}). {ctx.warnings[-3:]}")
    return runs
