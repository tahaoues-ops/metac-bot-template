"""
Evaluation data:

  eval_data/questions.jsonl                         resolved questions (collected once)
  eval_data/research/<set_name>/<key>.json          FROZEN research per question, cut off at the
                                                    question's open time, + leak-check verdict
  eval_runs/<config_name>/<key>.json                one config's forecast on one question
                                                    (aggregate + every member's prediction + prompts)

Research is frozen so every experiment reuses exactly the same evidence: cheaper,
and comparisons between configs are fair (only the forecasting part differs).
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path

from advanced_bot import llm
from advanced_bot.checks import FormatError, extract_json
from advanced_bot.config import Config
from advanced_bot.llm import RunContext
from advanced_bot.aggregation import aggregate
from advanced_bot.forecasters import forecaster_slots, run_forecaster
from advanced_bot.question_view import QuestionView, question_from_dict, question_to_dict
from advanced_bot.researcher import ResearchResult, run_research, summarize

logger = logging.getLogger(__name__)

SUPPORTED_TYPES = ["binary", "multiple_choice", "numeric", "discrete", "date"]


# ============================================================================ paths


def data_dir(cfg: Config) -> Path:
    return Path(cfg.get_path("eval.data_dir", "eval_data"))


def questions_path(cfg: Config) -> Path:
    return data_dir(cfg) / "questions.jsonl"


def research_dir(cfg: Config) -> Path:
    return data_dir(cfg) / "research" / cfg.get_path("research.set_name", "default")


def run_dir(cfg: Config) -> Path:
    # keyed by what affects the LLM calls, so configs that differ only in aggregation or
    # number of forecasters share (and extend) the same stored forecasts
    return Path(cfg.get_path("eval.runs_dir", "eval_runs")) / cfg.forecaster_fingerprint()


def cutoff_for(qv: QuestionView) -> datetime:
    """What the bot is allowed to know: the moment the question opened."""
    d = qv.q.open_time or qv.q.published_time
    if d is None:
        raise ValueError(f"Question {qv.key} has no open/published time")
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


# ============================================================================ 1. collect


def default_tournaments() -> list:
    from forecasting_tools import MetaculusClient

    c = MetaculusClient
    return [c.AIB_FALL_2025_ID, c.AIB_SPRING_2026_ID, "minibench"]


async def collect_questions(cfg: Config, n: int, tournaments: list | None, seed: int = 0,
                            per_tournament_cap: int = 2000) -> list:
    """Resolved (not annulled) questions from past bot tournaments, sampled across types."""
    from forecasting_tools import MetaculusClient
    from forecasting_tools.helpers.metaculus_client import ApiFilter

    client = MetaculusClient()
    tournaments = tournaments or default_tournaments()
    found: dict[str, object] = {}
    for t in tournaments:
        api_filter = ApiFilter(allowed_statuses=["resolved"], allowed_types=SUPPORTED_TYPES,
                               allowed_tournaments=[t], group_question_mode="unpack_subquestions")
        try:
            qs = await client.get_questions_matching_filter(api_filter, num_questions=per_tournament_cap,
                                                            error_if_question_target_missed=False)
        except Exception as e:
            logger.warning(f"Could not fetch tournament {t}: {e}")
            continue
        usable = 0
        for q in qs:
            try:
                qv = QuestionView(q)
                if qv.outcome() is None or (q.open_time is None and q.published_time is None):
                    continue
            except Exception:
                continue  # unsupported type (e.g. conditional) or unparsable resolution
            found[qv.key] = q
            usable += 1
        print(f"  tournament {t}: {len(qs)} resolved, {usable} usable")

    by_kind: dict[str, list] = {}
    for q in found.values():
        by_kind.setdefault(QuestionView(q).kind, []).append(q)
    rng = random.Random(seed)
    for lst in by_kind.values():
        rng.shuffle(lst)
    # proportional sample, then fill
    total = sum(len(v) for v in by_kind.values())
    picked = []
    for kind, lst in by_kind.items():
        picked += lst[: max(1, round(n * len(lst) / total))] if total else []
    picked = picked[:n]
    save_questions(cfg, picked)
    return picked


def save_questions(cfg: Config, questions: list) -> None:
    path = questions_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for q in questions:
            qv = QuestionView(q)
            f.write(json.dumps({"key": qv.key, "kind": qv.kind, "question": question_to_dict(q)},
                               ensure_ascii=False) + "\n")


def load_questions(cfg: Config) -> list[QuestionView]:
    path = questions_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python eval.py collect")
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(QuestionView(question_from_dict(json.loads(line)["question"])))
    return out


# ============================================================================ 2. frozen research + leak check

LEAK_PROMPT = """You are auditing a research report used to back-test a forecasting bot.
The report must only contain information available BEFORE {cutoff}.

Question: {question}
The question actually resolved as: {outcome}

Does the research report reveal or strongly hint at the actual outcome through information
from AFTER {cutoff} (e.g. reports of the final result, events after the cutoff, "it was announced
that...", final figures)? Normal pre-cutoff information that merely makes the outcome likely is NOT a leak.

Research report:
{report}

Answer with JSON only: {{"leaked": true|false, "evidence": ["exact quote(s) that leak, if any"]}}"""


async def leak_check(ctx: RunContext, cfg: Config, qv: QuestionView, research: ResearchResult) -> dict:
    outcome = qv.outcome()
    if qv.kind == "binary":
        outcome_text = "YES" if outcome["yes"] else "NO"
    elif qv.kind == "multiple_choice":
        outcome_text = outcome["option"]
    else:
        outcome_text = qv.fmt_value(outcome["value"])
    res = await llm.complete(ctx, "leak_judge", cfg.get_path("models.leak_judge"), [{"role": "user", "content":
        LEAK_PROMPT.format(cutoff=ctx.today, question=qv.q.question_text, outcome=outcome_text,
                           report=research.report)}], temperature=0.0, meta={"question": qv})
    try:
        verdict = extract_json(res.text)
    except FormatError:
        verdict = {"leaked": True, "evidence": ["leak judge returned unparsable output; excluded to be safe"]}
    late = [s["url"] for s in research.sources
            if s.get("published") and datetime.fromisoformat(s["published"]) > ctx.cutoff]
    if late:
        verdict["leaked"] = True
        verdict.setdefault("evidence", []).append(f"sources dated after cutoff: {late[:5]}")
    verdict["leaked"] = bool(verdict.get("leaked"))
    return verdict


async def build_research(cfg: Config, limit: int | None = None, concurrency: int = 2, seed: int = 0) -> dict:
    """Research every question once (cut off at open time) and save it with its leak verdict."""
    out_dir = research_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    questions = load_questions(cfg)[: limit or None]
    sem = asyncio.Semaphore(concurrency)
    stats = {"done": 0, "skipped": 0, "leaked": 0, "failed": 0, "cost": 0.0}

    async def one(qv: QuestionView):
        path = out_dir / f"{qv.key}.json"
        if path.exists():
            stats["skipped"] += 1
            return
        async with sem:
            ctx = RunContext(cutoff=cutoff_for(qv), backtest=True, seed=seed)
            try:
                research = await run_research(ctx, cfg, qv)
                if cfg.get_path("summary.enabled"):
                    research.summary = await summarize(ctx, cfg, qv, research)
                leak = await leak_check(ctx, cfg, qv, research)
            except Exception as e:
                stats["failed"] += 1
                logger.warning(f"research failed for {qv.key}: {e}")
                return
            record = {"key": qv.key, "cutoff": ctx.cutoff.isoformat(), "research": research.to_dict(),
                      "leak": leak, "cost": ctx.total_cost, "seconds": ctx.elapsed_seconds,
                      "research_config": {"tools": cfg.get_path("research.tools"),
                                          "model": cfg.get_path("models.researcher"),
                                          "iterations": [cfg.get_path("research.min_iterations"),
                                                         cfg.get_path("research.max_iterations")]}}
            path.write_text(json.dumps(record, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
            stats["done"] += 1
            stats["leaked"] += int(leak["leaked"])
            stats["cost"] += ctx.total_cost
            print(f"  [{stats['done'] + stats['skipped']}/{len(questions)}] {qv.key} "
                  f"{'LEAKED (excluded)' if leak['leaked'] else 'ok'}")

    await asyncio.gather(*(one(qv) for qv in questions))
    return stats


def load_research(cfg: Config, qv: QuestionView) -> dict | None:
    path = research_dir(cfg) / f"{qv.key}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def usable_questions(cfg: Config) -> list[tuple[QuestionView, dict]]:
    """Questions with frozen research that passed the leak check."""
    out = []
    for qv in load_questions(cfg):
        rec = load_research(cfg, qv)
        if rec and not rec["leak"]["leaked"]:
            out.append((qv, rec))
    return out


# ============================================================================ 3. run a config


def _keep_call(c) -> bool:
    return c.role.startswith("forecaster") or c.role == "critic"


def _slot_id(member: str, sample: int) -> str:
    return f"{member}#{sample}"


def _read(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def needed_slots(cfg: Config) -> list:
    return forecaster_slots(cfg)


def pending(cfg: Config, limit: int | None = None) -> list[tuple[QuestionView, dict, list]]:
    """(question, research record, missing forecaster slots) for every question with work left."""
    out = []
    slots = needed_slots(cfg)
    for qv, rec in usable_questions(cfg)[: limit or None]:
        existing = _read(run_dir(cfg) / f"{qv.key}.json") or {}
        have = {_slot_id(m["member"], m["sample"]) for m in existing.get("members", [])}
        missing = [s for s in slots if _slot_id(s.member, s.sample) not in have]
        if missing:
            out.append((qv, rec, missing))
    return out


async def run_config(cfg: Config, limit: int | None = None, concurrency: int = 4, seed: int = 0) -> dict:
    """Run only the forecaster slots each question is missing, and merge them into the stored record."""
    out_dir = run_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "_configs.json"
    names = set(json.loads(meta_path.read_text())) if meta_path.exists() else set()
    meta_path.write_text(json.dumps(sorted(names | {cfg.name})))
    work = pending(cfg, limit)
    sem = asyncio.Semaphore(concurrency)
    stats = {"questions": len(work), "forecasters_run": 0, "failed": 0, "cost": 0.0}

    async def one(qv: QuestionView, rec: dict, missing: list):
        async with sem:
            ctx = RunContext(cutoff=datetime.fromisoformat(rec["cutoff"]), backtest=True, seed=seed)
            research = ResearchResult.from_dict(rec["research"])
            text = research.summary if cfg.get_path("summary.use_for_forecast") and research.summary else research.report
            results = await asyncio.gather(*(run_forecaster(ctx, cfg, qv, s, text) for s in missing),
                                           return_exceptions=True)
            path = out_dir / f"{qv.key}.json"
            record = _read(path) or {"key": qv.key, "kind": qv.kind, "fingerprint": cfg.forecaster_fingerprint(),
                                     "members": [], "calls": [], "cost": 0.0, "warnings": []}
            for slot, res in zip(missing, results):
                if isinstance(res, BaseException):
                    stats["failed"] += 1
                    record["warnings"].append(f"{_slot_id(slot.member, slot.sample)} failed: {res}")
                    continue
                record["members"].append({
                    "member": res.member, "model": res.model, "sample": res.sample,
                    "prediction": res.prediction, "flags": res.flags,
                    "draft_prediction": res.stages[0]["prediction"] if res.stages else None,
                })
                stats["forecasters_run"] += 1
            # prompt/output pairs for later RL training (export-rl)
            record["calls"] += [{"role": c.role, "model": c.model, "messages": c.messages, "output": c.output}
                                for c in ctx.calls if _keep_call(c)]
            record["cost"] += ctx.total_cost
            record["warnings"] += [w for w in ctx.warnings if w not in record["warnings"]]
            path.write_text(json.dumps(record, ensure_ascii=False, default=str), encoding="utf-8")
            stats["cost"] += ctx.total_cost

    await asyncio.gather(*(one(*w) for w in work))
    return stats


def load_run(cfg: Config, recompute: bool = True) -> dict[str, dict]:
    """
    key -> {"kind", "prediction", "members"} for this config: members restricted to the
    config's forecaster slots, prediction re-aggregated with the config's aggregation settings.
    Questions missing any slot are left out.
    """
    out: dict[str, dict] = {}
    d = run_dir(cfg)
    if not d.exists():
        return out
    slots = [_slot_id(s.member, s.sample) for s in needed_slots(cfg)]
    views = {qv.key: qv for qv in load_questions(cfg)}
    for p in d.glob("*.json"):
        if p.name.startswith("_"):
            continue
        rec = json.loads(p.read_text(encoding="utf-8"))
        qv = views.get(rec["key"])
        by_slot = {_slot_id(m["member"], m["sample"]): m for m in rec["members"]}
        members = [by_slot[s] for s in slots if s in by_slot]
        if qv is None or len(members) < len(slots):
            continue
        pred = aggregate(qv, [m["prediction"] for m in members], [m["member"] for m in members],
                         cfg.get("aggregation", {}), cfg.get("output", {}))
        out[rec["key"]] = {"kind": rec["kind"], "prediction": pred, "members": members,
                           "cost": rec.get("cost", 0.0) * len(slots) / max(len(rec["members"]), 1),
                           "calls": rec.get("calls", [])}
    return out
