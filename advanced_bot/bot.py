"""
AdvancedBot: a forecasting-tools ForecastBot whose research / forecast /
aggregation steps are delegated to advanced_bot.pipeline.

ForecastBot is configured with 1 research report and 1 "prediction" per
question: our pipeline runs the whole ensemble + critique loop inside that one
prediction call and returns the already-aggregated result.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from forecasting_tools import (
    BinaryQuestion,
    DateQuestion,
    ForecastBot,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
    ReasonedPrediction,
)

from advanced_bot import llm
from advanced_bot.config import Config
from advanced_bot.llm import RunContext
from advanced_bot.pipeline import QuestionForecast, forecast_from_research, readable, to_forecasting_tools
from advanced_bot.question_view import QuestionView
from advanced_bot.researcher import ResearchResult, run_research, summarize

logger = logging.getLogger(__name__)


class AdvancedBot(ForecastBot):
    def __init__(self, cfg: Config, publish: bool = False, max_concurrent_questions: int = 2, **kwargs) -> None:
        self.cfg = cfg
        models = cfg.get("models", {})
        super().__init__(
            research_reports_per_question=1,
            predictions_per_research_report=1,
            use_research_summary_to_forecast=False,
            publish_reports_to_metaculus=publish,
            enable_summarize_research=bool(cfg.get_path("summary.enabled", True)),
            extra_metadata_in_explanation=True,
            llms={  # informational only (shown in reports); the pipeline picks models from cfg
                "default": cfg.get_path("ensemble.members")[0]["model"],
                "summarizer": models.get("summarizer"),
                "researcher": models.get("researcher"),
                "parser": models.get("summarizer"),
            },
            **kwargs,
        )
        self._ctx: dict[str, RunContext] = {}
        self._research: dict[str, ResearchResult] = {}
        self._forecasts: dict[str, QuestionForecast] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent_questions)

    # ------------------------------------------------------------------ context per question

    def _context(self, question: MetaculusQuestion) -> RunContext:
        key = QuestionView(question).key
        if key not in self._ctx:
            self._ctx[key] = RunContext(cutoff=llm.utcnow())
        return self._ctx[key]

    # ------------------------------------------------------------------ ForecastBot hooks

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._semaphore:
            qv = QuestionView(question)
            result = await run_research(self._context(question), self.cfg, qv)
            self._research[qv.key] = result
            return result.report

    async def summarize_research(self, question: MetaculusQuestion, research: str) -> str:
        if not self.enable_summarize_research:
            return "Summary disabled"
        qv = QuestionView(question)
        result = self._research.get(qv.key)
        if result is None:
            return await super().summarize_research(question, research)
        result.summary = await summarize(self._context(question), self.cfg, qv, result)
        return result.summary

    async def _forecast(self, question: MetaculusQuestion, research: str) -> ReasonedPrediction:
        qv = QuestionView(question)
        ctx = self._context(question)
        result = self._research.get(qv.key) or ResearchResult(report=research, sources=[], window=["", ""])
        fc = await forecast_from_research(ctx, self.cfg, qv, result)
        self._forecasts[qv.key] = fc
        await self._write_log(qv, ctx, result, fc)
        reasoning = "\n\n".join(m.reasoning for m in fc.members)
        return ReasonedPrediction(prediction_value=to_forecasting_tools(qv, fc.prediction), reasoning=reasoning)

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str):
        return await self._forecast(question, research)

    async def _run_forecast_on_multiple_choice(self, question: MultipleChoiceQuestion, research: str):
        return await self._forecast(question, research)

    async def _run_forecast_on_numeric(self, question: NumericQuestion, research: str):
        return await self._forecast(question, research)

    async def _run_forecast_on_date(self, question: DateQuestion, research: str):
        return await self._forecast(question, research)

    def _create_unified_explanation(self, question, research_prediction_collections, aggregated_prediction,
                                    final_cost, time_spent_in_minutes) -> str:
        fc = self._forecasts.get(QuestionView(question).key)
        if fc is None:
            return super()._create_unified_explanation(question, research_prediction_collections,
                                                       aggregated_prediction, final_cost, time_spent_in_minutes)
        text = fc.explanation
        return text if len(text) < 140000 else text[:139000] + "\n\n(truncated)"

    # ------------------------------------------------------------------ per-question log

    async def _write_log(self, qv: QuestionView, ctx: RunContext, research: ResearchResult,
                         fc: QuestionForecast) -> None:
        folder = Path(self.cfg.get_path("output.log_dir", "logs/advanced")) / datetime.now(timezone.utc).strftime("%Y-%m-%d")
        folder.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "-", qv.q.question_text.lower()).strip("-")[:50]
        base = folder / f"{qv.key}_{slug}"
        arabic = ""
        if self.cfg.get_path("output.arabic_summary"):
            try:
                arabic = await arabic_summary(ctx, self.cfg, qv, fc)
            except Exception as e:  # a log nicety must never break a forecast
                logger.warning(f"Arabic summary failed: {e}")
        md = fc.explanation + (f"\n\n# ملخص بالعربية\n<div dir=\"rtl\">\n\n{arabic}\n\n</div>\n" if arabic else "")
        base.with_suffix(".md").write_text(md, encoding="utf-8")
        record = {
            "question": {"key": qv.key, "url": qv.q.page_url, "text": qv.q.question_text, "type": qv.kind},
            "config": self.cfg.name, "config_fingerprint": self.cfg.fingerprint(),
            "prediction": fc.prediction, "readable": readable(qv, fc.prediction),
            "cost_usd": ctx.total_cost, "cost_by_model": ctx.cost_by_model(),
            "seconds": ctx.elapsed_seconds, "models_used": ctx.models_used(),
            "members": [{"member": m.member, "model": m.model, "sample": m.sample,
                         "prediction": m.prediction, "flags": m.flags} for m in fc.members],
            "warnings": fc.warnings, "tool_log": ctx.tool_log, "published": self.publish_reports_to_metaculus,
        }
        base.with_suffix(".json").write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str),
                                              encoding="utf-8")
        logger.info(f"Log written: {base}.md")


async def arabic_summary(ctx: RunContext, cfg: Config, qv: QuestionView, fc: QuestionForecast) -> str:
    prompt = (
        "اكتب ملخصاً عربياً مبسطاً (150-200 كلمة) لهذا التوقع لقارئ غير متخصص: السؤال، التوقع النهائي، "
        "أهم الأدلة مع أرقام المصادر [n]، وأهم نقاط عدم اليقين. لا تغيّر الأرقام.\n\n"
        f"Question: {qv.q.question_text}\nFinal forecast: {readable(qv, fc.prediction)}\n\n"
        f"{fc.explanation[:12000]}"
    )
    res = await llm.complete(ctx, "summarizer", cfg.get_path("models.summarizer"),
                             [{"role": "user", "content": prompt}], temperature=0.2, meta={"question": qv})
    return res.text.strip()
