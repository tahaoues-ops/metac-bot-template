"""
The forecasting pipeline, independent of ForecastBot so the evaluation code can
run it on cached research:

    research (or cached research)  ->  forecasters + critique  ->  aggregate  ->  markdown
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from forecasting_tools import (
    NumericDistribution,
    Percentile,
    PredictedOption,
    PredictedOptionList,
)

from advanced_bot.aggregation import aggregate
from advanced_bot.config import Config
from advanced_bot.forecasters import MemberRun, run_all_forecasters
from advanced_bot.llm import RunContext
from advanced_bot.question_view import QuestionView
from advanced_bot.researcher import ResearchResult


@dataclass
class QuestionForecast:
    question_key: str
    kind: str
    prediction: dict                 # aggregated, normalized
    members: list[MemberRun]
    explanation: str
    cost: float = 0.0
    seconds: float = 0.0
    models_used: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["members"] = [m.to_dict() for m in self.members]
        return d


async def forecast_from_research(
    ctx: RunContext, cfg: Config, qv: QuestionView, research: ResearchResult
) -> QuestionForecast:
    research_text = research.summary if cfg.get_path("summary.use_for_forecast") and research.summary else research.report
    members = await run_all_forecasters(ctx, cfg, qv, research_text)
    final = aggregate(qv, [m.prediction for m in members], [m.member for m in members],
                      cfg.get("aggregation", {}), cfg.get("output", {}))
    fc = QuestionForecast(
        question_key=qv.key, kind=qv.kind, prediction=final, members=members, explanation="",
        cost=ctx.total_cost, seconds=ctx.elapsed_seconds, models_used=ctx.models_used(),
        warnings=list(dict.fromkeys(ctx.warnings)),
    )
    fc.explanation = render_explanation(cfg, qv, research, fc, ctx)
    return fc


def readable(qv: QuestionView, pred: dict) -> str:
    if "p" in pred:
        return f"{pred['p'] * 100:.1f}%"
    if "probs" in pred:
        return ", ".join(f"{k}: {v * 100:.1f}%" for k, v in pred["probs"].items())
    pts = dict((p, v) for p, v in pred["percentiles"])
    return " | ".join(f"P{p:g}: {qv.fmt_value(pts[p])}" for p in (5, 20, 50, 80, 95))


def render_explanation(cfg: Config, qv: QuestionView, research: ResearchResult, fc: QuestionForecast,
                       ctx: RunContext) -> str:
    agg = cfg.get("aggregation", {})
    method = agg.get("binary") if qv.kind == "binary" else agg.get("multiple_choice") if qv.kind == "multiple_choice" else agg.get("numeric")
    member_lines = "\n".join(
        f"- {m.member} #{m.sample} ({m.model}): {readable(qv, m.prediction)}" for m in fc.members
    )
    cost_lines = "\n".join(f"  - {k}: ${v:.4f}" for k, v in ctx.cost_by_model().items())
    warnings = "\n".join(f"- {w}" for w in fc.warnings) or "- none"
    reasoning = "\n\n".join(m.reasoning for m in fc.members)
    return f"""# Summary
*Question*: {qv.q.question_text}
*Final prediction*: {readable(qv, fc.prediction)}
*Aggregation*: {method} (extremize={agg.get('extremize', 1.0)})
*Config*: {cfg.name}
*Cost*: ${fc.cost:.4f} (LiteLLM estimate; search APIs not included)
{cost_lines}
*Time*: {fc.seconds / 60:.1f} minutes
*Research window*: {research.window[0][:10]} to {research.window[1][:10]}, {research.iterations} search rounds

## Forecasters
{member_lines}

## Research summary
{research.summary or '(summary disabled)'}

## Automatic check warnings
{warnings}

# Research report
{research.report}

# Forecaster reasoning
{reasoning}
"""


def to_forecasting_tools(qv: QuestionView, pred: dict):
    """Normalized prediction -> the type ForecastBot expects for this question."""
    if qv.kind == "binary":
        return pred["p"]
    if qv.kind == "multiple_choice":
        return PredictedOptionList(predicted_options=[
            PredictedOption(option_name=k, probability=v) for k, v in pred["probs"].items()])
    pts = [Percentile(percentile=p / 100, value=v) for p, v in pred["percentiles"]]
    return NumericDistribution.from_question(pts, qv.q)  # type: ignore[arg-type]


def utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
