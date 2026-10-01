"""
A uniform view over forecasting-tools question classes, used by the pipeline,
the checks and the evaluation code. Date questions are handled as numeric
questions whose values are Unix timestamps.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from forecasting_tools import (
    BinaryQuestion,
    DateQuestion,
    DiscreteQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)

# The 15 percentiles every numeric/date forecast must give (in percent).
PERCENTILES = [0.1, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 99.9]

QUESTION_CLASSES = {
    "binary": BinaryQuestion,
    "multiple_choice": MultipleChoiceQuestion,
    "numeric": NumericQuestion,
    "discrete": DiscreteQuestion,
    "date": DateQuestion,
}


@dataclass
class QuestionView:
    q: MetaculusQuestion

    @property
    def kind(self) -> str:
        if isinstance(self.q, BinaryQuestion):
            return "binary"
        if isinstance(self.q, MultipleChoiceQuestion):
            return "multiple_choice"
        if isinstance(self.q, DateQuestion):
            return "date"
        if isinstance(self.q, NumericQuestion):
            return "numeric"
        raise ValueError(f"Unsupported question type {type(self.q).__name__}")

    @property
    def is_continuous(self) -> bool:
        return self.kind in ("numeric", "date")

    @property
    def options(self) -> list[str]:
        return list(getattr(self.q, "options", []) or [])

    @property
    def lower(self) -> float:
        lb = self.q.lower_bound  # type: ignore[attr-defined]
        return lb.timestamp() if isinstance(lb, datetime) else float(lb)

    @property
    def upper(self) -> float:
        ub = self.q.upper_bound  # type: ignore[attr-defined]
        return ub.timestamp() if isinstance(ub, datetime) else float(ub)

    @property
    def open_lower(self) -> bool:
        return bool(self.q.open_lower_bound)  # type: ignore[attr-defined]

    @property
    def open_upper(self) -> bool:
        return bool(self.q.open_upper_bound)  # type: ignore[attr-defined]

    @property
    def resolution_date(self) -> datetime | None:
        d = self.q.scheduled_resolution_time or self.q.close_time
        if d and d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d

    @property
    def key(self) -> str:
        """Stable id for caches: post id, plus question id for group/sub-questions."""
        return f"{self.q.id_of_post}-{self.q.id_of_question}" if self.q.id_of_question else str(self.q.id_of_post)

    def fmt_value(self, v: float) -> str:
        if self.kind == "date":
            return datetime.fromtimestamp(v, tz=timezone.utc).strftime("%Y-%m-%d")
        return f"{v:,.6g}"

    def bounds_text(self) -> str:
        if not self.is_continuous:
            return ""
        lo = "open (values below are possible)" if self.open_lower else "CLOSED (no value can be below it)"
        hi = "open (values above are possible)" if self.open_upper else "CLOSED (no value can be above it)"
        unit = f" Units: {self.q.unit_of_measure}." if self.q.unit_of_measure else ""
        return (
            f"Lower bound: {self.fmt_value(self.lower)} — {lo}.\n"
            f"Upper bound: {self.fmt_value(self.upper)} — {hi}.{unit}"
        )

    def corpus(self) -> str:
        """All text the question itself provides (counts as 'verified' for the fact check)."""
        parts = [
            self.q.question_text,
            self.q.background_info or "",
            self.q.resolution_criteria or "",
            self.q.fine_print or "",
            " ".join(self.options),
            self.bounds_text(),
        ]
        return "\n".join(parts)

    def prompt_block(self, today: str) -> str:
        lines = [
            f"Question: {self.q.question_text}",
            f"Question type: {self.kind}",
        ]
        if self.options:
            lines.append(f"Options: {self.options}")
        if self.is_continuous:
            lines.append(self.bounds_text())
        lines += [
            f"Background: {self.q.background_info or 'none'}",
            f"Resolution criteria: {self.q.resolution_criteria or 'none'}",
            f"Fine print: {self.q.fine_print or 'none'}",
            f"Scheduled resolution date: {self.resolution_date.date() if self.resolution_date else 'unknown'}",
            f"Today's date: {today}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------- outcomes (eval)

    def outcome(self) -> dict | None:
        """Ground truth in the same normalized shape as predictions, or None if unusable."""
        res = self.q.typed_resolution
        if res is None or type(res).__name__ == "CanceledResolution":
            return None
        if self.kind == "binary":
            return {"yes": bool(res)} if isinstance(res, bool) else None
        if self.kind == "multiple_choice":
            return {"option": res} if isinstance(res, str) and res in self.options else None
        name = type(res).__name__
        if name == "OutOfBoundsResolution":
            eps = (self.upper - self.lower) * 1e-6
            value = self.upper + eps if res.value == "above_upper_bound" else self.lower - eps
            return {"value": value, "out_of_bounds": res.value}
        if isinstance(res, datetime):
            return {"value": res.timestamp()}
        if isinstance(res, (int, float)):
            return {"value": float(res)}
        return None


def question_to_dict(q: MetaculusQuestion) -> dict:
    data = q.model_dump(mode="json")
    data["_class"] = type(q).__name__
    return data


def question_from_dict(data: dict) -> MetaculusQuestion:
    data = dict(data)
    cls_name = data.pop("_class", None)
    classes = {c.__name__: c for c in QUESTION_CLASSES.values()}
    cls = classes.get(cls_name) or QUESTION_CLASSES[data.get("question_type", "binary")]
    return cls.model_validate(data)
