"""Synthetic resolved questions for offline tests (no Metaculus access needed)."""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from forecasting_tools import BinaryQuestion, DateQuestion, MultipleChoiceQuestion, NumericQuestion


def make_questions(n: int = 24, seed: int = 0) -> list:
    rng = random.Random(seed)
    base = datetime(2025, 3, 1, tzinfo=timezone.utc)
    out = []
    for i in range(n):
        opened = base + timedelta(days=7 * i)
        resolves = opened + timedelta(days=rng.randint(30, 120))
        common = dict(
            id_of_post=900000 + i, id_of_question=800000 + i,
            page_url=f"https://www.metaculus.com/questions/{900000 + i}/",
            background_info="Synthetic test question.", fine_print="",
            resolution_criteria="Resolves according to official sources.",
            open_time=opened, published_time=opened, close_time=resolves - timedelta(days=1),
            scheduled_resolution_time=resolves, actual_resolution_time=resolves,
        )
        kind = i % 4
        if kind in (0, 1):
            out.append(BinaryQuestion(question_text=f"Will synthetic event {i} happen before {resolves.date()}?",
                                      resolution_string="yes" if rng.random() < 0.35 else "no", **common))
        elif kind == 2:
            opts = ["Alpha", "Beta", "Gamma", "Delta"]
            out.append(MultipleChoiceQuestion(question_text=f"Which party wins synthetic election {i}?", options=opts,
                                              resolution_string=rng.choice(opts), **common))
        elif i % 8 == 3:
            lo, hi = 0.0, 1000.0
            out.append(NumericQuestion(question_text=f"How many synthetic units in case {i}?",
                                       lower_bound=lo, upper_bound=hi, open_lower_bound=False,
                                       open_upper_bound=True, unit_of_measure="units",
                                       resolution_string=str(round(rng.uniform(100, 900), 1)), **common))
        else:
            lo_d, hi_d = opened, opened + timedelta(days=365)
            res = lo_d + timedelta(days=rng.randint(20, 300))
            out.append(DateQuestion(question_text=f"When will synthetic milestone {i} happen?",
                                    lower_bound=lo_d, upper_bound=hi_d, open_lower_bound=False,
                                    open_upper_bound=True, resolution_string=res.isoformat(), **common))
    return out
