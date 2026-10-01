"""
Scoring rules. Per-question scores are returned as a dict; "lower is better"
unless noted.

  binary / multiple choice : brier, log_loss (= -ln p(outcome)), baseline (Metaculus, higher is better)
  numeric / date           : crps (normalized by question range), pinball (avg over the 15
                             percentiles, normalized), baseline (Metaculus-style, higher is better)

`baseline` exists for every type, so it is the headline metric when comparing
configs across mixed question types.

Metaculus continuous baseline (approximation of the official rule): the CDF is
evaluated on the question's grid; pmf = probability of the bucket containing
the outcome; baseline pmf = uniform over inner buckets, with 5% per open
out-of-bounds tail. score = 100 * ln(pmf / baseline_pmf) / 2.
"""
from __future__ import annotations

import math

import numpy as np

from advanced_bot.question_view import QuestionView

P_FLOOR = 1e-4


def score_question(qv: QuestionView, prediction: dict, outcome: dict) -> dict:
    if qv.kind == "binary":
        return binary_scores(prediction["p"], outcome["yes"])
    if qv.kind == "multiple_choice":
        return mc_scores(prediction["probs"], outcome["option"])
    return continuous_scores(qv, prediction["percentiles"], outcome["value"])


def binary_scores(p: float, yes: bool) -> dict:
    y = 1.0 if yes else 0.0
    p_out = min(max(p if yes else 1 - p, P_FLOOR), 1.0)
    return {
        "brier": (p - y) ** 2,
        "log_loss": -math.log(p_out),
        "baseline": 100 * (math.log2(p_out) + 1),
    }


def mc_scores(probs: dict[str, float], outcome: str) -> dict:
    n = len(probs)
    p_out = min(max(probs.get(outcome, 0.0), P_FLOOR), 1.0)
    return {
        "brier": sum((p - (1.0 if k == outcome else 0.0)) ** 2 for k, p in probs.items()),
        "log_loss": -math.log(p_out),
        "baseline": 100 * (math.log(p_out) / math.log(n) + 1),
    }


def cdf_on_grid(qv: QuestionView, percentiles: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    """Metaculus-style CDF (x grid, F) built by forecasting-tools from the 15 percentiles."""
    from forecasting_tools import NumericDistribution, Percentile

    dist = NumericDistribution.from_question(
        [Percentile(percentile=p / 100, value=v) for p, v in percentiles], qv.q  # type: ignore[arg-type]
    )
    cdf = dist.get_cdf()
    return np.array([c.value for c in cdf]), np.array([c.percentile for c in cdf])


def continuous_scores(qv: QuestionView, percentiles: list[list[float]], y: float) -> dict:
    rng = qv.upper - qv.lower
    # pinball loss on the declared quantiles (scale-free)
    pin = 0.0
    for p, q in percentiles:
        tau = p / 100
        pin += max(tau * (y - q), (tau - 1) * (y - q))
    pin = pin / len(percentiles) / rng

    x, F = cdf_on_grid(qv, percentiles)
    # CRPS over the question range (outcome clipped to range), normalized by range
    yc = min(max(y, x[0]), x[-1])
    step = (x >= yc).astype(float)
    crps = float(np.trapezoid((F - step) ** 2, x)) / rng

    # baseline score from the pmf of the bucket containing the outcome
    n_inner = len(F) - 1
    open_lo, open_hi = qv.open_lower, qv.open_upper
    if y < x[0]:
        pmf, base = F[0], 0.05 if open_lo else None
    elif y > x[-1]:
        pmf, base = 1 - F[-1], 0.05 if open_hi else None
    else:
        i = min(int(np.searchsorted(x, y, side="right")) - 1, n_inner - 1)
        pmf = F[i + 1] - F[i]
        base = (1 - 0.05 * (open_lo + open_hi)) / n_inner
    if base is None:  # outcome outside a closed bound: data problem, score as the inner edge
        base = (1 - 0.05 * (open_lo + open_hi)) / n_inner
        pmf = (F[1] - F[0]) if y < x[0] else (F[-1] - F[-2])
    baseline = 100 * math.log(max(pmf, 1e-9) / base) / 2
    return {"crps": crps, "pinball": pin, "baseline": baseline}


# metric -> True if higher is better
HIGHER_IS_BETTER = {"baseline": True, "brier": False, "log_loss": False, "crps": False, "pinball": False}


def calibration_table(pairs: list[tuple[float, bool]], bins: int = 10) -> list[dict]:
    """[(predicted probability, happened)] -> rows per probability bin."""
    rows = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [(p, y) for p, y in pairs if (lo <= p < hi) or (b == bins - 1 and p == 1.0)]
        if not sel:
            continue
        rows.append({
            "bin": f"{lo:.0%}-{hi:.0%}",
            "n": len(sel),
            "mean_predicted": sum(p for p, _ in sel) / len(sel),
            "observed": sum(1 for _, y in sel if y) / len(sel),
        })
    return rows


def calibration_pairs(qv: QuestionView, prediction: dict, outcome: dict) -> list[tuple[float, bool]]:
    if qv.kind == "binary":
        return [(prediction["p"], outcome["yes"])]
    if qv.kind == "multiple_choice":
        return [(p, k == outcome["option"]) for k, p in prediction["probs"].items()]
    # continuous: is the outcome below each declared percentile? (PIT-style calibration)
    return [(p / 100, outcome["value"] <= v) for p, v in prediction["percentiles"]]


def paired_bootstrap(diffs: list[float], samples: int = 10000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Mean difference with a percentile bootstrap CI and a two-sided bootstrap p-value."""
    d = np.asarray(diffs, dtype=float)
    if len(d) == 0:
        return {"n": 0, "mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "p": float("nan")}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(samples, len(d)))
    means = d[idx].mean(axis=1)
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return {
        "n": int(len(d)),
        "mean": float(d.mean()),
        "lo": float(np.quantile(means, alpha / 2)),
        "hi": float(np.quantile(means, 1 - alpha / 2)),
        "p": float(min(p, 1.0)),
    }

