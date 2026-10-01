"""
Combining the forecasters' predictions. All methods work on the normalized
prediction shapes from checks.py and are pure functions, so the evaluation code
can re-aggregate stored member predictions under different settings for free.

Weights are per ensemble MEMBER (model family); samples of the same member share
that member's weight equally.
"""
from __future__ import annotations

import math
import statistics

from advanced_bot.checks import apply_mc_floor, repair_percentiles
from advanced_bot.question_view import PERCENTILES, QuestionView

EPS = 1e-6


def logit(p: float) -> float:
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def forecaster_weights(member_names: list[str], weights: dict[str, float], min_weight: float) -> list[float]:
    """One weight per forecaster (member_names may repeat when a member has several samples)."""
    members = list(dict.fromkeys(member_names))
    raw = {m: float(weights.get(m, 1.0)) if weights else 1.0 for m in members}
    total = sum(raw.values()) or 1.0
    norm = {m: max(raw[m] / total, min_weight) for m in members}
    total = sum(norm.values())
    norm = {m: v / total for m, v in norm.items()}
    counts = {m: member_names.count(m) for m in members}
    return [norm[m] / counts[m] for m in member_names]


def aggregate(
    qv: QuestionView,
    predictions: list[dict],
    member_names: list[str],
    agg_cfg: dict,
    output_cfg: dict,
) -> dict:
    if not predictions:
        raise ValueError("No predictions to aggregate")
    w = forecaster_weights(member_names, agg_cfg.get("weights") or {}, float(agg_cfg.get("min_weight", 0.0)))
    d = float(agg_cfg.get("extremize", 1.0))

    if qv.kind == "binary":
        p = aggregate_binary([x["p"] for x in predictions], w, agg_cfg.get("binary", "weighted_logodds"), d)
        lo, hi = output_cfg.get("min_probability", 0.01), output_cfg.get("max_probability", 0.99)
        return {"p": min(max(p, lo), hi)}
    if qv.kind == "multiple_choice":
        probs = aggregate_mc([x["probs"] for x in predictions], qv.options, w,
                             agg_cfg.get("multiple_choice", "mean"), d)
        return {"probs": apply_mc_floor(probs, output_cfg.get("mc_min_probability", 0.01))}
    values = aggregate_quantiles([x["percentiles"] for x in predictions], w, agg_cfg.get("numeric", "weighted_quantile_mean"))
    return {"percentiles": [[p, v] for p, v in zip(PERCENTILES, repair_percentiles(qv, values))]}


def aggregate_binary(ps: list[float], w: list[float], method: str, extremize: float = 1.0) -> float:
    if method == "median":
        base = statistics.median(ps)
    elif method == "mean":
        base = sum(wi * p for wi, p in zip(w, ps)) / sum(w)
    elif method == "weighted_logodds":
        base = sigmoid(sum(wi * logit(p) for wi, p in zip(w, ps)) / sum(w))
    else:
        raise ValueError(f"Unknown binary aggregation {method}")
    return sigmoid(extremize * logit(base)) if extremize != 1.0 else base


def aggregate_mc(dists: list[dict[str, float]], options: list[str], w: list[float], method: str,
                 extremize: float = 1.0) -> dict[str, float]:
    if method == "median":
        out = {o: statistics.median(d[o] for d in dists) for o in options}
    elif method == "mean":
        out = {o: sum(wi * d[o] for wi, d in zip(w, dists)) / sum(w) for o in options}
    elif method == "weighted_logodds":
        # logarithmic pooling: weighted geometric mean, renormalized
        out = {o: math.exp(sum(wi * math.log(max(d[o], EPS)) for wi, d in zip(w, dists)) / sum(w)) for o in options}
    else:
        raise ValueError(f"Unknown multiple-choice aggregation {method}")
    if extremize != 1.0:
        out = {o: max(v, EPS) ** extremize for o, v in out.items()}
    s = sum(out.values())
    return {o: v / s for o, v in out.items()}


def aggregate_quantiles(percentile_lists: list[list[list[float]]], w: list[float], method: str) -> list[float]:
    """Quantile averaging (Vincentization): average the VALUE at each percentile level."""
    rows = [[v for _, v in pl] for pl in percentile_lists]
    if method == "weighted_quantile_mean":
        return [sum(wi * r[i] for wi, r in zip(w, rows)) / sum(w) for i in range(len(PERCENTILES))]
    if method == "median_quantile":
        return [statistics.median(r[i] for r in rows) for i in range(len(PERCENTILES))]
    raise ValueError(f"Unknown numeric aggregation {method}")
