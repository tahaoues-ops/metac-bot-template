"""
Mixture-of-scenarios tool (inspired by Mantic).

Instead of writing 15 percentiles directly, the forecaster describes 1-4
scenarios, each with a weight and a distribution:

  normal      {"mean": m, "sd": s}
  lognormal   {"median": m, "sigma": s}            (sigma = sd of log(value); value > 0)
  student_t   {"loc": m, "scale": s, "df": d}

Code computes the mixture CDF, applies the question bounds (closed bound ->
the distribution is truncated there; open bound -> mass may lie beyond it) and
reads off the percentiles.

Date questions: parameters are in DAYS FROM TODAY.
Binary "before date Y" questions (time-to-event): scenarios describe when the
event happens (days from today) plus p_never; P(yes) = P(event before Y).
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from advanced_bot.checks import FormatError, _num
from advanced_bot.question_view import PERCENTILES, QuestionView

DISTRIBUTIONS = ("normal", "lognormal", "student_t")
SECONDS_PER_DAY = 86400.0


def parse_mixture(raw) -> list[dict]:
    if not isinstance(raw, list) or not 1 <= len(raw) <= 4:
        raise FormatError('"mixture" must be a list of 1 to 4 scenarios.')
    out = []
    for i, sc in enumerate(raw):
        if not isinstance(sc, dict):
            raise FormatError(f"Scenario {i + 1} must be an object.")
        dist = str(sc.get("distribution", "")).lower().replace("-", "_")
        if dist not in DISTRIBUTIONS:
            raise FormatError(f"Scenario {i + 1}: distribution must be one of {DISTRIBUTIONS}.")
        w = _num(sc.get("weight", 0))
        if w <= 0:
            raise FormatError(f"Scenario {i + 1}: weight must be > 0.")
        p = sc.get("params") or {}
        try:
            if dist == "normal":
                params = {"mean": _num(p["mean"]), "sd": _num(p["sd"])}
                ok = params["sd"] > 0
            elif dist == "lognormal":
                params = {"median": _num(p["median"]), "sigma": _num(p["sigma"])}
                ok = params["median"] > 0 and params["sigma"] > 0
            else:
                params = {"loc": _num(p["loc"]), "scale": _num(p["scale"]), "df": _num(p.get("df", 3))}
                ok = params["scale"] > 0 and params["df"] > 0
        except KeyError as e:
            raise FormatError(f"Scenario {i + 1} ({dist}) is missing parameter {e}.")
        if not ok:
            raise FormatError(f"Scenario {i + 1}: scale/sd/sigma/median/df must be positive.")
        out.append({"name": str(sc.get("name", f"scenario {i + 1}")), "weight": w,
                    "distribution": dist, "params": params})
    total = sum(s["weight"] for s in out)
    for s in out:
        s["weight"] /= total
    return out


def _component_cdf(sc: dict, x: np.ndarray) -> np.ndarray:
    p = sc["params"]
    if sc["distribution"] == "normal":
        return stats.norm.cdf(x, loc=p["mean"], scale=p["sd"])
    if sc["distribution"] == "lognormal":
        return stats.lognorm.cdf(np.maximum(x, 1e-300), s=p["sigma"], scale=p["median"]) * (x > 0)
    return stats.t.cdf(x, df=p["df"], loc=p["loc"], scale=p["scale"])


def mixture_cdf(scenarios: list[dict], x: np.ndarray) -> np.ndarray:
    return sum(sc["weight"] * _component_cdf(sc, x) for sc in scenarios)


def mixture_percentiles(qv: QuestionView, scenarios: list[dict], today_ts: float) -> list[float]:
    """15 percentile values in the question's units (timestamps for date questions)."""
    is_date = qv.kind == "date"
    to_native = (lambda v: (v - today_ts) / SECONDS_PER_DAY) if is_date else (lambda v: v)
    from_native = (lambda d: today_ts + d * SECONDS_PER_DAY) if is_date else (lambda d: d)
    lo, hi = to_native(qv.lower), to_native(qv.upper)
    rng = hi - lo
    a = lo if not qv.open_lower else lo - rng
    b = hi if not qv.open_upper else hi + rng

    grid = np.linspace(a, b, 20001)
    F = mixture_cdf(scenarios, grid)
    lo_mass = float(mixture_cdf(scenarios, np.array([lo]))[0]) if not qv.open_lower else 0.0
    hi_mass = float(mixture_cdf(scenarios, np.array([hi]))[0]) if not qv.open_upper else 1.0
    inside = hi_mass - lo_mass
    if inside < 1e-4:
        raise FormatError("Your mixture puts (almost) no probability inside the question's allowed range; "
                          "check units and parameters.")
    G = np.clip((F - lo_mass) / inside, 0, 1)        # truncation at closed bounds
    G = np.maximum.accumulate(G)
    targets = np.array(PERCENTILES) / 100
    # first grid point where G >= target
    idx = np.searchsorted(G, targets, side="left").clip(0, len(grid) - 1)
    values = grid[idx]
    return [from_native(float(v)) for v in values]


def mixture_summary(qv: QuestionView, scenarios: list[dict]) -> str:
    unit = "days from today" if qv.kind == "date" else (qv.q.unit_of_measure or "units")
    lines = [f"Mixture model ({unit}):"]
    for s in scenarios:
        params = ", ".join(f"{k}={v:g}" for k, v in s["params"].items())
        lines.append(f"- {s['name']}: weight {s['weight']:.0%}, {s['distribution']}({params})")
    return "\n".join(lines)


def time_to_event_probability(raw: dict, days_to_resolution: float) -> tuple[float, list[dict], float]:
    """P(event happens before the resolution date). Mass at t<=0 is treated as 'not yet' and renormalized."""
    if not isinstance(raw, dict):
        raise FormatError('"time_to_event" must be an object with "p_never" and "scenarios".')
    p_never = _num(raw.get("p_never", 0))
    if not 0 <= p_never <= 1:
        raise FormatError('"p_never" must be between 0 and 1.')
    scenarios = parse_mixture(raw.get("scenarios"))
    f0 = float(mixture_cdf(scenarios, np.array([0.0]))[0])
    fd = float(mixture_cdf(scenarios, np.array([max(days_to_resolution, 0.0)]))[0])
    p_by_date = (fd - f0) / max(1 - f0, 1e-9)
    return (1 - p_never) * p_by_date, scenarios, p_never
