"""
Deterministic checks (code, not the model):

  parse_and_validate   strict output-format validation per question type
  arithmetic_issues    recomputes ratios / percentages / equations found in text
  unverified_claims    numbers and names a forecaster/critic introduced that are not in the research
  consistency_issue    "evidence pushes up" but the number went down (and vice versa)

Normalized prediction shapes used everywhere in the package:
  binary            {"p": 0.37}
  multiple_choice   {"probs": {"Option A": 0.5, ...}}
  numeric / date    {"percentiles": [[0.1, v], [1, v], ... [99.9, v]]}   (percent, value)
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone

from advanced_bot.question_view import PERCENTILES, QuestionView


class FormatError(ValueError):
    """Raised when model output fails validation; the message is sent back to the model."""


# ============================================================================ JSON extraction


def extract_json(text: str) -> dict:
    """Last JSON object in the text (models put the answer at the end)."""
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = fenced[::-1] if fenced else []
    if not candidates:
        # fall back to the last balanced {...}
        depth, start, spans = 0, None, []
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}" and depth:
                depth -= 1
                if depth == 0 and start is not None:
                    spans.append(text[start : i + 1])
        candidates = spans[::-1]
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    raise FormatError("No valid JSON object found. End your answer with the JSON block exactly as specified.")


def _num(x) -> float:
    if isinstance(x, bool):
        raise FormatError(f"Expected a number, got {x!r}")
    if isinstance(x, (int, float)):
        v = float(x)
    else:
        s = str(x).strip().replace(",", "").replace("%", "")
        try:
            v = float(s)
        except ValueError:
            raise FormatError(f"Expected a number, got {x!r}")
    if math.isnan(v) or math.isinf(v):
        raise FormatError(f"Expected a finite number, got {x!r}")
    return v


def _parse_value(qv: QuestionView, x) -> float:
    """Numeric values, or ISO dates for date questions (converted to timestamps)."""
    if qv.kind == "date":
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            return float(x)
        try:
            d = datetime.fromisoformat(str(x).strip().replace("Z", "+00:00"))
        except ValueError:
            raise FormatError(f"Dates must be ISO format YYYY-MM-DD, got {x!r}")
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d.timestamp()
    return _num(x)


# ============================================================================ format validation


def parse_and_validate(qv: QuestionView, obj: dict, cfg_output: dict) -> dict:
    """Turns the model's JSON into a normalized prediction, or raises FormatError."""
    if qv.kind == "binary":
        if "probability" not in obj:
            raise FormatError('Missing "probability" (a number from 0 to 100).')
        p = _num(obj["probability"])
        if not 0 <= p <= 100:
            raise FormatError(f'"probability" must be between 0 and 100, got {p}.')
        lo, hi = cfg_output.get("min_probability", 0.01), cfg_output.get("max_probability", 0.99)
        return {"p": min(max(p / 100, lo), hi)}

    if qv.kind == "multiple_choice":
        raw = obj.get("probabilities")
        if not isinstance(raw, dict):
            raise FormatError('Missing "probabilities": an object mapping EVERY option name to a percentage.')
        matched: dict[str, float] = {}
        lookup = {o.strip().lower(): o for o in qv.options}
        for k, v in raw.items():
            name = lookup.get(str(k).strip().lower())
            if name is None:
                raise FormatError(f"Unknown option {k!r}. Use exactly these names: {qv.options}")
            matched[name] = _num(v)
        missing = [o for o in qv.options if o not in matched]
        if missing:
            raise FormatError(f"Missing options {missing}. Give a percentage for every option (min 1%).")
        if any(v < 0 for v in matched.values()):
            raise FormatError("Probabilities cannot be negative.")
        total = sum(matched.values())
        if abs(total - 100) > 2:
            raise FormatError(f"Option percentages must sum to 100, they sum to {total:.1f}.")
        probs = {k: v / total for k, v in matched.items()}
        return {"probs": apply_mc_floor(probs, cfg_output.get("mc_min_probability", 0.01))}

    # numeric / date
    raw = obj.get("percentiles")
    if not isinstance(raw, dict):
        raise FormatError(f'Missing "percentiles": an object with keys {PERCENTILES}.')
    by_key = {}
    for k, v in raw.items():
        try:
            by_key[float(str(k).replace("p", "").replace("%", ""))] = v
        except ValueError:
            raise FormatError(f"Bad percentile key {k!r}; use keys like \"5\", \"50\", \"99.9\".")
    missing = [p for p in PERCENTILES if p not in by_key]
    if missing:
        raise FormatError(f"Missing percentiles {missing}. Give all 15: {PERCENTILES}.")
    values = [_parse_value(qv, by_key[p]) for p in PERCENTILES]
    return {"percentiles": validate_percentile_values(qv, values)}


def validate_percentile_values(qv: QuestionView, values: list[float], strict: bool = True) -> list[list[float]]:
    """Checks ordering and bounds. strict=False repairs instead of raising (last resort)."""
    rng = qv.upper - qv.lower
    problems = []
    for i in range(1, len(values)):
        # dates have day resolution, so equal neighbours are fine (repaired below); decreasing is not
        if values[i] < values[i - 1] or (values[i] == values[i - 1] and qv.kind != "date"):
            problems.append(
                f"Percentile {PERCENTILES[i]} ({qv.fmt_value(values[i])}) is not greater than "
                f"percentile {PERCENTILES[i - 1]} ({qv.fmt_value(values[i - 1])}); values must strictly increase."
            )
            break
    tol = rng * 1e-9
    if not qv.open_lower and min(values) < qv.lower - tol:
        problems.append(f"The lower bound {qv.fmt_value(qv.lower)} is CLOSED: no value may be below it.")
    if not qv.open_upper and max(values) > qv.upper + tol:
        problems.append(f"The upper bound {qv.fmt_value(qv.upper)} is CLOSED: no value may be above it.")
    if min(values) < qv.lower - rng or max(values) > qv.upper + rng:
        problems.append("Some values are absurdly far outside the question range; re-check units.")
    if problems and strict:
        raise FormatError(" ".join(problems))
    return [[p, v] for p, v in zip(PERCENTILES, repair_percentiles(qv, values))]


def repair_percentiles(qv: QuestionView, values: list[float]) -> list[float]:
    """Sort, clamp to closed bounds / sane range, and force strict increase."""
    rng = qv.upper - qv.lower
    lo = qv.lower if not qv.open_lower else qv.lower - rng
    hi = qv.upper if not qv.open_upper else qv.upper + rng
    vals = sorted(min(max(v, lo), hi) for v in values)
    eps = rng * 1e-4
    for i in range(1, len(vals)):
        if vals[i] <= vals[i - 1]:
            vals[i] = vals[i - 1] + eps
    # pushing up may cross a closed upper bound: shift down from the top
    if not qv.open_upper and vals[-1] > qv.upper:
        vals[-1] = qv.upper
        for i in range(len(vals) - 2, -1, -1):
            if vals[i] >= vals[i + 1]:
                vals[i] = vals[i + 1] - eps
    return vals


def apply_mc_floor(probs: dict[str, float], floor: float) -> dict[str, float]:
    """Every option >= floor, sum exactly 1."""
    n = len(probs)
    if floor * n >= 1:
        return {k: 1 / n for k in probs}
    total = sum(probs.values()) or 1.0
    p = {k: v / total for k, v in probs.items()}
    for _ in range(n):  # converges in at most n passes
        low = [k for k, v in p.items() if v < floor]
        if not low:
            break
        rest = [k for k in p if k not in low]
        rest_mass = sum(p[k] for k in rest)
        budget = 1 - floor * len(low)
        p = {k: (floor if k in low else p[k] / rest_mass * budget) for k in p}
    s = sum(p.values())
    return {k: v / s for k, v in p.items()}


# ============================================================================ arithmetic check

_NUM = r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
_SCALE = r"(?:\s*(k|thousand|m|mn|million|bn|b|billion|trillion|tn))?"
_SCALES = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mn": 1e6, "million": 1e6, "bn": 1e9, "b": 1e9,
           "billion": 1e9, "trillion": 1e12, "tn": 1e12}


def _to_float(num: str, scale: str | None = None) -> float:
    v = float(num.replace(",", ""))
    return v * _SCALES.get((scale or "").lower(), 1.0)


def _close(a: float, b: float, abs_tol: float, rel_tol: float = 0.02) -> bool:
    return abs(a - b) <= max(abs_tol, rel_tol * max(abs(a), abs(b)))


def arithmetic_issues(text: str) -> list[str]:
    """Finds explicit calculations in text and recomputes them. Returns human-readable issues."""
    issues: list[str] = []

    # "A of B (P%)", "A out of B (P%)", "A/B (P%)", "A of the last B ... (P%)"
    for m in re.finditer(
        rf"\b{_NUM}\s*(?:out of|of the (?:last|past)?|of|/)\s*{_NUM}\b[^()\n]{{0,60}}?\(\s*(?:≈|~|about\s*)?{_NUM}\s*%\s*\)",
        text,
        re.I,
    ):
        a, b, p = (_to_float(g) for g in m.groups())
        if b > 0 and a <= b and not _close(a / b * 100, p, abs_tol=1.0):
            issues.append(f'"{m.group(0).strip()}": {a:g}/{b:g} = {a / b * 100:.1f}%, not {p:g}%.')

    # explicit equations: A + B = C, A - B = C, A x B = C, A / B = C
    ops = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "−": lambda a, b: a - b, "*": lambda a, b: a * b,
           "×": lambda a, b: a * b, "x": lambda a, b: a * b, "/": lambda a, b: a / b if b else math.nan,
           "÷": lambda a, b: a / b if b else math.nan}
    for m in re.finditer(rf"{_NUM}{_SCALE}\s*([+\-−*×x/÷])\s*{_NUM}{_SCALE}\s*=\s*{_NUM}{_SCALE}", text):
        a = _to_float(m.group(1), m.group(2))
        op = m.group(3)
        b = _to_float(m.group(4), m.group(5))
        c = _to_float(m.group(6), m.group(7))
        if op == "x" and not re.search(r"\d\s*x\s*\d", m.group(0)):
            continue
        correct = ops[op](a, b)
        if not math.isnan(correct) and not _close(correct, c, abs_tol=0.01 * max(1, abs(correct)), rel_tol=0.01):
            issues.append(f'"{m.group(0).strip()}": correct result is {correct:,.4g}.')

    # "from A to B (+P%)" / "from A to B, a P% increase/decrease"
    for m in re.finditer(
        rf"from\s*\$?{_NUM}{_SCALE}\s*(?:to)\s*\$?{_NUM}{_SCALE}[^.\n]{{0,25}}?([+\-−]?)\s*{_NUM}\s*%\s*(increase|decrease|rise|drop|fall|decline|growth|higher|lower|up|down)?",
        text,
        re.I,
    ):
        a = _to_float(m.group(1), m.group(2))
        b = _to_float(m.group(3), m.group(4))
        sign, p, word = m.group(5), _to_float(m.group(6)), (m.group(7) or "").lower()
        if a == 0:
            continue
        actual = (b - a) / abs(a) * 100
        stated = -p if (sign in "-−" and sign) or word in ("decrease", "drop", "fall", "decline", "lower", "down") else p
        if not _close(actual, stated, abs_tol=1.0, rel_tol=0.05):
            issues.append(f'"{m.group(0).strip()}": change from {a:,.4g} to {b:,.4g} is {actual:+.1f}%, not {stated:+g}%.')

    # "N times in M years" with a stated per-year rate "(P% per year)"
    for m in re.finditer(
        rf"{_NUM}\s*(?:times|events|cases|occurrences)\s*in\s*(?:the\s*(?:last|past)\s*)?{_NUM}\s*years[^()\n]{{0,40}}?\(\s*(?:≈|~)?\s*{_NUM}\s*%\s*(?:per|a|each)\s*year\s*\)",
        text,
        re.I,
    ):
        n, years, p = (_to_float(g) for g in m.groups())
        if years > 0 and not _close(n / years * 100, p, abs_tol=1.0):
            issues.append(f'"{m.group(0).strip()}": {n:g}/{years:g} years = {n / years * 100:.1f}% per year, not {p:g}%.')

    return list(dict.fromkeys(issues))


# ============================================================================ fact verification

_STOP_CAPS = set(
    """A An The This That These Those It Its I We You He She They My Our Your Their His Her
    If When While Because Although However Therefore Thus Also But And Or Not No Yes So Then
    Given Based Overall Finally First Second Third Fourth Fifth Moreover Furthermore Meanwhile
    Status Quo Base Rate Outside Inside View Low High Scenario Scenarios Forecast Forecasts
    Probability Probabilities Evidence Criticism Criticisms Defense Accepted Updated New Insights
    Time Expert Experts Market Markets Trend Trends Research Report Question Answer Summary
    Percentile Percentiles Option Options Yes No None Today Tomorrow Monday Tuesday Wednesday
    Thursday Friday Saturday Sunday January February March April May June July August September
    October November December Q1 Q2 Q3 Q4 JSON Key Mechanisms Information Gaps Current Situation
    Latest Data Prediction Models Historical Rates Note Notes Unverified Critic Draft Round""".split()
)


def _norm(s: str) -> str:
    return re.sub(r"[\s,]+", " ", s.lower()).strip()


def _strip_json_and_citations(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"\{[^{}]*\}\s*$", " ", text.strip(), flags=re.S)
    return text


def unverified_claims(text: str, research: str, question_corpus: str, max_items: int = 12) -> list[str]:
    """
    Specific facts (exact figures, names) in `text` that appear nowhere in the
    research or the question. Probabilities/percentages are opinions and are skipped.
    """
    corpus = _norm(research + "\n" + question_corpus)
    corpus_digits = re.sub(r"[^\d.]", " ", research + " " + question_corpus)
    corpus_numbers = {n.rstrip(".") for n in corpus_digits.split() if n}
    body = _strip_json_and_citations(text)
    found: list[str] = []

    # numbers: skip percentages, tiny integers, and citation markers [n]
    for m in re.finditer(rf"(?<![\[\w.]){_NUM}(?:\s*(k|thousand|million|billion|bn|trillion))?(?!\s*%|\]|\w)", body):
        raw = m.group(1).replace(",", "")
        try:
            val = float(raw)
        except ValueError:
            continue
        if "." not in raw and val <= 12:
            continue
        if raw.rstrip("0").rstrip(".") in {n.rstrip("0").rstrip(".") for n in corpus_numbers} or raw in corpus_numbers:
            continue
        start = max(0, m.start() - 40)
        snippet = body[start : m.end() + 30].replace("\n", " ").strip()
        found.append(f"figure {m.group(0).strip()} (\"…{snippet}…\")")

    # proper names: runs of Capitalized words not at sentence start, not in corpus
    for m in re.finditer(r"(?<![.!?:\n]\s)(?<!^)\b([A-Z][a-zA-Z\-]+(?:\s+(?:[A-Z][a-zA-Z\-]+|of|de|al|bin|von))*)", body):
        name = m.group(1).strip()
        words = name.split()
        if all(w in _STOP_CAPS for w in words):
            continue
        name_core = " ".join(w for w in words if w not in _STOP_CAPS)
        if len(name_core) < 3 or _norm(name_core) in corpus:
            continue
        found.append(f"name \"{name_core}\"")

    return list(dict.fromkeys(found))[:max_items]


# ============================================================================ consistency


def consistency_issue(qv: QuestionView, obj: dict, prediction: dict, previous: dict | None) -> str | None:
    """
    The model states the direction evidence pushes ("up"/"down"/"neutral") relative to
    its outside view (base rate / status quo), and in critique rounds the direction of
    its update. If the number moved the other way, return an explanation request.
    """
    direction = str(obj.get("evidence_direction", "")).lower().strip()
    update_dir = str(obj.get("update_direction", "")).lower().strip()

    def central(pred: dict) -> float | None:
        if "p" in pred:
            return pred["p"] * 100
        if "percentiles" in pred:
            return dict((p, v) for p, v in pred["percentiles"])[50]
        return None

    now = central(prediction)
    if now is None:
        return None  # multiple choice: no single direction

    anchor_key = "base_rate" if qv.kind == "binary" else "status_quo_value"
    if anchor_key in obj and direction in ("up", "down"):
        try:
            anchor = _parse_value(qv, obj[anchor_key]) if qv.is_continuous else _num(obj[anchor_key])
        except FormatError:
            anchor = None
        if anchor is not None:
            tol = 1.0 if qv.kind == "binary" else abs(anchor) * 0.01 + (qv.upper - qv.lower) * 0.005
            if direction == "up" and now < anchor - tol:
                return (f'You said the evidence pushes UP from your outside view ({anchor_key} = {obj[anchor_key]}), '
                        f"but your forecast ({qv.fmt_value(now) if qv.is_continuous else f'{now:.0f}%'}) is BELOW it.")
            if direction == "down" and now > anchor + tol:
                return (f'You said the evidence pushes DOWN from your outside view ({anchor_key} = {obj[anchor_key]}), '
                        f"but your forecast ({qv.fmt_value(now) if qv.is_continuous else f'{now:.0f}%'}) is ABOVE it.")

    if previous is not None and update_dir in ("up", "down"):
        before = central(previous)
        if before is not None:
            tol = 0.5 if qv.kind == "binary" else (qv.upper - qv.lower) * 0.002
            if update_dir == "up" and now < before - tol:
                return f"You said you updated UP, but your forecast went down ({before:.4g} -> {now:.4g})."
            if update_dir == "down" and now > before + tol:
                return f"You said you updated DOWN, but your forecast went up ({before:.4g} -> {now:.4g})."
    return None
