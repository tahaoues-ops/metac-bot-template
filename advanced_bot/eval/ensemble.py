"""
Section 2: weighted ensemble analysis on stored eval forecasts (no new LLM calls).

  * member-level forecasts: samples of the same model family are pooled first
  * weights (with a floor, so no model is removed) + extremizing factor are fitted
    by minimizing log loss on binary + multiple-choice questions
  * everything is reported with 5-fold cross-validation, so the gain is out-of-sample
  * correlation matrix of members' log-odds (binary questions)
  * leave-one-out: how much worse the CV log loss gets when each member is removed
"""
from __future__ import annotations

import math

import numpy as np
import yaml
from scipy.optimize import minimize

from advanced_bot.config import Config
from advanced_bot.eval.data import load_questions, load_run
from advanced_bot.eval.metrics import paired_bootstrap

EPS = 1e-6


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def build_matrix(cfg: Config):
    """
    Returns (members, items) where each item is
      ("binary", logits[k], y)                    logits: member log-odds
      ("mc", logprobs[k, n_options], idx_true)
    plus continuous items for reporting.
    """
    run = load_run(cfg)
    views = {qv.key: qv for qv in load_questions(cfg)}
    members = list(dict.fromkeys(m["member"] for rec in run.values() for m in rec["members"]))
    items, continuous = [], []
    for key, rec in run.items():
        qv = views[key]
        outcome = qv.outcome()
        if outcome is None:
            continue
        by_member: dict[str, list] = {}
        for m in rec["members"]:
            by_member.setdefault(m["member"], []).append(m["prediction"])
        if set(by_member) != set(members):
            continue
        if qv.kind == "binary":
            logits = np.array([np.mean(_logit(np.array([p["p"] for p in by_member[m]]))) for m in members])
            items.append(("binary", logits, 1.0 if outcome["yes"] else 0.0))
        elif qv.kind == "multiple_choice":
            opts = qv.options
            lp = np.array([np.mean([[math.log(max(p["probs"][o], EPS)) for o in opts] for p in by_member[m]], axis=0)
                           for m in members])
            lp = lp - np.log(np.exp(lp).sum(axis=1, keepdims=True))
            items.append(("mc", lp, opts.index(outcome["option"])))
        else:
            continuous.append((qv, {m: by_member[m] for m in members}, outcome["value"]))
    return members, items, continuous


def weights_from(theta: np.ndarray, floor: float) -> np.ndarray:
    k = len(theta) + 1
    z = np.concatenate([[0.0], theta])          # first weight's logit fixed at 0 (identifiability)
    soft = np.exp(z - z.max())
    soft /= soft.sum()
    return floor + (1 - k * floor) * soft


def item_losses(items, w: np.ndarray, d: float) -> np.ndarray:
    out = []
    for kind, x, y in items:
        if kind == "binary":
            z = d * float(w @ x)
            p = 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))
            p_true = p if y == 1.0 else 1 - p
            out.append(-math.log(max(p_true, EPS)))
        else:
            s = d * (w @ x)
            s = s - s.max()
            lp = s - math.log(np.exp(s).sum())
            out.append(-float(lp[y]))
    return np.array(out)


def fit(items, k: int, floor: float, fit_extremize: bool, fit_weights: bool) -> tuple[np.ndarray, float]:
    if not items:
        return np.full(k, 1 / k), 1.0
    equal = np.full(k, 1 / k)

    def unpack(v):
        w = weights_from(v[: k - 1], floor) if fit_weights else equal
        d = float(np.exp(np.clip(v[-1], math.log(0.5), math.log(3.0)))) if fit_extremize else 1.0
        return w, d

    x0 = np.zeros(k)
    res = minimize(lambda v: item_losses(items, *unpack(v)).mean(), x0, method="Nelder-Mead",
                   options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-7})
    return unpack(res.x)


def cv_losses(items, k: int, floor: float, fit_extremize: bool, fit_weights: bool, folds: int = 5,
              seed: int = 0) -> np.ndarray:
    """Out-of-sample per-item losses."""
    n = len(items)
    order = np.random.default_rng(seed).permutation(n)
    losses = np.zeros(n)
    for f in range(folds):
        test = order[f::folds]
        train = [items[i] for i in order if i not in set(test)]
        w, d = fit(train, k, floor, fit_extremize, fit_weights)
        losses[test] = item_losses([items[i] for i in test], w, d)
    return losses


def ensemble_report(cfg: Config, seed: int = 0) -> str:
    floor = float(cfg.get_path("aggregation.min_weight", 0.05))
    members, items, continuous = build_matrix(cfg)
    k = len(members)
    if k < 2 or len(items) < 10:
        return (f"Need >= 2 members and >= 10 binary/multiple-choice questions with all members "
                f"(have {k} members, {len(items)} questions). Run more questions first.")
    lines = [f"# Ensemble analysis — config '{cfg.name}'",
             f"{len(items)} binary/multiple-choice questions, members: {members}, weight floor {floor}", ""]

    # 1. out-of-sample comparison of aggregation variants
    variants = {
        "equal weights, no extremizing": (False, False),
        "equal weights + extremizing": (True, False),
        "fitted weights, no extremizing": (False, True),
        "fitted weights + extremizing": (True, True),
    }
    cv = {name: cv_losses(items, k, floor, fe, fw, seed=seed) for name, (fe, fw) in variants.items()}
    base = cv["equal weights, no extremizing"]
    lines += ["## Cross-validated log loss (5-fold; lower is better)", "",
              "| Variant | CV log loss | Gain vs equal | 95% CI | p |", "|---|---|---|---|---|"]
    for name, l in cv.items():
        bs = paired_bootstrap(list(base - l), seed=seed)
        lines.append(f"| {name} | {l.mean():.4f} | {bs['mean']:+.4f} | [{bs['lo']:+.4f}, {bs['hi']:+.4f}] | {bs['p']:.3f} |")

    # 2. final fit on all data
    w, d = fit(items, k, floor, True, True)
    lines += ["", "## Fitted on all questions", "",
              "| Member | Weight |", "|---|---|"] + [f"| {m} | {wi:.3f} |" for m, wi in zip(members, w)]
    lines += ["", f"Extremizing factor: {d:.3f}"]

    # 3. correlations of member log-odds (binary)
    bin_x = np.array([x for kind, x, _ in items if kind == "binary"])
    if len(bin_x) >= 5:
        corr = np.corrcoef(bin_x.T)
        lines += ["", "## Correlation of members' log-odds (binary questions)", "",
                  "| | " + " | ".join(members) + " |", "|---" * (k + 1) + "|"]
        for i, m in enumerate(members):
            lines.append(f"| {m} | " + " | ".join(f"{corr[i, j]:.2f}" for j in range(k)) + " |")
        indiv = [item_losses([(kd, x, y) for kd, x, y in items], np.eye(k)[i], 1.0).mean() for i in range(k)]
        lines += ["", "Individual log loss: " + ", ".join(f"{m} {l:.4f}" for m, l in zip(members, indiv))]

    # 4. leave-one-out harm
    full = cv["fitted weights + extremizing"]
    lines += ["", "## Leave-one-out (CV log loss when the member is removed)", "",
              "| Removed | CV log loss | Harm of removing | Harm % | 95% CI | p |", "|---|---|---|---|---|---|"]
    for i, m in enumerate(members):
        sub = [(kd, np.delete(x, i, axis=0), y) for kd, x, y in items]
        l = cv_losses(sub, k - 1, floor, True, True, seed=seed)
        bs = paired_bootstrap(list(l - full), seed=seed)
        lines.append(f"| {m} | {l.mean():.4f} | {bs['mean']:+.4f} | {100 * bs['mean'] / full.mean():+.1f}% | "
                     f"[{bs['lo']:+.4f}, {bs['hi']:+.4f}] | {bs['p']:.3f} |")
    lines += ["", "Positive harm = the ensemble gets worse without this member (it adds real information)."]

    # 5. config snippet: each component must earn its place out-of-sample
    alpha = cfg.get_path("eval.significance", 0.05)
    w_gain = paired_bootstrap(list(cv["equal weights + extremizing"] - cv["fitted weights + extremizing"]), seed=seed)
    d_gain = paired_bootstrap(list(cv["equal weights, no extremizing"] - cv["equal weights + extremizing"]), seed=seed)
    use_w = w_gain["p"] < alpha and w_gain["mean"] > 0
    use_d = d_gain["p"] < alpha and d_gain["mean"] > 0
    w_final, d_final = (w, d) if use_w and use_d else fit(items, k, floor, use_d, use_w)
    snippet = {"aggregation": {"binary": "weighted_logodds", "multiple_choice": "weighted_logodds",
                               "numeric": "weighted_quantile_mean", "extremize": round(d_final, 3),
                               "weights": {m: round(float(wi), 3) for m, wi in zip(members, w_final)} if use_w else {},
                               "min_weight": floor}}
    lines += ["", "## Suggested config", "",
              f"- Fitted weights vs equal (both extremized): gain {w_gain['mean']:+.4f}, p={w_gain['p']:.3f} -> "
              f"{'USE fitted weights' if use_w else 'keep EQUAL weights'}",
              f"- Extremizing vs none (equal weights): gain {d_gain['mean']:+.4f}, p={d_gain['p']:.3f} -> "
              f"{'USE extremize=' + format(d_final, '.2f') if use_d else 'keep extremize=1.0'}"]
    if use_d and d_final >= 2.99:
        lines.append("- Note: the extremizing factor hit the search limit (3.0); treat it with caution.")
    lines += ["", "```yaml", yaml.safe_dump(snippet, sort_keys=False).strip(), "```"]
    if continuous:
        lines += ["", f"(Numeric/date questions: {len(continuous)}; the same member weights are applied to "
                      "weighted quantile averaging.)"]
    return "\n".join(lines)
