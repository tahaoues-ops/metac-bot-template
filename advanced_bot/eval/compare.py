"""Scoring a config on the eval set and comparing configs with paired bootstrap CIs."""
from __future__ import annotations

from pathlib import Path

from advanced_bot.config import Config
from advanced_bot.eval.data import load_questions, load_run
from advanced_bot.eval.metrics import (
    HIGHER_IS_BETTER,
    calibration_pairs,
    calibration_table,
    paired_bootstrap,
    score_question,
)
from advanced_bot.question_view import QuestionView

TYPE_GROUPS = {"binary": "binary", "multiple_choice": "multiple_choice", "numeric": "continuous",
               "date": "continuous"}
GROUP_METRICS = {"all": ["baseline"], "binary": ["baseline", "brier", "log_loss"],
                 "multiple_choice": ["baseline", "brier", "log_loss"], "continuous": ["baseline", "crps", "pinball"]}


def score_run(cfg: Config, run: dict[str, dict] | None = None) -> dict[str, dict]:
    """key -> {"group", "scores", "prediction"}"""
    run = run if run is not None else load_run(cfg)
    views: dict[str, QuestionView] = {qv.key: qv for qv in load_questions(cfg)}
    out = {}
    for key, rec in run.items():
        qv = views.get(key)
        outcome = qv.outcome() if qv else None
        if outcome is None:
            continue
        out[key] = {"group": TYPE_GROUPS[qv.kind], "scores": score_question(qv, rec["prediction"], outcome),
                    "prediction": rec["prediction"], "qv": qv, "outcome": outcome, "cost": rec.get("cost", 0.0)}
    return out


def summary_rows(scored: dict[str, dict]) -> list[dict]:
    rows = []
    for group, metrics in GROUP_METRICS.items():
        items = [v for v in scored.values() if group == "all" or v["group"] == group]
        if not items:
            continue
        row = {"group": group, "n": len(items)}
        for m in metrics:
            row[m] = sum(v["scores"][m] for v in items) / len(items)
        rows.append(row)
    return rows


def compare(cfg_a: Config, cfg_b: Config, samples: int = 10000, alpha: float = 0.05) -> dict:
    sa, sb = score_run(cfg_a), score_run(cfg_b)
    common = sorted(set(sa) & set(sb))
    result = {"a": cfg_a.name, "b": cfg_b.name, "n_common": len(common), "rows": []}
    for group, metrics in GROUP_METRICS.items():
        keys = [k for k in common if group == "all" or sa[k]["group"] == group]
        if not keys:
            continue
        for m in metrics:
            # positive diff = B is better, whatever the metric's direction
            sign = 1 if HIGHER_IS_BETTER[m] else -1
            diffs = [sign * (sb[k]["scores"][m] - sa[k]["scores"][m]) for k in keys]
            bs = paired_bootstrap(diffs, samples=samples, alpha=alpha)
            result["rows"].append({
                "group": group, "metric": m, "n": len(keys),
                "a": sum(sa[k]["scores"][m] for k in keys) / len(keys),
                "b": sum(sb[k]["scores"][m] for k in keys) / len(keys),
                "b_better_by": bs["mean"], "ci_lo": bs["lo"], "ci_hi": bs["hi"], "p": bs["p"],
                "significant": bs["p"] < alpha,
            })
    result["cost_a"] = sum(sa[k]["cost"] for k in common)
    result["cost_b"] = sum(sb[k]["cost"] for k in common)
    return result


def format_compare(res: dict, alpha: float = 0.05) -> str:
    lines = [f"Comparison: A = {res['a']}  vs  B = {res['b']}   ({res['n_common']} common questions)",
             "positive 'B better by' = B is better (direction-adjusted). CI = "
             f"{100 * (1 - alpha):.0f}% paired bootstrap.",
             "",
             f"{'group':<16}{'metric':<10}{'n':>5}{'A':>10}{'B':>10}{'B better by':>13}{'CI':>22}{'p':>8}  verdict",
             "-" * 104]
    for r in res["rows"]:
        if r["significant"]:
            verdict = "B better ✔" if r["b_better_by"] > 0 else "A better ✔"
        else:
            verdict = "no significant difference"
        lines.append(f"{r['group']:<16}{r['metric']:<10}{r['n']:>5}{r['a']:>10.4f}{r['b']:>10.4f}"
                     f"{r['b_better_by']:>+13.4f}   [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]{r['p']:>8.3f}  {verdict}")
    lines.append(f"\nLLM cost on these questions: A ${res['cost_a']:.2f}, B ${res['cost_b']:.2f}")
    return "\n".join(lines)


def calibration_report(cfg: Config) -> str:
    scored = score_run(cfg)
    sections = []
    for label, kinds in (("binary questions", {"binary"}), ("multiple-choice options", {"multiple_choice"}),
                         ("numeric/date percentiles (PIT)", {"numeric", "date"})):
        pairs = []
        for v in scored.values():
            if v["qv"].kind in kinds:
                pairs += calibration_pairs(v["qv"], v["prediction"], v["outcome"])
        if not pairs:
            continue
        rows = calibration_table(pairs)
        lines = [f"Calibration — {label} ({len(pairs)} points)",
                 f"{'bin':<10}{'n':>6}{'predicted':>11}{'observed':>10}  chart (|=predicted, #=observed)"]
        for r in rows:
            bar = [" "] * 21
            bar[min(int(r["observed"] * 20), 20)] = "#"
            bar[min(int(r["mean_predicted"] * 20), 20)] = "|" if bar[min(int(r["mean_predicted"] * 20), 20)] == " " else "X"
            lines.append(f"{r['bin']:<10}{r['n']:>6}{r['mean_predicted']:>11.2f}{r['observed']:>10.2f}  [{''.join(bar)}]")
        sections.append("\n".join(lines))
    return "\n\n".join(sections) or "No scored questions yet."


def save_calibration_png(cfg: Config, path: Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    scored = score_run(cfg)
    pairs = []
    for v in scored.values():
        if v["qv"].kind in ("binary", "multiple_choice"):
            pairs += calibration_pairs(v["qv"], v["prediction"], v["outcome"])
    rows = calibration_table(pairs)
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999", linewidth=1, label="perfect")
    ax.plot([r["mean_predicted"] for r in rows], [r["observed"] for r in rows], marker="o", color="#2a6f97",
            label=cfg.name)
    for r in rows:
        ax.annotate(str(r["n"]), (r["mean_predicted"], r["observed"]), textcoords="offset points",
                    xytext=(4, -10), fontsize=7, color="#555")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title(f"Calibration — {cfg.name}")
    ax.legend(loc="upper left", frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path
