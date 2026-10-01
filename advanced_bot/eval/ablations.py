"""
Section 4 (ablations) and section 5 (research tool sets).

Each ablation compares the config WITHOUT a feature (A) to the config WITH it (B)
on the same frozen research, using the headline metric of the question group the
feature affects. Verdicts:

  KEEP     B is significantly better (bootstrap p < alpha)
  DROP     no significant gain -> remove it and save the cost
  HARMFUL  A is significantly better

`--out` writes the report; a recommended config (default + all DROP/HARMFUL
removals) is written to advanced_bot/configs/recommended.yaml for you to review.
Nothing in default.yaml is changed automatically.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from advanced_bot.eval.compare import compare

# (feature, config without, config with, group, metric, change to apply when dropping)
ABLATIONS = [
    ("Critique loop (on vs off)", "no_critic", "default", "all", "baseline", {"critique": {"enabled": False}}),
    ("Critique rounds (2 vs 1)", "critic_1round", "default", "all", "baseline", {"critique": {"rounds": 1}}),
    ("Structured (a)-(h) template vs simple prompt", "simple_prompt", "default", "all", "baseline",
     {"prompt_style": "simple"}),
    ("3 forecasters vs 1", "forecasters_1", "default", "all", "baseline", {"ensemble": {"total_forecasters": 1}}),
    ("5 forecasters vs 3", "default", "forecasters_5", "all", "baseline", None),
    ("Mixture tool for numeric/date", "default", "mixture_numeric", "continuous", "baseline", None),
    ("Time-to-event mixture for binary", "default", "mixture_time_to_event", "binary", "baseline", None),
]
# when "with" is an ablation config, KEEP means adopting it:
ADOPT = {
    "forecasters_5": {"ensemble": {"total_forecasters": 5}},
    "mixture_numeric": {"mixture": {"numeric": True}},
    "mixture_time_to_event": {"mixture": {"binary_time_to_event": True}},
}


def _row(res: dict, group: str, metric: str) -> dict | None:
    for r in res["rows"]:
        if r["group"] == group and r["metric"] == metric:
            return r
    return None


def _verdict(r: dict | None) -> str:
    if r is None:
        return "NO DATA"
    if not r["significant"]:
        return "DROP"
    return "KEEP" if r["b_better_by"] > 0 else "HARMFUL"


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def run_ablations(args, load, ensure_run) -> None:
    rows, overrides = [], {}
    names = {n for _, a, b, *_ in ABLATIONS for n in (a, b)}
    cfgs = {n: load(n, args) for n in names}
    for name in sorted(names):  # run everything first (stored runs are shared where possible)
        if not ensure_run(cfgs[name], args):
            return
    alpha = cfgs["default"].get_path("eval.significance", 0.05)
    samples = cfgs["default"].get_path("eval.bootstrap_samples", 10000)
    for feature, without, with_, group, metric, drop_change in ABLATIONS:
        res = compare(cfgs[without], cfgs[with_], samples=samples, alpha=alpha)
        r = _row(res, group, metric)
        verdict = _verdict(r)
        rows.append((feature, without, with_, group, r, verdict, res["cost_a"], res["cost_b"]))
        if with_ == "default" and verdict in ("DROP", "HARMFUL") and drop_change:
            overrides = _merge(overrides, drop_change)
        if with_ in ADOPT and verdict == "KEEP":
            overrides = _merge(overrides, ADOPT[with_])

    lines = ["# Ablation report", "",
             f"Metric: Metaculus baseline score (higher is better), paired bootstrap {100 * (1 - alpha):.0f}% CI.",
             "Gain = (with feature) - (without feature).", "",
             "| Feature | Group | n | Without | With | Gain | 95% CI | p | Verdict | Cost without | Cost with |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for feature, without, with_, group, r, verdict, ca, cb in rows:
        if r is None:
            lines.append(f"| {feature} | {group} | 0 | | | | | | NO DATA | | |")
            continue
        lines.append(f"| {feature} | {group} | {r['n']} | {r['a']:.2f} | {r['b']:.2f} | {r['b_better_by']:+.2f} | "
                     f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] | {r['p']:.3f} | **{verdict}** | ${ca:.2f} | ${cb:.2f} |")
    lines += ["", "KEEP = significant gain. DROP = no significant gain (remove to save cost). "
                  "HARMFUL = significantly worse.",
              "With few questions most results are DROP simply for lack of power; 200+ questions recommended."]
    report = "\n".join(lines)
    print("\n" + report)

    out = Path(args.out or Path(cfgs["default"].get_path("eval.runs_dir")) / "ablation_report.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    rec = {"extends": "default.yaml", "name": "recommended", **overrides}
    rec_path = out.parent / "recommended.yaml"
    rec_path.write_text("# Written by `python eval.py ablate`. Review it, then copy it to advanced_bot/configs/\n"
                        "# and use:  python run_advanced.py --config recommended ...\n"
                        + yaml.safe_dump(rec, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"\nReport: {out}\nRecommended config (review it): {rec_path}\n  changes vs default: {overrides or 'none'}")


def compare_research_sets(args, load, ensure_run) -> None:
    """Section 5: AskNews only vs FutureSearch vs both (each with its own frozen research)."""
    import asyncio

    from advanced_bot.eval.data import build_research

    names = ["research_asknews_only", "research_futuresearch", "research_asknews_futuresearch"]
    cfgs = {n: load(n, args) for n in names}
    for n, cfg in cfgs.items():
        print(f"Research set {cfg.get_path('research.set_name')}:")
        stats = asyncio.run(build_research(cfg, limit=args.limit, concurrency=2, seed=args.seed))
        print(f"  new {stats['done']}, cached {stats['skipped']}, leaked {stats['leaked']}, failed {stats['failed']}")
        if not ensure_run(cfg, args):
            return
    base = cfgs["research_asknews_only"]
    alpha = base.get_path("eval.significance", 0.05)
    from advanced_bot.eval.compare import format_compare

    for other in names[1:]:
        res = compare(base, cfgs[other], samples=base.get_path("eval.bootstrap_samples", 10000), alpha=alpha)
        print("\n" + format_compare(res, alpha))
    print("\nNote: FutureSearch has no date filter unless futuresearch.supports_date_filter is true, so its "
          "leak rate (above) matters; leaked questions are excluded per research set and comparisons use "
          "only questions usable in both sets.")
