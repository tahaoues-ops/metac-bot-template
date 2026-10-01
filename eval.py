"""
Evaluation environment for the advanced bot (back-testing on resolved questions).

  python eval.py collect --n 300                       # 1. resolved questions from past bot tournaments
  python eval.py research --config default             # 2. frozen research (cut off at open date) + leak check
  python eval.py run --config default                  # 3. forecasts for one config (resumable)
  python eval.py score --config default                # scores + calibration
  python eval.py --config A --config B                 # compare two configs (runs what is missing)
  python eval.py ablate                                # section 4: all ablations vs the baseline
  python eval.py ensemble --config ensemble            # section 2: fit weights / extremizing, correlations, LOO
  python eval.py research-sets                         # section 5: compare research tool sets
  python eval.py export-rl --config default            # section 6: JSONL for later RL training

Offline demo (no keys, no cost):  add --fake --data-dir eval_demo, after `python eval.py demo-data`.
Nothing here ever publishes to Metaculus.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

import dotenv

dotenv.load_dotenv()

from bot_helpers import silence_noisy_dependencies  # noqa: E402

silence_noisy_dependencies()

SUBCOMMANDS = ["collect", "demo-data", "research", "run", "score", "compare", "ablate", "ensemble",
               "research-sets", "export-rl"]


def _load(name, args):
    from advanced_bot.config import load_config

    cfg = load_config(name)
    if args.data_dir:
        cfg["eval"]["data_dir"] = args.data_dir
    if args.runs_dir:
        cfg["eval"]["runs_dir"] = args.runs_dir
    return cfg


def _confirm_spend(cfg, limit, args) -> bool:
    from advanced_bot.eval.data import pending

    work = pending(cfg, limit)
    slots = sum(len(m) for _, _, m in work)
    if not slots:
        return True
    rounds = cfg.get_path("critique.rounds") if cfg.get_path("critique.enabled") else 0
    calls = slots * (1 + 2 * rounds)
    msg = (f"Config '{cfg.name}': {len(work)} questions need {slots} forecaster runs "
           f"(~{calls} LLM calls).")
    if args.fake or args.yes:
        print(msg)
        return True
    if not sys.stdin.isatty():
        print(msg + " Re-run with --yes to spend the money (non-interactive).")
        return False
    return input(msg + " Continue? [y/N] ").strip().lower() == "y"


def ensure_run(cfg, args) -> bool:
    from advanced_bot.eval.data import run_config

    if args.no_run:
        return True
    if not _confirm_spend(cfg, args.limit, args):
        return False
    stats = asyncio.run(run_config(cfg, limit=args.limit, concurrency=args.concurrency, seed=args.seed))
    if stats["questions"]:
        print(f"  ran {stats['forecasters_run']} forecasters on {stats['questions']} questions "
              f"(failed {stats['failed']}, cost ${stats['cost']:.2f})")
    return True


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] not in SUBCOMMANDS and argv[0] not in ("-h", "--help"):
        argv = ["compare"] + argv  # `python eval.py --config A --config B`

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=SUBCOMMANDS)
    p.add_argument("--config", action="append", default=[], help="config name/path (repeat for compare)")
    p.add_argument("--n", type=int, default=300, help="collect: number of questions (200-500 recommended)")
    p.add_argument("--tournament", action="append", default=[], help="collect: tournament id/slug (repeatable)")
    p.add_argument("--limit", type=int, default=None, help="only the first N questions")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--runs-dir", default=None)
    p.add_argument("--fake", action="store_true", help="offline fake backend (testing only)")
    p.add_argument("--yes", action="store_true", help="don't ask before spending on LLM calls")
    p.add_argument("--no-run", action="store_true", help="only score what is already stored")
    p.add_argument("--out", default=None, help="output file (export-rl / reports)")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if args.fake:
        os.environ["ADV_BOT_FAKE_LLM"] = "1"
    cmd = args.command
    configs = args.config or ["default"]

    if cmd == "collect":
        from advanced_bot.eval.data import collect_questions

        cfg = _load(configs[0], args)
        qs = asyncio.run(collect_questions(cfg, args.n, args.tournament or None, seed=args.seed))
        kinds = {}
        for q in qs:
            kinds[type(q).__name__] = kinds.get(type(q).__name__, 0) + 1
        print(f"Saved {len(qs)} questions: {kinds}")

    elif cmd == "demo-data":
        from advanced_bot.eval.data import save_questions
        from advanced_bot.testing.fixtures import make_questions

        cfg = _load(configs[0], args)
        qs = make_questions(args.n if args.n != 300 else 40, seed=args.seed)
        save_questions(cfg, qs)
        print(f"Saved {len(qs)} SYNTHETIC questions to {cfg.get_path('eval.data_dir')}/ (for --fake testing only)")

    elif cmd == "research":
        from advanced_bot.eval.data import build_research

        from advanced_bot.eval.data import load_questions, research_dir

        for name in configs:
            cfg = _load(name, args)
            todo = [qv for qv in load_questions(cfg)[: args.limit or None]
                    if not (research_dir(cfg) / f"{qv.key}.json").exists()]
            msg = (f"Research set '{cfg.get_path('research.set_name')}': {len(todo)} questions need research "
                   f"(~{cfg.get_path('research.max_iterations')} search rounds + 3 LLM calls each).")
            if todo and not (args.fake or args.yes):
                if not sys.stdin.isatty():
                    print(msg + " Re-run with --yes to spend the money (non-interactive).")
                    continue
                if input(msg + " Continue? [y/N] ").strip().lower() != "y":
                    continue
            print(f"Research set '{cfg.get_path('research.set_name')}' (config {cfg.name}):")
            stats = asyncio.run(build_research(cfg, limit=args.limit, concurrency=min(args.concurrency, 2),
                                               seed=args.seed))
            print(f"  new {stats['done']}, cached {stats['skipped']}, leaked+excluded {stats['leaked']}, "
                  f"failed {stats['failed']}, cost ${stats['cost']:.2f}")

    elif cmd == "run":
        for name in configs:
            ensure_run(_load(name, args), args)

    elif cmd == "score":
        from advanced_bot.eval.compare import calibration_report, save_calibration_png, score_run, summary_rows

        for name in configs:
            cfg = _load(name, args)
            if not ensure_run(cfg, args):
                return
            rows = summary_rows(score_run(cfg))
            print(f"\nScores for config '{cfg.name}' (baseline: higher is better; others: lower is better)")
            for r in rows:
                print("  " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
            print("\n" + calibration_report(cfg))
            png = save_calibration_png(cfg, __import__("pathlib").Path(cfg.get_path("eval.runs_dir")) /
                                       f"calibration_{cfg.name}.png")
            if png:
                print(f"\nCalibration plot: {png}")

    elif cmd == "compare":
        from advanced_bot.eval.compare import compare, format_compare

        if len(configs) != 2:
            sys.exit("compare needs exactly two --config")
        a, b = _load(configs[0], args), _load(configs[1], args)
        if not (ensure_run(a, args) and ensure_run(b, args)):
            return
        alpha = a.get_path("eval.significance", 0.05)
        res = compare(a, b, samples=a.get_path("eval.bootstrap_samples", 10000), alpha=alpha)
        print("\n" + format_compare(res, alpha))

    elif cmd == "ablate":
        from advanced_bot.eval.ablations import run_ablations

        run_ablations(args, _load, ensure_run)

    elif cmd == "ensemble":
        from advanced_bot.eval.ensemble import ensemble_report

        cfg = _load(configs[0] if args.config else "ensemble", args)
        if not ensure_run(cfg, args):
            return
        print(ensemble_report(cfg, seed=args.seed))

    elif cmd == "research-sets":
        from advanced_bot.eval.ablations import compare_research_sets

        compare_research_sets(args, _load, ensure_run)

    elif cmd == "export-rl":
        from advanced_bot.eval.export import export_rl

        for name in configs:
            cfg = _load(name, args)
            path, n = export_rl(cfg, args.out)
            print(f"Wrote {n} records to {path}")


if __name__ == "__main__":
    main()
