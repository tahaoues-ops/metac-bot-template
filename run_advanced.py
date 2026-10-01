"""
Run the ADVANCED bot (advanced_bot/). The simple bots (main.py, geo_forecaster.py) are unchanged.

Examples:
  # one custom yes/no question (never published)
  python run_advanced.py --mode custom --question "Will X happen by 2026-12-31?" --resolution-date 2026-12-31

  # specific Metaculus questions, dry run (default: nothing is published)
  python run_advanced.py --mode urls --url https://www.metaculus.com/questions/12345/

  # open tournament questions, dry run
  python run_advanced.py --mode tournament

  # try everything offline with the fake backend (no keys, no cost)
  python run_advanced.py --mode custom --question "Test?" --resolution-date 2026-12-31 --fake

Publishing requires BOTH `publish_to_metaculus: true` in the config AND --publish.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timezone

import dotenv

dotenv.load_dotenv()

from bot_helpers import silence_noisy_dependencies  # noqa: E402

silence_noisy_dependencies()

from forecasting_tools import BinaryQuestion, MetaculusClient  # noqa: E402

from advanced_bot.bot import AdvancedBot  # noqa: E402
from advanced_bot.config import load_config  # noqa: E402

TOURNAMENT_URLS = {
    "tournament": "https://www.metaculus.com/tournament/summer-futureeval-2026/",
    "test_questions": "https://www.metaculus.com/tournament/bot-testing-area/",
}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    p = argparse.ArgumentParser(description="Advanced forecasting bot")
    p.add_argument("--mode", choices=["tournament", "test_questions", "urls", "custom"], default="custom")
    p.add_argument("--config", default=None, help="config name or path (default: advanced_bot/configs/default.yaml)")
    p.add_argument("--url", action="append", default=[], help="Metaculus question URL (repeatable)")
    p.add_argument("--question-file", help="custom mode: JSON file like questions/example_question.json")
    p.add_argument("--question", help="custom mode: yes/no question text")
    p.add_argument("--resolution-date", help="custom mode: YYYY-MM-DD")
    p.add_argument("--resolution-criteria", default="", help="custom mode: what counts as YES")
    p.add_argument("--background", default="")
    p.add_argument("--publish", action="store_true", help="post to Metaculus (also needs publish_to_metaculus: true)")
    p.add_argument("--max-concurrent", type=int, default=2)
    p.add_argument("--limit", type=int, default=None, help="forecast at most N questions (keeps test runs cheap)")
    p.add_argument("--fake", action="store_true", help="offline fake LLM/tools (testing only)")
    args = p.parse_args()

    if args.fake:
        os.environ["ADV_BOT_FAKE_LLM"] = "1"
    cfg = load_config(args.config)
    publish = bool(args.publish and cfg.get("publish_to_metaculus"))
    if args.publish and not publish:
        print("⚠️  --publish ignored: set publish_to_metaculus: true in the config to allow publishing.")
    if args.mode == "custom" and publish:
        print("⚠️  Custom questions are never published.")
        publish = False
    if publish and not os.getenv("METACULUS_TOKEN"):
        sys.exit("❌ Publishing needs METACULUS_TOKEN.")
    print(f"🤖 Advanced bot | config={cfg.name} | mode={args.mode} | publish={'YES' if publish else 'no (dry run)'}\n")

    bot = AdvancedBot(cfg, publish=publish, max_concurrent_questions=args.max_concurrent,
                      skip_previously_forecasted_questions=publish and args.mode == "tournament")
    client = MetaculusClient()

    if args.mode == "custom":
        if args.question_file:
            import json

            with open(args.question_file, encoding="utf-8") as f:
                spec = json.load(f)
            args.question = args.question or spec.get("question")
            args.resolution_date = args.resolution_date or spec.get("resolution_date")
            args.resolution_criteria = args.resolution_criteria or spec.get("resolution_criteria", "")
            args.background = args.background or spec.get("background", "")
        if not (args.question and args.resolution_date):
            sys.exit("❌ custom mode needs --question and --resolution-date")
        resolves = datetime.fromisoformat(args.resolution_date).replace(tzinfo=timezone.utc)
        if resolves <= datetime.now(timezone.utc):
            sys.exit("❌ resolution date must be in the future")
        q = BinaryQuestion(
            question_text=args.question,
            resolution_criteria=args.resolution_criteria or
            "Resolves YES if the event clearly happens, as reported by major reputable outlets, by the resolution date.",
            background_info=args.background, fine_print="",
            scheduled_resolution_time=resolves, close_time=resolves, id_of_post=0,
        )
        reports = asyncio.run(bot.forecast_questions([q], return_exceptions=True))
    elif args.mode == "urls":
        if not args.url:
            sys.exit("❌ urls mode needs at least one --url")
        questions = [client.get_question_by_url(u) for u in args.url]
        reports = asyncio.run(bot.forecast_questions(questions, return_exceptions=True))
    else:
        ids = ["bot-testing-area"] if args.mode == "test_questions" else [client.CURRENT_AI_COMPETITION_ID,
                                                                          client.CURRENT_MINIBENCH_ID]
        questions = [q for t in ids for q in client.get_all_open_questions_from_tournament(t)]
        if bot.skip_previously_forecasted_questions:
            questions = [q for q in questions if not q.already_forecasted]
        if args.limit:
            questions = questions[: args.limit]
        print(f"Forecasting {len(questions)} question(s).")
        reports = asyncio.run(bot.forecast_questions(questions, return_exceptions=True))

    ok = [r for r in reports if not isinstance(r, BaseException)]
    bad = [r for r in reports if isinstance(r, BaseException)]
    print("\n" + "=" * 80)
    print(f"{'Published' if publish else 'Produced (dry run)'}: {len(ok)} | failed: {len(bad)}")
    for r in ok:
        print(f"  ✅ {r.question.page_url or r.question.question_text[:70]} -> "
              f"{type(r).make_readable_prediction(r.prediction)}")
    for e in bad:
        print(f"  ❌ {type(e).__name__}: {str(e)[:300]}")
    print(f"Logs: {cfg.get_path('output.log_dir')}/")
    print("=" * 80)
    if bad and not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
