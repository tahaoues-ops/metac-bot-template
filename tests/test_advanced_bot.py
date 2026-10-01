"""
Offline tests for advanced_bot (no network, no keys): python -m unittest discover tests -v
Uses the fake LLM backend (ADV_BOT_FAKE_LLM=1) and synthetic questions.
"""
import asyncio
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

os.environ["ADV_BOT_FAKE_LLM"] = "1"

from forecasting_tools import NumericQuestion  # noqa: E402

from advanced_bot import aggregation, checks, mixture  # noqa: E402
from advanced_bot.config import load_config  # noqa: E402
from advanced_bot.eval import metrics  # noqa: E402
from advanced_bot.llm import RunContext  # noqa: E402
from advanced_bot.question_view import PERCENTILES, QuestionView  # noqa: E402
from advanced_bot.testing.fixtures import make_questions  # noqa: E402

QS = make_questions(8)
BINARY, MC, NUMERIC, DATE = QuestionView(QS[0]), QuestionView(QS[2]), QuestionView(QS[3]), QuestionView(QS[7])
OUT = {"min_probability": 0.01, "max_probability": 0.99, "mc_min_probability": 0.01}


def nq(lo, hi, open_lo, open_hi):
    return QuestionView(NumericQuestion(question_text="x", lower_bound=lo, upper_bound=hi,
                                        open_lower_bound=open_lo, open_upper_bound=open_hi))


class TestConfig(unittest.TestCase):
    def test_inheritance_and_names(self):
        c = load_config("no_critic")
        self.assertEqual(c.name, "no_critic")
        self.assertFalse(c.get_path("critique.enabled"))
        self.assertEqual(c.get_path("critique.rounds"), 2)  # inherited

    def test_forecaster_fingerprint_sharing(self):
        d = load_config()
        self.assertEqual(d.forecaster_fingerprint(), load_config("forecasters_5").forecaster_fingerprint())
        self.assertEqual(d.forecaster_fingerprint(), load_config("forecasters_1").forecaster_fingerprint())
        self.assertNotEqual(d.forecaster_fingerprint(), load_config("no_critic").forecaster_fingerprint())

    def test_publish_off_by_default(self):
        self.assertFalse(load_config().get("publish_to_metaculus"))


class TestFormatValidation(unittest.TestCase):
    def test_binary(self):
        self.assertAlmostEqual(checks.parse_and_validate(BINARY, {"probability": 37}, OUT)["p"], 0.37)
        self.assertEqual(checks.parse_and_validate(BINARY, {"probability": 0}, OUT)["p"], 0.01)
        with self.assertRaises(checks.FormatError):
            checks.parse_and_validate(BINARY, {"probability": 140}, OUT)

    def test_multiple_choice(self):
        ok = checks.parse_and_validate(MC, {"probabilities": {"alpha": 70, "Beta": 29.5, "Gamma": 0.5, "Delta": 0}}, OUT)
        self.assertAlmostEqual(sum(ok["probs"].values()), 1.0)
        self.assertTrue(all(v >= 0.01 - 1e-12 for v in ok["probs"].values()))
        with self.assertRaises(checks.FormatError):  # sum far from 100
            checks.parse_and_validate(MC, {"probabilities": {"Alpha": 50, "Beta": 10, "Gamma": 10, "Delta": 10}}, OUT)
        with self.assertRaises(checks.FormatError):  # missing option
            checks.parse_and_validate(MC, {"probabilities": {"Alpha": 50, "Beta": 50}}, OUT)

    def test_numeric_order_and_bounds(self):
        good = {f"{p:g}": 100 + i * 10 for i, p in enumerate(PERCENTILES)}
        pred = checks.parse_and_validate(NUMERIC, {"percentiles": good}, OUT)
        self.assertEqual(len(pred["percentiles"]), 15)
        bad_order = dict(good, **{"50": 1})
        with self.assertRaises(checks.FormatError):
            checks.parse_and_validate(NUMERIC, {"percentiles": bad_order}, OUT)
        below_closed = dict(good, **{"0.1": -5})  # lower bound 0 is closed
        with self.assertRaises(checks.FormatError):
            checks.parse_and_validate(NUMERIC, {"percentiles": below_closed}, OUT)
        missing = {k: v for k, v in good.items() if k != "99.9"}
        with self.assertRaises(checks.FormatError):
            checks.parse_and_validate(NUMERIC, {"percentiles": missing}, OUT)

    def test_repair_strictly_increasing_inside_closed_bounds(self):
        qv = nq(0, 100, False, False)
        vals = checks.repair_percentiles(qv, [100] * 15)
        self.assertTrue(all(b > a for a, b in zip(vals, vals[1:])))
        self.assertLessEqual(vals[-1], 100)

    def test_extract_json_takes_last_block(self):
        text = 'first {"probability": 10}\n```json\n{"probability": 55}\n```'
        self.assertEqual(checks.extract_json(text)["probability"], 55)


class TestArithmeticAndVerification(unittest.TestCase):
    def test_detects_wrong_and_accepts_right(self):
        wrong = "3 of the last 10 years (40%). 12 + 30 = 45. From 200 to 150, a 20% decrease."
        right = "3 of the last 10 years (30%). 12 + 30 = 42. From 200 to 150, a 25% decrease. 4 times in 20 years (20% per year)."
        self.assertEqual(len(checks.arithmetic_issues(wrong)), 3)
        self.assertEqual(checks.arithmetic_issues(right), [])

    def test_unverified_claims(self):
        research = "Talks in Geneva between Zelensky and Putin; 1,000 troops moved."
        text = "Zelensky met Erdogan in Istanbul; 25,400 soldiers moved; 1,000 troops; I estimate 35%."
        found = checks.unverified_claims(text, research, "question")
        names = " ".join(c for c in found if c.startswith("name"))
        figures = [c.split(" (")[0] for c in found if c.startswith("figure")]
        self.assertIn("Erdogan", names)
        self.assertNotIn("Zelensky", names)
        self.assertEqual(figures, ["figure 25,400"])  # 1,000 is in the research; 35% is an opinion

    def test_consistency(self):
        pred = {"p": 0.2}
        self.assertIsNotNone(checks.consistency_issue(BINARY, {"base_rate": 30, "evidence_direction": "up"}, pred, None))
        self.assertIsNone(checks.consistency_issue(BINARY, {"base_rate": 30, "evidence_direction": "down"}, pred, None))
        self.assertIsNotNone(checks.consistency_issue(BINARY, {"update_direction": "up"}, pred, {"p": 0.3}))


class TestAggregation(unittest.TestCase):
    def test_binary_methods(self):
        ps, w = [0.2, 0.5, 0.8], [1 / 3] * 3
        self.assertAlmostEqual(aggregation.aggregate_binary(ps, w, "median"), 0.5)
        self.assertAlmostEqual(aggregation.aggregate_binary(ps, w, "weighted_logodds"), 0.5)
        self.assertGreater(aggregation.aggregate_binary([0.7, 0.7], [0.5, 0.5], "weighted_logodds", 2.0), 0.8)

    def test_weights_floor(self):
        w = aggregation.forecaster_weights(["a", "b", "b"], {"a": 1.0, "b": 0.0}, 0.1)
        self.assertAlmostEqual(sum(w), 1.0)
        self.assertGreaterEqual(w[1] + w[2], 0.09)

    def test_numeric_quantile_mean(self):
        a = [[p, 10.0 + i] for i, p in enumerate(PERCENTILES)]
        b = [[p, 30.0 + i] for i, p in enumerate(PERCENTILES)]
        out = aggregation.aggregate(NUMERIC, [{"percentiles": a}, {"percentiles": b}], ["x", "y"],
                                    {"numeric": "weighted_quantile_mean"}, OUT)
        self.assertAlmostEqual(dict((p, v) for p, v in out["percentiles"])[50], 27.0)


class TestMixture(unittest.TestCase):
    def test_matches_scipy_and_truncates(self):
        from scipy import stats

        sc = mixture.parse_mixture([{"weight": 1, "distribution": "normal", "params": {"mean": 100, "sd": 50}}])
        v = mixture.mixture_percentiles(nq(-1000, 1000, True, True), sc, 0)
        for p, x in zip(PERCENTILES, v):
            self.assertAlmostEqual(x, stats.norm.ppf(p / 100, 100, 50), delta=0.5)
        v = mixture.mixture_percentiles(nq(0, 1000, False, True), sc, 0)
        self.assertGreaterEqual(min(v), 0)

    def test_time_to_event(self):
        p, _, _ = mixture.time_to_event_probability(
            {"p_never": 0.2, "scenarios": [{"weight": 1, "distribution": "lognormal",
                                            "params": {"median": 100, "sigma": 0.5}}]}, 100)
        self.assertAlmostEqual(p, 0.4, places=3)

    def test_rejects_bad(self):
        with self.assertRaises(checks.FormatError):
            mixture.parse_mixture([{"weight": 1, "distribution": "normal", "params": {"mean": 1, "sd": 0}}])


class TestMetrics(unittest.TestCase):
    def test_binary_and_mc(self):
        self.assertAlmostEqual(metrics.binary_scores(0.5, True)["baseline"], 0.0)
        self.assertAlmostEqual(metrics.binary_scores(0.8, False)["brier"], 0.64)
        self.assertAlmostEqual(metrics.mc_scores({"a": .25, "b": .25, "c": .25, "d": .25}, "a")["baseline"], 0.0)

    def test_continuous_good_beats_bad(self):
        y = NUMERIC.outcome()["value"]
        good = [[p, y - 300 + i * 40] for i, p in enumerate(PERCENTILES)]
        bad = [[p, 1 + i] for i, p in enumerate(PERCENTILES)]
        g, b = metrics.continuous_scores(NUMERIC, good, y), metrics.continuous_scores(NUMERIC, bad, y)
        self.assertGreater(g["baseline"], b["baseline"])
        self.assertLess(g["crps"], b["crps"])
        self.assertLess(g["pinball"], b["pinball"])

    def test_bootstrap(self):
        r = metrics.paired_bootstrap([0.1] * 30 + [0.2] * 30)
        self.assertLess(r["p"], 0.05)
        self.assertGreater(r["lo"], 0)


class TestResearchAndPipeline(unittest.TestCase):
    def test_research_loop_limits_and_cleaning(self):
        from advanced_bot.researcher import run_research

        cfg = load_config()
        cfg["research"]["min_iterations"] = 5
        cfg["research"]["max_iterations"] = 5
        ctx = RunContext(cutoff=BINARY.q.open_time, backtest=True)
        r = asyncio.run(run_research(ctx, cfg, BINARY))
        self.assertEqual(r.iterations, 5)
        self.assertNotIn("<think>", r.report)
        self.assertNotIn("[99]", r.report)
        self.assertIn("## Sources", r.report)
        self.assertTrue(r.window[1].startswith(BINARY.q.open_time.date().isoformat()))

    def test_tools_never_return_post_cutoff(self):
        from advanced_bot import tools
        from advanced_bot.testing import fake

        cutoff = datetime(2025, 1, 1, tzinfo=timezone.utc)
        orig = fake.fake_search

        def leaky(ctx, key, query, w):
            items = orig(ctx, key, query, w)
            items[0].published = cutoff + timedelta(days=30)
            return items

        fake.fake_search = leaky
        try:
            ctx = RunContext(cutoff=cutoff, backtest=True)
            w = tools.TimeWindow(start=cutoff - timedelta(days=90), end=cutoff)
            items = asyncio.run(tools.run_tool(ctx, load_config(), w, "search_news", {"query": "x"}))
        finally:
            fake.fake_search = orig
        self.assertTrue(all(i.published <= cutoff + timedelta(hours=1) for i in items))
        self.assertTrue(any("after cutoff" in x for x in ctx.warnings))

    def test_window_narrowing_cannot_extend(self):
        from advanced_bot.tools import TimeWindow

        end = datetime(2025, 1, 1, tzinfo=timezone.utc)
        w = TimeWindow(start=end - timedelta(days=30), end=end)
        self.assertEqual(w.narrowed(9999).start, w.start)
        self.assertEqual(w.narrowed(7).start, end - timedelta(days=7))

    def test_pipeline_all_types_valid_for_metaculus(self):
        from advanced_bot.pipeline import forecast_from_research, to_forecasting_tools
        from advanced_bot.researcher import run_research

        for cfg_name in ("default", "mixture_numeric", "mixture_time_to_event", "simple_prompt"):
            cfg = load_config(cfg_name)
            for qv in (BINARY, MC, NUMERIC, DATE):
                ctx = RunContext(cutoff=qv.q.open_time, backtest=True)
                r = asyncio.run(run_research(ctx, cfg, qv))
                fc = asyncio.run(forecast_from_research(ctx, cfg, qv, r))
                to_forecasting_tools(qv, fc.prediction)  # raises if Metaculus would reject it
                self.assertEqual(len(fc.members), 3, (cfg_name, qv.kind))


class TestEvalEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _cfg(self, name):
        c = load_config(name)
        c["eval"]["data_dir"] = os.path.join(self.tmp, "data")
        c["eval"]["runs_dir"] = os.path.join(self.tmp, "runs")
        return c

    def test_collect_research_run_compare(self):
        from advanced_bot.eval import data
        from advanced_bot.eval.compare import compare

        base = self._cfg("default")
        data.save_questions(base, make_questions(20))
        stats = asyncio.run(data.build_research(base))
        self.assertEqual(stats["leaked"], 2)  # synthetic keys ending in 5 leak on purpose
        self.assertEqual(len(data.usable_questions(base)), 18)

        asyncio.run(data.run_config(base))
        one = self._cfg("forecasters_1")
        self.assertEqual(data.pending(one), [])  # 1-forecaster run is contained in the 3-forecaster run
        res = compare(one, base, samples=500)
        self.assertEqual(res["n_common"], 18)
        self.assertTrue(any(r["group"] == "all" for r in res["rows"]))

    def test_export_rl(self):
        import json

        from advanced_bot.eval import data
        from advanced_bot.eval.export import export_rl

        cfg = self._cfg("no_critic")
        data.save_questions(cfg, make_questions(8))
        asyncio.run(data.build_research(cfg))
        asyncio.run(data.run_config(cfg))
        path, n = export_rl(cfg)
        self.assertGreater(n, 0)
        with open(path, encoding="utf-8") as f:
            rec = json.loads(f.readline())
        for key in ("research_report", "prompt_messages", "output", "prediction", "outcome", "reward"):
            self.assertIn(key, rec)
        self.assertTrue(0 <= rec["reward"] <= 1)


if __name__ == "__main__":
    unittest.main()
