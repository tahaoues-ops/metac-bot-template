"""
Section 6: export resolved-question data for later RL training (no training here).

One JSONL line per forecaster call that produced a prediction (draft, critique
responses, clarifications), with the frozen research, the exact prompt messages,
the model output, the parsed prediction and the actual outcome. The reward is
the Brier-based reward 1 - Brier (binary) or 1 - Brier/2 (multiple choice), in
[0, 1]; numeric questions get 1 - normalized CRPS.
"""
from __future__ import annotations

import json
from pathlib import Path

from advanced_bot.checks import FormatError, extract_json
from advanced_bot.config import Config
from advanced_bot.eval.data import load_questions, load_research, load_run
from advanced_bot.eval.metrics import score_question
from advanced_bot.forecasters import to_prediction
from advanced_bot.llm import RunContext
from datetime import datetime


def reward(kind: str, scores: dict) -> float:
    if kind == "binary":
        return 1 - scores["brier"]
    if kind == "multiple_choice":
        return 1 - scores["brier"] / 2
    return max(0.0, 1 - scores["crps"])


def export_rl(cfg: Config, out: str | None = None) -> tuple[Path, int]:
    path = Path(out or Path(cfg.get_path("eval.runs_dir")) / f"rl_dataset_{cfg.name}.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    run = load_run(cfg)
    views = {qv.key: qv for qv in load_questions(cfg)}
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for key, rec in run.items():
            qv = views.get(key)
            outcome = qv.outcome() if qv else None
            research = load_research(cfg, qv) if qv else None
            if outcome is None or research is None:
                continue
            ctx = RunContext(cutoff=datetime.fromisoformat(research["cutoff"]), backtest=True)
            for call in rec.get("calls", []):
                if not call["role"].startswith("forecaster"):
                    continue
                try:
                    pred = to_prediction(ctx, cfg, qv, extract_json(call["output"]))
                except (FormatError, Exception):
                    continue  # unparsable outputs are not training targets
                scores = score_question(qv, pred, outcome)
                f.write(json.dumps({
                    "question_key": key,
                    "question_type": qv.kind,
                    "question": qv.q.question_text,
                    "cutoff": research["cutoff"],
                    "research_report": research["research"]["report"],
                    "role": call["role"],
                    "model": call["model"],
                    "prompt_messages": call["messages"],
                    "output": call["output"],
                    "prediction": pred,
                    "outcome": outcome,
                    "scores": scores,
                    "reward": reward(qv.kind, scores),
                    "config": cfg.name,
                }, ensure_ascii=False, default=str) + "\n")
                n += 1
    return path, n
