"""
Geopolitical forecaster for your own questions (manual, local results only).

What it does:
  1. Takes a yes/no question you write yourself plus its resolution date.
  2. Searches the web for recent news (OpenRouter web search) and collects sources.
  3. Asks the model to forecast several times independently and takes the median.
  4. Saves a JSON file and an Arabic Markdown report in `results/`.

It never talks to Metaculus and never publishes anything. The only key it
needs is OPENROUTER_API_KEY (read from `.env` locally or from GitHub Secrets).

Examples:
  python geo_forecaster.py --question-file questions/example_question.json
  python geo_forecaster.py --question "Will ...?" --resolution-date 2026-12-31
  python geo_forecaster.py --question-file questions/example_question.json --offline
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import dotenv
import requests

dotenv.load_dotenv()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Override any of these in .env (or GitHub Secrets/Variables) without editing code.
DEFAULT_MODEL = os.getenv("GEO_MODEL", "anthropic/claude-sonnet-4.5")
DEFAULT_RESEARCH_MODEL = os.getenv("GEO_RESEARCH_MODEL", DEFAULT_MODEL)
SEARCH_RESULTS_PER_QUERY = int(os.getenv("GEO_SEARCH_RESULTS", "8"))
REQUEST_TIMEOUT_SECONDS = 180

_PLACEHOLDERS = {"", "REPLACE_ME", "your-api-key-here"}


# ----------------------------------------------------------------------------- input


def load_question(args: argparse.Namespace) -> dict:
    question: dict = {}
    if args.question_file:
        with open(args.question_file, encoding="utf-8") as f:
            question = json.load(f)
    # Command-line values override the file.
    for key in ("question", "resolution_date", "resolution_criteria", "background"):
        value = getattr(args, key)
        if value:
            question[key] = value

    if not question.get("question"):
        sys.exit("❌ No question given. Use --question or --question-file.")
    if not question.get("resolution_date"):
        sys.exit("❌ No resolution date given. Use --resolution-date YYYY-MM-DD.")
    try:
        resolution = date.fromisoformat(question["resolution_date"])
    except ValueError:
        sys.exit("❌ Resolution date must look like 2026-12-31 (YYYY-MM-DD).")
    if resolution <= date.today():
        sys.exit("❌ Resolution date is today or in the past; nothing to forecast.")

    question.setdefault(
        "resolution_criteria",
        "Resolves YES if the event in the question clearly happens, as reported by "
        "major reputable news outlets, on or before the resolution date. Otherwise NO.",
    )
    question.setdefault("background", "")
    return question


# ----------------------------------------------------------------------------- LLM calls


def get_api_key() -> str:
    key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if key in _PLACEHOLDERS:
        sys.exit(
            "❌ OPENROUTER_API_KEY is missing.\n"
            "   Locally: put it in the .env file (see .env.template).\n"
            "   On GitHub: Settings → Secrets and variables → Actions → New repository secret.\n"
            "   To test without a key, add --offline."
        )
    return key


def call_openrouter(
    api_key: str,
    model: str,
    messages: list[dict],
    web_search: bool = False,
    temperature: float = 0.3,
) -> dict:
    payload: dict = {"model": model, "messages": messages, "temperature": temperature}
    if web_search:
        payload["plugins"] = [{"id": "web", "max_results": SEARCH_RESULTS_PER_QUERY}]
    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": "geo-forecaster",
        },
        json=payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    if response.status_code != 200:
        # Never echo request headers: they contain the key.
        raise RuntimeError(
            f"OpenRouter error {response.status_code}: {response.text[:500]}"
        )
    message = response.json()["choices"][0]["message"]
    return {
        "content": message.get("content") or "",
        "annotations": message.get("annotations") or [],
    }


# ----------------------------------------------------------------------------- research

RESEARCH_ANGLES = [
    (
        "latest developments",
        "Find the most recent news (prioritise the last 30 days) directly relevant "
        "to this question. Give concrete dated facts: who did/said what, when.",
    ),
    (
        "base rates and obstacles",
        "Find historical precedents and base rates for events like this, official "
        "positions of the key actors, scheduled dates (summits, votes, deadlines), "
        "and the main obstacles or reasons it might NOT happen.",
    ),
]


def research_prompt(question: dict, angle_instruction: str) -> str:
    return f"""You are a research assistant for a geopolitical forecaster. Today is {date.today().isoformat()}.

Question: {question['question']}
Resolution date: {question['resolution_date']}
Resolution criteria: {question['resolution_criteria']}
Background from the user: {question['background'] or 'none'}

Task: {angle_instruction}

Rules:
- Report facts, not a forecast.
- Put the publication date next to each fact when known.
- Cite the source of every fact.
- Clearly say if the information is old or if you found nothing recent.
- Write in English, as bullet points, max ~400 words."""


def run_research(api_key: str, model: str, question: dict) -> tuple[str, list[dict]]:
    sections: list[str] = []
    sources: dict[str, dict] = {}
    for title, instruction in RESEARCH_ANGLES:
        print(f"🔎 Searching: {title} ...")
        result = call_openrouter(
            api_key,
            model,
            [{"role": "user", "content": research_prompt(question, instruction)}],
            web_search=True,
            temperature=0.1,
        )
        sections.append(f"### {title}\n{result['content']}")
        for ann in result["annotations"]:
            cite = ann.get("url_citation") if ann.get("type") == "url_citation" else None
            if cite and cite.get("url") and cite["url"] not in sources:
                sources[cite["url"]] = {
                    "url": cite["url"],
                    "title": cite.get("title") or cite["url"],
                }
    numbered = [{"id": i + 1, **s} for i, s in enumerate(sources.values())]
    return "\n\n".join(sections), numbered


# ----------------------------------------------------------------------------- forecast


def forecast_prompt(question: dict, research: str, sources: list[dict]) -> str:
    source_list = "\n".join(f"[{s['id']}] {s['title']} — {s['url']}" for s in sources)
    return f"""You are an experienced, well-calibrated superforecaster specialising in geopolitics. Today is {date.today().isoformat()}.

Question: {question['question']}
Resolution date: {question['resolution_date']}
Resolution criteria: {question['resolution_criteria']}
Background from the user: {question['background'] or 'none'}

Research notes (from a web search done today):
{research}

Numbered sources:
{source_list or '(no sources returned)'}

Think step by step:
1. How much time is left until the resolution date?
2. What happens if nothing changes (status quo)? The world usually changes slowly.
3. What is the historical base rate for events like this?
4. Which recent evidence pushes the probability up, and which pushes it down?
5. Avoid overconfidence: never give 0 or 100.

Answer ONLY with one JSON object (no text before or after) with exactly these keys:
{{
  "probability": <integer 1-99, chance the question resolves YES>,
  "status_quo_ar": "<one sentence in Arabic: what happens if nothing changes>",
  "base_rate_ar": "<one or two sentences in Arabic about the historical base rate>",
  "evidence_for": [{{"point_ar": "<evidence in Arabic that raises the probability>", "source_ids": [<source numbers>]}}],
  "evidence_against": [{{"point_ar": "<evidence in Arabic that lowers the probability>", "source_ids": [<source numbers>]}}],
  "key_uncertainties_ar": ["<Arabic>", "..."],
  "what_would_change_my_mind_ar": ["<Arabic: a signal that would move the forecast a lot>", "..."],
  "explanation_ar": "<a clear explanation in simple Modern Standard Arabic, 150-250 words, for a non-expert reader, explaining how you reached the probability>"
}}
Only use source numbers that exist in the list above. Write all *_ar fields in Arabic."""


def parse_forecast_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Model did not return JSON")
    data = json.loads(match.group(0))
    data["probability"] = max(1, min(99, int(round(float(data["probability"])))))
    return data


def run_forecasts(
    api_key: str, model: str, question: dict, research: str, sources: list[dict], runs: int
) -> list[dict]:
    prompt = forecast_prompt(question, research, sources)
    forecasts = []
    for i in range(runs):
        print(f"🧠 Forecast {i + 1}/{runs} ...")
        try:
            result = call_openrouter(
                api_key, model, [{"role": "user", "content": prompt}], temperature=0.7
            )
            forecasts.append(parse_forecast_json(result["content"]))
        except (ValueError, KeyError, json.JSONDecodeError, RuntimeError) as e:
            print(f"   ⚠️ This attempt failed and was skipped: {e}")
    if not forecasts:
        sys.exit("❌ All forecast attempts failed. See the messages above.")
    return forecasts


# ----------------------------------------------------------------------------- offline demo


def offline_research(question: dict) -> tuple[str, list[dict]]:
    research = (
        "### offline demo\n- This is fake research used only to test the pipeline "
        "without an API key. It contains no real news."
    )
    sources = [{"id": 1, "title": "Example source (offline demo)", "url": "https://example.com"}]
    return research, sources


def offline_forecasts(runs: int) -> list[dict]:
    demo = {
        "status_quo_ar": "هذه تجربة دون اتصال؛ لا توجد بيانات حقيقية.",
        "base_rate_ar": "لا يوجد معدل أساسي حقيقي في وضع التجربة.",
        "evidence_for": [{"point_ar": "دليل تجريبي يرفع الاحتمال.", "source_ids": [1]}],
        "evidence_against": [{"point_ar": "دليل تجريبي يخفض الاحتمال.", "source_ids": [1]}],
        "key_uncertainties_ar": ["هذه نتيجة وهمية لاختبار البرنامج فقط."],
        "what_would_change_my_mind_ar": ["تشغيل البوت بمفتاح حقيقي."],
        "explanation_ar": "هذا تقرير تجريبي للتأكد من أن البرنامج يعمل ويحفظ الملفات. "
        "الأرقام هنا ليست توقعاً حقيقياً.",
    }
    return [{**demo, "probability": p} for p in [30, 35, 40, 25, 45][:runs]]


# ----------------------------------------------------------------------------- output


def aggregate(forecasts: list[dict]) -> tuple[int, dict]:
    probs = [f["probability"] for f in forecasts]
    median = int(round(statistics.median(probs)))
    # Use the reasoning of the run closest to the median as the main explanation.
    representative = min(forecasts, key=lambda f: abs(f["probability"] - median))
    return median, representative


def _cites(ids: list, valid_ids: set[int]) -> str:
    # Drop any source number the model invented that isn't in the list.
    ids = [i for i in ids or [] if i in valid_ids]
    return " " + "".join(f"[{i}]" for i in ids) if ids else ""


def build_markdown(record: dict) -> str:
    q = record["question"]
    rep = record["representative_forecast"]
    probs = ", ".join(f"{p}%" for p in record["individual_probabilities"])
    valid_ids = {s["id"] for s in record["sources"]}
    lines = [
        '<div dir="rtl">',
        "",
        f"# تقرير توقع: {q['question']}",
        "",
        f"- **الاحتمال النهائي (نعم):** {record['final_probability']}%",
        f"- **موعد الحسم:** {q['resolution_date']}",
        f"- **تاريخ التوقع:** {record['created_at']}",
        f"- **التوقعات الفردية:** {probs} (النهائي هو الوسيط)",
        f"- **النموذج:** {record['model']}",
        "- **النشر على Metaculus:** لا (حفظ محلي فقط)",
    ]
    if record.get("offline"):
        lines.append("- ⚠️ **تشغيل تجريبي دون اتصال — النتيجة وهمية**")
    lines += [
        "",
        "## معيار الحسم",
        q["resolution_criteria"],
        "",
        "## التفسير",
        rep.get("explanation_ar", ""),
        "",
        "## الوضع الراهن والمعدل الأساسي",
        f"- **إذا لم يتغير شيء:** {rep.get('status_quo_ar', '')}",
        f"- **المعدل التاريخي:** {rep.get('base_rate_ar', '')}",
        "",
        "## أدلة ترفع الاحتمال",
        *[f"- {e.get('point_ar', '')}{_cites(e.get('source_ids'), valid_ids)}" for e in rep.get("evidence_for", [])],
        "",
        "## أدلة تخفض الاحتمال",
        *[f"- {e.get('point_ar', '')}{_cites(e.get('source_ids'), valid_ids)}" for e in rep.get("evidence_against", [])],
        "",
        "## أهم نقاط عدم اليقين",
        *[f"- {u}" for u in rep.get("key_uncertainties_ar", [])],
        "",
        "## ما الذي قد يغيّر التوقع",
        *[f"- {u}" for u in rep.get("what_would_change_my_mind_ar", [])],
        "",
        "## المصادر",
        *[f"{s['id']}. [{s['title']}]({s['url']})" for s in record["sources"]],
        "",
        "</div>",
        "",
        "<details><summary>Raw research notes (English)</summary>",
        "",
        record["research"],
        "",
        "</details>",
        "",
    ]
    return "\n".join(lines)


def save_results(record: dict, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^a-z0-9]+", "-", record["question"]["question"].lower()).strip("-")[:50]
    base = output_dir / f"{stamp}_{slug or 'question'}"
    json_path = base.with_suffix(".json")
    md_path = base.with_suffix(".md")
    json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(build_markdown(record), encoding="utf-8")
    return json_path, md_path


# ----------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description="Forecast a custom geopolitical question")
    parser.add_argument("--question-file", help="JSON file with the question (see questions/)")
    parser.add_argument("--question", help="Yes/no question text")
    parser.add_argument("--resolution-date", help="YYYY-MM-DD")
    parser.add_argument("--resolution-criteria", help="Exactly what counts as YES")
    parser.add_argument("--background", help="Optional context")
    parser.add_argument("--runs", type=int, default=3, help="Independent forecasts (default 3)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--research-model", default=DEFAULT_RESEARCH_MODEL)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--offline", action="store_true", help="Fake run with no API calls (tests setup only)"
    )
    args = parser.parse_args()

    question = load_question(args)
    runs = max(1, min(args.runs, 5))
    print(f"\n📌 Question: {question['question']}\n📅 Resolution date: {question['resolution_date']}")
    print("🔒 Results are saved locally only — nothing is published to Metaculus.\n")

    if args.offline:
        research, sources = offline_research(question)
        forecasts = offline_forecasts(runs)
    else:
        api_key = get_api_key()
        research, sources = run_research(api_key, args.research_model, question)
        forecasts = run_forecasts(api_key, args.model, question, research, sources, runs)

    final_probability, representative = aggregate(forecasts)
    record = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "offline": args.offline,
        "model": "offline-demo" if args.offline else args.model,
        "question": question,
        "final_probability": final_probability,
        "individual_probabilities": [f["probability"] for f in forecasts],
        "representative_forecast": representative,
        "all_forecasts": forecasts,
        "sources": sources,
        "research": research,
        "published_to_metaculus": False,
    }
    json_path, md_path = save_results(record, Path(args.output_dir))

    print("\n" + "=" * 70)
    print(f"✅ Final probability (YES): {final_probability}%")
    print(f"   Individual runs: {record['individual_probabilities']}")
    print(f"   Sources found: {len(sources)}")
    print(f"   Arabic report: {md_path}")
    print(f"   Full data:     {json_path}")
    print("=" * 70)

    # In GitHub Actions, also show the report on the run's summary page.
    summary_file = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a", encoding="utf-8") as f:
            f.write(build_markdown(record))


if __name__ == "__main__":
    main()
