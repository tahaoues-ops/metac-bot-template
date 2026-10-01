"""
Research tools. Every tool is bound to a time window that ENDS at ctx.cutoff,
so in evaluation (cutoff = question open time) nothing published later can
reach the researcher. As a second guard, any result dated after the cutoff is
dropped in code, whatever the provider returned.

Tools:
  asknews      AskNews news search (pub_date filter)           ASKNEWS_CLIENT_ID + ASKNEWS_SECRET (or ASKNEWS_API_KEY)
  web          Exa web search (published-date filter)          EXA_API_KEY
  wikipedia    Wikipedia article *as it was* on the cutoff date (no key)
  futuresearch FutureSearch MCP server (optional)              see `futuresearch:` in config
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import requests

from advanced_bot.config import Config
from advanced_bot.llm import RunContext

logger = logging.getLogger(__name__)

WIKI_API = "https://en.wikipedia.org/w/api.php"
EXA_API = "https://api.exa.ai/search"
USER_AGENT = "metac-advanced-bot/0.1 (research tool; contact via repository)"


@dataclass
class SourceItem:
    title: str
    url: str
    snippet: str
    published: datetime | None
    provider: str

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "published": self.published.isoformat() if self.published else None,
            "provider": self.provider,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SourceItem":
        pub = d.get("published")
        return cls(
            title=d.get("title", ""),
            url=d.get("url", ""),
            snippet=d.get("snippet", ""),
            published=datetime.fromisoformat(pub) if pub else None,
            provider=d.get("provider", ""),
        )


@dataclass
class TimeWindow:
    start: datetime
    end: datetime  # == ctx.cutoff

    @classmethod
    def for_question(cls, cfg: Config, cutoff: datetime, resolution: datetime | None) -> "TimeWindow":
        """Lookback scales with the question horizon: clamp(factor * days_to_resolution, min, max)."""
        r = cfg.get_path
        if resolution is not None:
            horizon_days = max((resolution - cutoff).total_seconds() / 86400, 1)
        else:
            horizon_days = r("research.lookback_min_days")
        days = horizon_days * float(r("research.lookback_factor"))
        days = min(max(days, r("research.lookback_min_days")), r("research.lookback_max_days"))
        return cls(start=cutoff - timedelta(days=days), end=cutoff)

    def narrowed(self, days_back: int | None) -> "TimeWindow":
        """The model may ask for a shorter window, never a longer one or one past the cutoff."""
        if not days_back:
            return self
        start = max(self.start, self.end - timedelta(days=int(days_back)))
        return TimeWindow(start=start, end=self.end)


def available_tools(cfg: Config) -> list[str]:
    """Configured tools whose credentials are present."""
    wanted = cfg.get_path("research.tools", [])
    have = []
    for name in wanted:
        if name == "asknews" and not (
            (os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET")) or os.getenv("ASKNEWS_API_KEY")
        ):
            logger.warning("asknews skipped: ASKNEWS_CLIENT_ID/ASKNEWS_SECRET not set")
            continue
        if name == "web" and not os.getenv("EXA_API_KEY"):
            logger.warning("web search skipped: EXA_API_KEY not set")
            continue
        if name == "futuresearch" and not (
            cfg.get_path("futuresearch.url") or cfg.get_path("futuresearch.command")
        ):
            logger.warning("futuresearch skipped: futuresearch.url/command not configured")
            continue
        have.append(name)
    return have


def tool_schemas(names: list[str]) -> list[dict]:
    """OpenAI-style function definitions shown to the researcher model."""
    days_back = {
        "type": "integer",
        "description": "Optional: only look this many days back from today (cannot extend the window).",
    }
    specs = {
        "asknews": {
            "name": "search_news",
            "description": "Search recent news articles (AskNews). Use specific queries: names, places, events.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "days_back": days_back},
                "required": ["query"],
            },
        },
        "web": {
            "name": "web_search",
            "description": "Search the web (official sites, data, prediction markets, expert analysis).",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "days_back": days_back},
                "required": ["query"],
            },
        },
        "wikipedia": {
            "name": "wikipedia",
            "description": "Read the Wikipedia article on a topic (as it was on today's date). Good for background and base rates.",
            "parameters": {
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"],
            },
        },
        "futuresearch": {
            "name": "futuresearch",
            "description": "Deep research search via FutureSearch. Use for complex questions needing many sources.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
    return [{"type": "function", "function": specs[n]} for n in names if n in specs]


TOOL_NAME_TO_KEY = {"search_news": "asknews", "web_search": "web", "wikipedia": "wikipedia", "futuresearch": "futuresearch"}


async def run_tool(
    ctx: RunContext, cfg: Config, window: TimeWindow, tool_name: str, args: dict
) -> list[SourceItem]:
    key = TOOL_NAME_TO_KEY.get(tool_name)
    if key is None:
        raise ValueError(f"Unknown tool {tool_name}")
    query = str(args.get("query") or args.get("topic") or "").strip()
    if not query:
        raise ValueError("empty query")
    w = window.narrowed(args.get("days_back"))

    if ctx.fake:
        from advanced_bot.testing.fake import fake_search

        items = fake_search(ctx, key, query, w)
    elif key == "asknews":
        items = await _asknews(cfg, query, w)
    elif key == "web":
        items = await asyncio.to_thread(_exa, cfg, query, w)
    elif key == "wikipedia":
        items = await asyncio.to_thread(_wikipedia, query, w.end)
    else:
        items = await _futuresearch(cfg, query)

    kept = []
    for it in items:
        if it.published and it.published > ctx.cutoff + timedelta(hours=1):
            ctx.warnings.append(f"Dropped source dated after cutoff: {it.url} ({it.published.date()})")
            continue
        kept.append(it)
    ctx.tool_log.append(
        {"tool": key, "query": query, "window": [w.start.isoformat(), w.end.isoformat()], "results": len(kept)}
    )
    return kept


# ----------------------------------------------------------------------------- providers


async def _asknews(cfg: Config, query: str, w: TimeWindow) -> list[SourceItem]:
    from asknews_sdk import AsyncAskNewsSDK

    async with AsyncAskNewsSDK(
        client_id=os.getenv("ASKNEWS_CLIENT_ID"),
        client_secret=os.getenv("ASKNEWS_SECRET"),
        api_key=os.getenv("ASKNEWS_API_KEY") if not os.getenv("ASKNEWS_CLIENT_ID") else None,
        scopes={"news"},
    ) as ask:
        response = await ask.news.search_news(
            query=query,
            n_articles=int(cfg.get_path("research.results_per_search", 8)),
            start_timestamp=int(w.start.timestamp()),
            end_timestamp=int(w.end.timestamp()),
            time_filter="pub_date",
            historical=True,
            return_type="dicts",
            method="nl",
            strategy="default",
        )
    await asyncio.sleep(float(cfg.get_path("research.asknews_rate_limit_seconds", 11)))
    items = []
    for a in response.as_dicts or []:
        pub = a.pub_date
        if pub and pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        items.append(
            SourceItem(
                title=a.eng_title or a.title or "",
                url=str(a.article_url),
                snippet=(a.summary or "")[:1200],
                published=pub,
                provider=f"asknews:{a.source_id}",
            )
        )
    return items


def _exa(cfg: Config, query: str, w: TimeWindow) -> list[SourceItem]:
    body = {
        "query": query,
        "numResults": int(cfg.get_path("research.results_per_search", 8)),
        "type": "auto",
        "startPublishedDate": w.start.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "endPublishedDate": w.end.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contents": {"text": {"maxCharacters": 1500}},
    }
    r = requests.post(
        EXA_API,
        json=body,
        headers={"x-api-key": os.environ["EXA_API_KEY"], "Content-Type": "application/json"},
        timeout=60,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Exa error {r.status_code}: {r.text[:300]}")
    items = []
    for res in r.json().get("results", []):
        pub = _parse_date(res.get("publishedDate"))
        items.append(
            SourceItem(
                title=res.get("title") or res.get("url", ""),
                url=res.get("url", ""),
                snippet=(res.get("text") or "")[:1500],
                published=pub,
                provider="exa",
            )
        )
    return items


def _wikipedia(topic: str, as_of: datetime) -> list[SourceItem]:
    headers = {"User-Agent": USER_AGENT}
    search = requests.get(
        WIKI_API,
        params={"action": "query", "list": "search", "srsearch": topic, "srlimit": 1, "format": "json"},
        headers=headers,
        timeout=30,
    ).json()
    hits = search.get("query", {}).get("search", [])
    if not hits:
        return []
    title = hits[0]["title"]
    data = requests.get(
        WIKI_API,
        params={
            "action": "query",
            "prop": "revisions",
            "titles": title,
            "rvlimit": 1,
            "rvdir": "older",
            "rvstart": as_of.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "rvprop": "content|timestamp|ids",
            "rvslots": "main",
            "format": "json",
            "formatversion": 2,
        },
        headers=headers,
        timeout=30,
    ).json()
    pages = data.get("query", {}).get("pages", [])
    if not pages or not pages[0].get("revisions"):
        return []  # article did not exist yet on that date
    rev = pages[0]["revisions"][0]
    text = _strip_wikitext(rev["slots"]["main"]["content"])
    return [
        SourceItem(
            title=f"Wikipedia: {title} (revision of {rev['timestamp'][:10]})",
            url=f"https://en.wikipedia.org/w/index.php?oldid={rev['revid']}",
            snippet=text[:4000],
            published=_parse_date(rev["timestamp"]),
            provider="wikipedia",
        )
    ]


async def _futuresearch(cfg: Config, query: str) -> list[SourceItem]:
    """Generic MCP client call; tool name/arguments come from config."""
    from mcp import ClientSession

    fs = cfg.get("futuresearch", {})
    tool = fs.get("tool_name")
    if not tool:
        raise ValueError("futuresearch.tool_name is not set in config")
    args = {fs.get("query_argument", "query"): query}
    key = os.getenv(fs.get("api_key_env", "FUTURESEARCH_API_KEY"), "")

    if fs.get("transport", "streamable_http") == "stdio":
        from mcp.client.stdio import StdioServerParameters, stdio_client

        parts = fs["command"].split()
        params = StdioServerParameters(command=parts[0], args=parts[1:], env={**os.environ})
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool, args)
    else:
        from mcp.client.streamable_http import streamablehttp_client

        headers = {"Authorization": f"Bearer {key}"} if key else {}
        async with streamablehttp_client(fs["url"], headers=headers) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool, args)

    text = "\n".join(getattr(c, "text", "") for c in result.content if getattr(c, "text", None))
    urls = list(dict.fromkeys(re.findall(r"https?://[^\s)\]>\"']+", text)))
    items = [SourceItem(title="FutureSearch result", url=urls[0] if urls else "futuresearch://result",
                        snippet=text[:4000], published=None, provider="futuresearch")]
    items += [SourceItem(title=u, url=u, snippet="(cited by FutureSearch)", published=None, provider="futuresearch")
              for u in urls[1:10]]
    return items


# ----------------------------------------------------------------------------- helpers


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        d = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _strip_wikitext(text: str) -> str:
    text = re.sub(r"<ref[^>]*/>", "", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.S)
    for _ in range(3):  # nested templates
        text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    text = re.sub(r"\[\[(?:File|Image|Category):[^\]]*\]\]", "", text)
    text = re.sub(r"\[\[([^|\]]*\|)?([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[https?://\S+ ([^\]]+)\]", r"\1", text)
    text = re.sub(r"'{2,}", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
