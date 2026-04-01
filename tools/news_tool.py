"""
Tool Name: news_search
Description: Searches news articles via NewsAPI combining a technology query
             with funding-specific keywords. Returns structured article metadata
             and press coverage volume metrics. Used as the Stage 3 signal in
             the Technology Pipeline Tracker.

             Stage 3 signal — Investment Press Coverage.
             When venture capital money moves into a technology, it hits the
             news. Funding announcements, Series A/B/C closings, and VC
             partnership deals are reliably covered by tech press within days
             of closing. NewsAPI allows combining a technology term with funding
             vocabulary ("raises", "Series A", "venture capital", "investment")
             to isolate investment-phase press coverage. A spike in funding
             articles for a technology = strong signal it has entered the VC
             growth phase. Coverage volume over a trailing window (get_news_volume)
             shows whether that signal is growing, peaking, or declining — which
             indicates whether the technology is entering or exiting the
             investment phase.

Parameters (search_funding_news):
    query      (str) : Technology or concept to search for in funding news.
    days_back  (int) : Look-back window in days (1–30). Default: 30.
                       NewsAPI free tier caps at 30 days of history.
    page_size  (int) : Max articles to return (1–100). Default: 10.

Parameters (get_news_volume):
    query      (str) : Technology or concept to measure press coverage for.
    days_back  (int) : Look-back window in days (1–30). Default: 90.
                       Note: free tier caps at 30 days — set accordingly.

Returns:
    search_funding_news → List[dict]: Article metadata + lineage.
    get_news_volume     → dict: Coverage volume metrics + lineage.

MCP Schema (search_funding_news):
    {
        "name": "news_search_funding",
        "description": "Search for funding/investment news about a technology via NewsAPI.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query":     {"type": "string"},
                "days_back": {"type": "integer", "default": 30, "minimum": 1, "maximum": 30},
                "page_size": {"type": "integer", "default": 10, "minimum": 1, "maximum": 100}
            },
            "required": ["query"]
        }
    }

MCP Schema (get_news_volume):
    {
        "name": "news_get_volume",
        "description": "Measure press coverage volume for a technology over a trailing window.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query":     {"type": "string"},
                "days_back": {"type": "integer", "default": 30, "minimum": 1, "maximum": 30}
            },
            "required": ["query"]
        }
    }

Required .env keys:
    NEWS_API_KEY  — API key from newsapi.org (free tier available).
                   Free tier: 100 requests/day, 30 days of history, 1 page.
                   Paid tier: unlimited history, pagination.

API Reference: https://newsapi.org/docs/endpoints/everything
Library:       pip install newsapi-python
"""

import logging
import os
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from newsapi import NewsApiClient
from newsapi.newsapi_exception import NewsAPIException

# ---------------------------------------------------------------------------
# Environment & logging setup
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("news_tool")

# Funding-specific terms ORed into the query to filter for investment-phase
# press coverage. These are the vocabulary of VC funding announcements.
_FUNDING_TERMS = [
    "funding",
    "Series A",
    "Series B",
    "Series C",
    "venture capital",
    "investment",
    "raises",
    "startup",
    "seed round",
]


# ---------------------------------------------------------------------------
# Client initialisation
# ---------------------------------------------------------------------------

def _get_newsapi_client() -> NewsApiClient:
    """
    Initialises and returns a NewsApiClient using the key from .env.

    Required .env key:
        NEWS_API_KEY  — obtain at https://newsapi.org/register
    """
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Missing NewsAPI credentials. Set NEWS_API_KEY in .env. "
            "Register for a free key at https://newsapi.org/register"
        )
    return NewsApiClient(api_key=api_key)


# ---------------------------------------------------------------------------
# W3C PROV lineage helper
# ---------------------------------------------------------------------------

def _build_prov_record(activity_label: str, query: str, **kwargs) -> dict:
    """
    Builds a W3C PROV-inspired lineage record for a NewsAPI fetch.

    activity_label distinguishes between search_funding_news and
    get_news_volume in the lineage trail.
    """
    return {
        "prov:type": "prov:Entity",
        "prov:wasGeneratedBy": {
            "prov:type": "prov:Activity",
            "prov:label": activity_label,
            "prov:startedAtTime": datetime.now(timezone.utc).isoformat(),
            "prov:used": {
                "source": "NewsAPI v2 (/everything endpoint)",
                "query": query,
                **kwargs,
            },
        },
        "prov:wasAttributedTo": {
            "prov:type": "prov:Agent",
            "prov:label": "TechPipelineTracker:news_tool",
        },
    }


# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------

def _build_funding_query(query: str) -> str:
    """
    Combines the technology query with funding vocabulary using NewsAPI's
    advanced query syntax.

    NewsAPI supports boolean operators in the `q` parameter:
        AND, OR, NOT, and phrase quoting with "".

    Strategy: require the technology term AND at least one funding keyword.
    This filters out general technology news and surfaces investment-phase
    coverage specifically.

    Example output for query="diffusion models":
        diffusion models AND (funding OR "Series A" OR "venture capital" OR
        investment OR raises OR startup OR "seed round")
    """
    funding_or = " OR ".join(
        f'"{term}"' if " " in term else term
        for term in _FUNDING_TERMS
    )
    return f'"{query}" AND ({funding_or})'


# ---------------------------------------------------------------------------
# Core tool functions
# ---------------------------------------------------------------------------

def search_funding_news(
    query: str,
    days_back: int = 30,
    page_size: int = 10,
) -> list[dict]:
    """
    Searches for news articles combining `query` with funding vocabulary and
    returns structured article metadata.

    Stage 3 signal: the presence of funding-vocabulary articles about a
    technology is a direct indicator it has entered the VC investment phase.
    Each article returned represents press coverage of a (likely) real funding
    event — names, dates, and sources allow the analyst agent to verify and cite.

    Args:
        query     : Technology or concept to search for in funding news.
        days_back : How many days back to search (1–30; free tier cap is 30).
        page_size : Max articles to return per call (1–100).

    Returns:
        List of article dicts, each containing:
            title, description, content_snippet, source_name, author,
            url, published_at, source, fetched_at, lineage

    Raises:
        EnvironmentError : If NEWS_API_KEY is not set.
        ValueError       : On invalid parameter values.
        RuntimeError     : On NewsAPI errors.
    """
    days_back = max(1, min(days_back, 30))   # Free tier hard cap
    page_size = max(1, min(page_size, 100))

    logger.info(
        "news_search_funding called | query=%r | days_back=%d | page_size=%d",
        query, days_back, page_size,
    )

    client = _get_newsapi_client()
    funding_query = _build_funding_query(query)
    lineage = _build_prov_record(
        "news_search_funding",
        query,
        effective_query=funding_query,
        days_back=days_back,
        page_size=page_size,
    )
    fetched_at = datetime.now(timezone.utc).isoformat()

    from_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime(
        "%Y-%m-%d"
    )

    try:
        response = client.get_everything(
            q=funding_query,
            from_param=from_date,
            language="en",
            sort_by="publishedAt",      # Most recent first
            page_size=page_size,
            page=1,
        )
    except NewsAPIException as e:
        logger.error("NewsAPI error: %s", e)
        raise RuntimeError(f"NewsAPI returned an error: {e}") from e

    if response.get("status") != "ok":
        raise RuntimeError(
            f"NewsAPI returned non-ok status: {response.get('status')} — "
            f"{response.get('message', 'no message')}"
        )

    articles = []
    for raw in response.get("articles", []):
        article = {
            "title":            raw.get("title") or "",
            "description":      raw.get("description") or "",
            # 'content' from NewsAPI is truncated at 200 chars on free tier
            "content_snippet":  raw.get("content") or "",
            "source_name":      (raw.get("source") or {}).get("name", ""),
            "author":           raw.get("author") or "",
            "url":              raw.get("url", ""),
            "published_at":     raw.get("publishedAt", ""),
            # Provenance fields
            "source":           "newsapi",
            "fetched_at":       fetched_at,
            "lineage":          lineage,
        }
        articles.append(article)

    logger.info(
        "news_search_funding returned %d articles for query=%r "
        "(total available: %d)",
        len(articles), query, response.get("totalResults", 0),
    )
    return articles


def get_news_volume(query: str, days_back: int = 30) -> dict:
    """
    Measures press coverage volume for a technology over a trailing window
    by fetching articles and aggregating metadata.

    Stage 3 signal: coverage volume is distinct from funding-specific search.
    This function captures *all* press coverage (not just funding news) to
    measure whether a technology is gaining or has already peaked in media
    attention. High total_articles with recent dates = technology is in active
    VC/growth phase press cycle. Falling coverage with older dates = may have
    moved past the investment phase into Big Tech absorption.

    Args:
        query     : Technology or concept to measure press coverage for.
        days_back : Trailing window in days (capped at 30 for free tier).

    Returns:
        dict with keys:
            query, total_articles, sources, earliest_article,
            latest_article, sample_headlines (up to 5),
            source, fetched_at, lineage

    Raises:
        EnvironmentError : If NEWS_API_KEY is not set.
        RuntimeError     : On NewsAPI errors.
    """
    days_back = max(1, min(days_back, 30))

    logger.info(
        "news_get_volume called | query=%r | days_back=%d", query, days_back
    )

    client = _get_newsapi_client()
    lineage = _build_prov_record(
        "news_get_volume", query, days_back=days_back
    )
    fetched_at = datetime.now(timezone.utc).isoformat()

    from_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime(
        "%Y-%m-%d"
    )

    try:
        # Fetch up to 100 articles to get a representative volume sample
        response = client.get_everything(
            q=f'"{query}"',
            from_param=from_date,
            language="en",
            sort_by="publishedAt",
            page_size=100,
            page=1,
        )
    except NewsAPIException as e:
        logger.error("NewsAPI error in get_news_volume: %s", e)
        raise RuntimeError(f"NewsAPI returned an error: {e}") from e

    if response.get("status") != "ok":
        raise RuntimeError(
            f"NewsAPI returned non-ok status: {response.get('status')}"
        )

    raw_articles = response.get("articles", [])
    total_results = response.get("totalResults", 0)

    # Collect unique sources and publication dates from the sample
    sources = sorted({
        (a.get("source") or {}).get("name", "")
        for a in raw_articles
        if (a.get("source") or {}).get("name")
    })

    dates = sorted([
        a["publishedAt"] for a in raw_articles if a.get("publishedAt")
    ])

    # Sample headlines give the analyst agent a quick sense of what the
    # coverage is actually about without returning full article content
    sample_headlines = [
        a.get("title", "") for a in raw_articles[:5] if a.get("title")
    ]

    volume = {
        "query":             query,
        "total_articles":    total_results,   # Full count, not just page 1
        "articles_sampled":  len(raw_articles),
        "sources":           sources,
        "earliest_article":  dates[0]  if dates else None,
        "latest_article":    dates[-1] if dates else None,
        "sample_headlines":  sample_headlines,
        # Provenance fields
        "source":            "newsapi",
        "fetched_at":        fetched_at,
        "lineage":           lineage,
    }

    logger.info(
        "news_get_volume | query=%r | total_articles=%d | sources=%d",
        query, total_results, len(sources),
    )
    return volume


# ---------------------------------------------------------------------------
# MCP-style tool metadata — used by agents for tool discovery
# ---------------------------------------------------------------------------

TOOL_DEFINITION_FUNDING = {
    "name": "news_search_funding",
    "description": (
        "Search for funding and investment news about a technology via NewsAPI. "
        "Combines the query with terms like 'Series A', 'venture capital', "
        "'raises', and 'investment' to surface VC-phase press coverage. "
        "Use this as a Stage 3 signal — funding news confirms a technology "
        "has entered the VC growth phase."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Technology or concept to search for in funding news.",
            },
            "days_back": {
                "type": "integer",
                "description": "Look-back window in days (max 30 on free tier).",
                "default": 30,
                "minimum": 1,
                "maximum": 30,
            },
            "page_size": {
                "type": "integer",
                "description": "Maximum articles to return.",
                "default": 10,
                "minimum": 1,
                "maximum": 100,
            },
        },
        "required": ["query"],
    },
}

TOOL_DEFINITION_VOLUME = {
    "name": "news_get_volume",
    "description": (
        "Measure overall press coverage volume for a technology over a trailing "
        "window. Returns total article count, unique sources, date range of "
        "coverage, and sample headlines. Use this to track whether media attention "
        "is growing (entering VC phase) or declining (exiting it)."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Technology or concept to measure press coverage for.",
            },
            "days_back": {
                "type": "integer",
                "description": "Trailing window in days (max 30 on free tier).",
                "default": 30,
                "minimum": 1,
                "maximum": 30,
            },
        },
        "required": ["query"],
    },
}

TOOL_DEFINITIONS = [TOOL_DEFINITION_FUNDING, TOOL_DEFINITION_VOLUME]


# ---------------------------------------------------------------------------
# Quick smoke-test — run directly to verify key and output shape
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    QUERY = "large language models"

    print(f"\n=== search_funding_news: '{QUERY}' ===")
    articles = search_funding_news(query=QUERY, days_back=30, page_size=3)
    for i, article in enumerate(articles, 1):
        print(f"\n--- Article {i} ---")
        print(f"Title     : {article['title']}")
        print(f"Source    : {article['source_name']}")
        print(f"Published : {article['published_at']}")
        print(f"URL       : {article['url']}")
        print(f"Lineage   : {article['lineage']['prov:wasGeneratedBy']['prov:label']}")

    print(f"\n=== get_news_volume: '{QUERY}' ===")
    vol = get_news_volume(query=QUERY, days_back=30)
    print(f"Total articles    : {vol['total_articles']}")
    print(f"Articles sampled  : {vol['articles_sampled']}")
    print(f"Sources           : {', '.join(vol['sources'][:5])}")
    print(f"Earliest          : {vol['earliest_article']}")
    print(f"Latest            : {vol['latest_article']}")
    print(f"Headlines sample  : {vol['sample_headlines']}")
    print(f"Fetched at        : {vol['fetched_at']}")
    print(f"Lineage           : {vol['lineage']['prov:wasGeneratedBy']['prov:label']}")
