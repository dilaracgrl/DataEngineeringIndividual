"""
Tool Name: trends_search
Description: Checks Google Trends real-time trending topics via the trendspyg
             library and returns whether a given query term appears among current
             trending searches. Used as the mainstream-phase signal in the
             Technology Pipeline Tracker — a technology appearing in mass Google
             Trends searches indicates it has crossed into public awareness and
             mainstream adoption.

             trendspyg fetches Google's public RSS trending feed (no auth, no
             headless browser required) returning ~20 currently trending terms
             with associated news headlines. The tool also performs a
             case-insensitive substring match to report whether the query term
             is represented in any current trend.

Parameters:
    query (str): Technology or concept to check for in trending topics.
    geo   (str): Country/region code for trends, e.g. "US", "GB", "DE".
                 Default: "US".

Returns:
    dict: {
        "query_found": bool,
        "matched_trends": list[str],
        "all_trends": list[dict],
        "fetched_at": str,
        "lineage": dict
    }

MCP Schema:
    {
        "name": "trends_search",
        "description": "Check if a technology appears in current Google Trends. Returns trending topics and whether the query matches any.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "geo":   {"type": "string", "default": "US"}
            },
            "required": ["query"]
        }
    }

Dependencies:
    pip install trendspyg

Library reference: https://github.com/flack0x/trendspyg
"""

import logging
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("trends_tool")


# ---------------------------------------------------------------------------
# W3C PROV lineage helper
# ---------------------------------------------------------------------------

def _build_prov_record(query: str, geo: str) -> dict:
    """
    Builds a W3C PROV-inspired lineage record for this Google Trends fetch.

    Tracks Entity (fetched trending dataset), Activity (this trends check),
    and Agent (this tool) so downstream consumers can trace data provenance.
    """
    return {
        "prov:type": "prov:Entity",
        "prov:wasGeneratedBy": {
            "prov:type": "prov:Activity",
            "prov:label": "trends_search",
            "prov:startedAtTime": datetime.now(timezone.utc).isoformat(),
            "prov:used": {
                "source": "Google Trends RSS (via trendspyg)",
                "query": query,
                "geo": geo,
                "method": "download_google_trends_rss",
            },
        },
        "prov:wasAttributedTo": {
            "prov:type": "prov:Agent",
            "prov:label": "TechPipelineTracker:trends_tool",
        },
    }


# ---------------------------------------------------------------------------
# Core tool function
# ---------------------------------------------------------------------------

def search_trends(query: str, geo: str = "US") -> dict:
    """
    Fetches current Google Trends RSS data and checks whether `query` appears
    among the trending topics.

    trendspyg.download_google_trends_rss() retrieves ~20 real-time trending
    search terms from Google's public RSS feed. This is the lightweight path
    (no Chrome required, ~0.2s, built-in 5-minute cache).

    The function performs a case-insensitive substring match: if any trending
    term *contains* the query (or the query contains a trending term), it is
    flagged as a match. This handles both exact hits ("ChatGPT" in trends) and
    partial hits ("GPT" matching "ChatGPT" trending).

    Args:
        query : Technology or concept name to search for in trending topics.
        geo   : Google Trends region code. Common values:
                "US" (United States), "GB" (UK), "DE" (Germany),
                "JP" (Japan), "AU" (Australia). Default: "US".

    Returns:
        dict with keys:
            query_found     (bool)       : True if the query matched any trend.
            matched_trends  (list[str])  : Trend terms that matched the query.
            all_trends      (list[dict]) : Full list of trending items returned.
                Each item: {"trend": str, "traffic": str, "news_articles": list}
            geo             (str)        : Region used for the fetch.
            fetched_at      (str)        : ISO 8601 timestamp of fetch.
            lineage         (dict)       : W3C PROV lineage record.

    Raises:
        ImportError  : If trendspyg is not installed.
        RuntimeError : On fetch or parsing errors from trendspyg.
    """
    try:
        import trendspyg
    except ImportError:
        raise ImportError(
            "trendspyg is not installed. Run: pip install trendspyg"
        )

    geo = geo.upper().strip()

    logger.info("trends_search called | query=%r | geo=%s", query, geo)

    lineage = _build_prov_record(query, geo)
    fetched_at = datetime.now(timezone.utc).isoformat()

    try:
        # download_google_trends_rss returns a list of dicts with keys:
        # "trend", "traffic", "news_articles"
        # cache=True (default) means repeated calls within ~5 minutes reuse
        # the same RSS response — prevents hammering Google's endpoint.
        raw_trends = trendspyg.download_google_trends_rss(geo=geo, cache=True)
    except Exception as e:
        logger.error("trendspyg fetch error: %s", e)
        raise RuntimeError(f"Failed to fetch Google Trends RSS: {e}") from e

    # Normalise to a consistent list-of-dicts shape regardless of what
    # trendspyg returns (it supports dict, DataFrame, JSON output formats;
    # the default is dict/list).
    all_trends: list[dict] = []
    if isinstance(raw_trends, list):
        for item in raw_trends:
            if isinstance(item, dict):
                all_trends.append({
                    "trend":         item.get("trend", ""),
                    "traffic":       item.get("traffic", ""),
                    "news_articles": item.get("news_articles", []),
                })
    elif hasattr(raw_trends, "to_dict"):
        # Pandas DataFrame fallback
        for _, row in raw_trends.iterrows():
            all_trends.append({
                "trend":         str(row.get("trend", "")),
                "traffic":       str(row.get("traffic", "")),
                "news_articles": [],
            })

    # Case-insensitive substring match between query and each trend term
    query_lower = query.lower()
    matched_trends = [
        t["trend"]
        for t in all_trends
        if query_lower in t["trend"].lower() or t["trend"].lower() in query_lower
    ]

    query_found = len(matched_trends) > 0

    if query_found:
        logger.info(
            "trends_search: query=%r FOUND in trends: %s", query, matched_trends
        )
    else:
        logger.info(
            "trends_search: query=%r not in current trending topics (geo=%s)",
            query, geo,
        )

    return {
        "query_found":    query_found,
        "matched_trends": matched_trends,
        "all_trends":     all_trends,
        "geo":            geo,
        "fetched_at":     fetched_at,
        "lineage":        lineage,
    }


# ---------------------------------------------------------------------------
# MCP-style tool metadata — used by agents for tool discovery
# ---------------------------------------------------------------------------

TOOL_DEFINITION = {
    "name": "trends_search",
    "description": (
        "Check if a technology or concept appears in current Google Trends "
        "real-time trending topics. Returns all trending terms for the specified "
        "region plus a flag indicating whether the query matches any of them. "
        "Use this as the mainstream signal — appearing in Google Trends indicates "
        "public awareness and mass adoption."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Technology or concept name to look for in trending topics.",
            },
            "geo": {
                "type": "string",
                "description": (
                    "Google Trends region code. Examples: 'US', 'GB', 'DE', 'JP', 'AU'. "
                    "Defaults to 'US'."
                ),
                "default": "US",
            },
        },
        "required": ["query"],
    },
}


# ---------------------------------------------------------------------------
# Quick smoke-test — run directly to verify trendspyg install and output shape
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = search_trends(query="AI", geo="US")
    print(f"\nQuery found in trends : {result['query_found']}")
    print(f"Matched trends        : {result['matched_trends']}")
    print(f"Total trends fetched  : {len(result['all_trends'])}")
    print(f"Fetched at            : {result['fetched_at']}")
    print(f"Lineage               : {result['lineage']['prov:wasGeneratedBy']['prov:label']}")
    print("\n--- All trends ---")
    for t in result["all_trends"]:
        print(f"  {t['trend']:40s}  traffic: {t['traffic']}")
