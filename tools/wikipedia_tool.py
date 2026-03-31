"""
Tool Name: wikipedia_search
Description: Checks whether a Wikipedia article exists for a technology and
             retrieves page view statistics via the Wikimedia REST and
             Pageviews APIs. Used as the Stage 5 signal in the Technology
             Pipeline Tracker.

             Stage 5 signal — Mainstream Awareness Confirmation.
             Wikipedia is edited by volunteers who write articles about things
             the general public cares about — not things academics or developers
             care about. A detailed Wikipedia page means a concept has crossed
             from specialist knowledge into general public awareness. Page view
             counts confirm whether that awareness is active (people are looking
             it up) vs. passive (the article exists but nobody reads it).
             A rising page view trend on a technology's Wikipedia article is
             one of the clearest signals that the concept has reached mainstream
             adoption — the same phase where non-technical people start asking
             "what is X?" and where consumer products start referencing it.

Parameters (search_wikipedia):
    query (str) : Technology or concept to check for on Wikipedia.

Parameters (get_page_views):
    query     (str) : Technology or concept (used to resolve the article title).
    days_back (int) : Number of days of view history to retrieve (1–90).
                      Default: 30.

Returns:
    search_wikipedia → dict: Page existence + summary + lineage.
    get_page_views   → dict: Daily view statistics + trend + lineage.

MCP Schema (search_wikipedia):
    {
        "name": "wikipedia_search",
        "description": "Check if a Wikipedia article exists for a technology and return its summary.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    }

MCP Schema (get_page_views):
    {
        "name": "wikipedia_page_views",
        "description": "Get daily Wikipedia page view counts for a technology article.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query":     {"type": "string"},
                "days_back": {"type": "integer", "default": 30, "minimum": 1, "maximum": 90}
            },
            "required": ["query"]
        }
    }

No authentication required. Both APIs are completely free and open.

API References:
    Search + Summary : https://en.wikipedia.org/w/api.php
    Page views       : https://wikimedia.org/api/rest_v1/metrics/pageviews
"""

import logging
from datetime import datetime, timedelta, timezone

import requests

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("wikipedia_tool")

_MEDIAWIKI_API  = "https://en.wikipedia.org/w/api.php"
_PAGEVIEWS_API  = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
_HEADERS = {
    "User-Agent": "TechPipelineTracker/1.0 (research tool; contact: research@example.com)",
    "Accept": "application/json",
}


# ---------------------------------------------------------------------------
# W3C PROV lineage helper
# ---------------------------------------------------------------------------

def _build_prov_record(activity_label: str, query: str, **kwargs) -> dict:
    """
    Builds a W3C PROV-inspired lineage record for a Wikipedia/Wikimedia fetch.

    activity_label distinguishes between search_wikipedia and get_page_views
    in the lineage trail.
    """
    return {
        "prov:type": "prov:Entity",
        "prov:wasGeneratedBy": {
            "prov:type": "prov:Activity",
            "prov:label": activity_label,
            "prov:startedAtTime": datetime.now(timezone.utc).isoformat(),
            "prov:used": {
                "source": "Wikipedia MediaWiki API / Wikimedia Pageviews API",
                "query": query,
                **kwargs,
            },
        },
        "prov:wasAttributedTo": {
            "prov:type": "prov:Agent",
            "prov:label": "TechPipelineTracker:wikipedia_tool",
        },
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_article_title(query: str) -> str | None:
    """
    Uses the Wikipedia search API to find the canonical article title for a
    query string.

    Wikipedia page titles are case-sensitive and often differ from natural
    queries (e.g. "neural net" → "Artificial neural network"). This step
    resolves the best-matching canonical title before further API calls.

    Returns:
        The canonical page title string, or None if no match found.
    """
    params = {
        "action":   "query",
        "list":     "search",
        "srsearch": query,
        "srlimit":  1,
        "format":   "json",
        "utf8":     1,
    }
    try:
        resp = requests.get(_MEDIAWIKI_API, headers=_HEADERS, params=params, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error("Wikipedia search API error: %s", e)
        raise RuntimeError(f"Wikipedia search API error: {e}") from e

    results = resp.json().get("query", {}).get("search", [])
    if not results:
        return None
    return results[0].get("title")


def _get_page_summary(title: str) -> dict:
    """
    Fetches the Wikipedia page summary using the MediaWiki extracts API.

    Returns a dict with: extract (plain text summary), page_id, url.
    The extract is the first paragraph of the article — sufficient for
    determining whether the page is about the right concept.
    """
    params = {
        "action":      "query",
        "prop":        "extracts|info|revisions",
        "exintro":     True,        # Only the intro section
        "explaintext": True,        # Plain text, not HTML
        "inprop":      "url",       # Include canonical URL
        "rvprop":      "timestamp", # Earliest revision = creation date
        "rvdir":       "newer",
        "rvlimit":     1,
        "titles":      title,
        "format":      "json",
        "utf8":        1,
    }
    try:
        resp = requests.get(_MEDIAWIKI_API, headers=_HEADERS, params=params, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error("Wikipedia extracts API error: %s", e)
        raise RuntimeError(f"Wikipedia extracts API error: {e}") from e

    pages = resp.json().get("query", {}).get("pages", {})
    # MediaWiki returns pages as a dict keyed by page_id; -1 = not found
    page = next(iter(pages.values()), {})

    if page.get("pageid", -1) == -1:
        return {}

    # Creation date from the earliest revision timestamp
    revisions = page.get("revisions", [])
    created_at = revisions[0].get("timestamp", "") if revisions else ""

    return {
        "page_id":    page.get("pageid"),
        "title":      page.get("title", ""),
        "extract":    page.get("extract", ""),
        "url":        page.get("canonicalurl", f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"),
        "created_at": created_at,
    }


# ---------------------------------------------------------------------------
# Core tool functions
# ---------------------------------------------------------------------------

def search_wikipedia(query: str) -> dict:
    """
    Checks whether a Wikipedia article exists for `query` and returns its
    summary if found.

    Stage 5 signal: the existence of a dedicated Wikipedia article is a
    binary mainstream-awareness checkpoint. A concept without a Wikipedia
    page has not yet crossed into public consciousness. A concept with a
    long, well-cited Wikipedia article has clearly achieved mainstream status.
    The summary extract allows the analyst agent to confirm the article is
    about the correct concept (not a disambiguation page).

    Args:
        query : Technology or concept name to look up on Wikipedia.

    Returns:
        dict with keys:
            exists (bool), title, summary (first paragraph),
            page_url, created_at, source, fetched_at, lineage

    Raises:
        RuntimeError : On HTTP or network errors.
    """
    logger.info("wikipedia_search called | query=%r", query)

    lineage = _build_prov_record("wikipedia_search", query)
    fetched_at = datetime.now(timezone.utc).isoformat()

    canonical_title = _resolve_article_title(query)

    if not canonical_title:
        logger.info("wikipedia_search: no article found for query=%r", query)
        return {
            "exists":     False,
            "title":      None,
            "summary":    None,
            "page_url":   None,
            "created_at": None,
            "source":     "wikipedia",
            "fetched_at": fetched_at,
            "lineage":    lineage,
        }

    page_data = _get_page_summary(canonical_title)

    if not page_data:
        return {
            "exists":     False,
            "title":      canonical_title,
            "summary":    None,
            "page_url":   None,
            "created_at": None,
            "source":     "wikipedia",
            "fetched_at": fetched_at,
            "lineage":    lineage,
        }

    # Truncate the extract to the first 500 chars — enough to confirm relevance
    # without bloating the response payload
    extract = page_data.get("extract", "")
    summary = extract[:500].rstrip() + ("…" if len(extract) > 500 else "")

    result = {
        "exists":     True,
        "title":      page_data.get("title", ""),
        "summary":    summary,
        "page_url":   page_data.get("url", ""),
        "created_at": page_data.get("created_at", ""),
        "source":     "wikipedia",
        "fetched_at": fetched_at,
        "lineage":    lineage,
    }

    logger.info(
        "wikipedia_search: found article '%s' for query=%r",
        result["title"], query,
    )
    return result


def get_page_views(query: str, days_back: int = 30) -> dict:
    """
    Retrieves daily Wikipedia page view counts for the article matching
    `query` and computes a trend direction.

    Stage 5 signal: page view volume and trend confirm whether mainstream
    awareness is active and growing. A rising trend (more views each week)
    on a technology's Wikipedia page confirms that public curiosity is
    increasing — the clearest late-stage adoption signal. A flat or falling
    trend on a page with millions of total views indicates the technology
    has been fully absorbed into mainstream knowledge.

    Trend calculation:
        Compares the average daily views in the first half of the window
        vs. the second half. More than 10% increase = "rising",
        more than 10% decrease = "falling", otherwise = "stable".

    Args:
        query     : Technology or concept name (resolved to article title).
        days_back : Days of view history to retrieve (1–90).

    Returns:
        dict with keys:
            query, article_title, average_daily_views, total_views,
            peak_day {date, views}, trend ("rising"|"falling"|"stable"),
            daily_views [{"date": str, "views": int}, ...],
            source, fetched_at, lineage

    Raises:
        RuntimeError : On HTTP or network errors.
        ValueError   : If no Wikipedia article found for query.
    """
    days_back = max(1, min(days_back, 90))

    logger.info(
        "wikipedia_page_views called | query=%r | days_back=%d", query, days_back
    )

    lineage = _build_prov_record("wikipedia_page_views", query, days_back=days_back)
    fetched_at = datetime.now(timezone.utc).isoformat()

    # Resolve canonical title first
    canonical_title = _resolve_article_title(query)
    if not canonical_title:
        raise ValueError(
            f"No Wikipedia article found for query '{query}'. "
            "Check the spelling or try a broader term."
        )

    # Wikimedia Pageviews API URL format:
    # /per-article/{project}/{access}/{agent}/{article}/{granularity}/{start}/{end}
    end_date   = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days_back)

    # Pageviews API expects YYYYMMDD00 format (trailing "00" = hour placeholder)
    start_str = start_date.strftime("%Y%m%d") + "00"
    end_str   = end_date.strftime("%Y%m%d") + "00"

    # Article title must be URL-encoded with spaces as underscores
    article_slug = canonical_title.replace(" ", "_")

    url = (
        f"{_PAGEVIEWS_API}/en.wikipedia/all-access/all-agents"
        f"/{requests.utils.quote(article_slug, safe='')}/daily/{start_str}/{end_str}"
    )

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=15)
    except requests.exceptions.ConnectionError as e:
        logger.error("Wikimedia Pageviews API network error: %s", e)
        raise RuntimeError(f"Network error reaching Wikimedia Pageviews API: {e}") from e
    except requests.exceptions.Timeout:
        raise RuntimeError("Wikimedia Pageviews API request timed out after 15 seconds")

    if resp.status_code == 404:
        # 404 means the article exists but has no view data for this range
        logger.warning(
            "No page view data found for '%s' in the requested date range",
            canonical_title,
        )
        return {
            "query":               query,
            "article_title":       canonical_title,
            "average_daily_views": 0,
            "total_views":         0,
            "peak_day":            None,
            "trend":               "no_data",
            "daily_views":         [],
            "source":              "wikimedia_pageviews",
            "fetched_at":          fetched_at,
            "lineage":             lineage,
        }

    if not resp.ok:
        raise RuntimeError(
            f"Wikimedia Pageviews API returned HTTP {resp.status_code}: {resp.text[:200]}"
        )

    items = resp.json().get("items", [])

    daily_views = [
        {
            "date":  item["timestamp"][:8],   # YYYYMMDD → strip trailing "00"
            "views": item.get("views", 0),
        }
        for item in items
    ]

    if not daily_views:
        return {
            "query":               query,
            "article_title":       canonical_title,
            "average_daily_views": 0,
            "total_views":         0,
            "peak_day":            None,
            "trend":               "no_data",
            "daily_views":         [],
            "source":              "wikimedia_pageviews",
            "fetched_at":          fetched_at,
            "lineage":             lineage,
        }

    views_list  = [d["views"] for d in daily_views]
    total_views = sum(views_list)
    avg_views   = round(total_views / len(views_list), 1)
    peak        = max(daily_views, key=lambda d: d["views"])

    # Trend: compare first half avg vs second half avg
    mid = len(views_list) // 2
    first_half_avg  = sum(views_list[:mid])  / max(mid, 1)
    second_half_avg = sum(views_list[mid:])  / max(len(views_list) - mid, 1)

    if first_half_avg == 0:
        trend = "rising" if second_half_avg > 0 else "stable"
    else:
        change_ratio = (second_half_avg - first_half_avg) / first_half_avg
        if change_ratio > 0.10:
            trend = "rising"
        elif change_ratio < -0.10:
            trend = "falling"
        else:
            trend = "stable"

    result = {
        "query":               query,
        "article_title":       canonical_title,
        "average_daily_views": avg_views,
        "total_views":         total_views,
        "peak_day":            peak,
        "trend":               trend,
        "daily_views":         daily_views,
        # Provenance fields
        "source":              "wikimedia_pageviews",
        "fetched_at":          fetched_at,
        "lineage":             lineage,
    }

    logger.info(
        "wikipedia_page_views | article='%s' | avg=%.1f | total=%d | trend=%s",
        canonical_title, avg_views, total_views, trend,
    )
    return result


# ---------------------------------------------------------------------------
# MCP-style tool metadata — used by agents for tool discovery
# ---------------------------------------------------------------------------

TOOL_DEFINITION_SEARCH = {
    "name": "wikipedia_search",
    "description": (
        "Check if a Wikipedia article exists for a technology and return its "
        "summary. Use this as a Stage 5 mainstream-awareness signal — the "
        "existence of a detailed Wikipedia article confirms a concept has crossed "
        "from specialist into general public knowledge."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Technology or concept name to look up on Wikipedia.",
            },
        },
        "required": ["query"],
    },
}

TOOL_DEFINITION_VIEWS = {
    "name": "wikipedia_page_views",
    "description": (
        "Get daily Wikipedia page view counts for a technology article and compute "
        "a trend direction (rising/falling/stable). Use this to confirm active "
        "mainstream awareness — rising page views on a technology article indicates "
        "the public is actively seeking information about it."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Technology or concept name to retrieve page views for.",
            },
            "days_back": {
                "type": "integer",
                "description": "Number of days of view history to retrieve.",
                "default": 30,
                "minimum": 1,
                "maximum": 90,
            },
        },
        "required": ["query"],
    },
}

TOOL_DEFINITIONS = [TOOL_DEFINITION_SEARCH, TOOL_DEFINITION_VIEWS]


# ---------------------------------------------------------------------------
# Quick smoke-test — run directly to verify connectivity and output shape
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    QUERY = "transformer neural network"

    print(f"\n=== search_wikipedia: '{QUERY}' ===")
    page = search_wikipedia(query=QUERY)
    print(f"Exists     : {page['exists']}")
    print(f"Title      : {page['title']}")
    print(f"URL        : {page['page_url']}")
    print(f"Created    : {page['created_at']}")
    print(f"Summary    : {page['summary'][:120]}...")
    print(f"Lineage    : {page['lineage']['prov:wasGeneratedBy']['prov:label']}")

    if page["exists"]:
        print(f"\n=== get_page_views: '{QUERY}' ===")
        views = get_page_views(query=QUERY, days_back=30)
        print(f"Article title     : {views['article_title']}")
        print(f"Avg daily views   : {views['average_daily_views']}")
        print(f"Total views       : {views['total_views']}")
        print(f"Peak day          : {views['peak_day']}")
        print(f"Trend             : {views['trend']}")
        print(f"Days of data      : {len(views['daily_views'])}")
        print(f"Fetched at        : {views['fetched_at']}")
        print(f"Lineage           : {views['lineage']['prov:wasGeneratedBy']['prov:label']}")
