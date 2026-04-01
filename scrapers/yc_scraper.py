"""
Tool Name: yc_search
Description: Scrapes Y Combinator's public company directory to find portfolio
             companies matching a technology keyword. Returns structured company
             metadata and aggregate funding statistics.

             Stage 2 signal — YC Startup Backing.
             Y Combinator is the world's most influential startup accelerator.
             When YC starts funding companies in a technology space, it is one
             of the earliest and strongest signals that the concept has moved
             from GitHub projects into real businesses with paying customers
             and investor conviction. The batch year is especially informative:
             if YC W21 companies were building in a space and YC W24 companies
             are still entering it, the space is in active early commercialisation.
             If YC stopped funding in that space after W22, it may have been
             absorbed by incumbents. Tracking the spread of batches (earliest
             to latest) shows how long YC has believed in a technology — a
             3+ year span with growing company count = approaching mainstream.

Parameters (search_yc_companies):
    query (str) : Technology keyword to search the YC directory for.
    limit (int) : Maximum companies to return (1–50). Default: 20.

Parameters (get_yc_stats):
    query (str) : Technology keyword to aggregate YC statistics for.

Returns:
    search_yc_companies → List[dict]: Company metadata + lineage.
    get_yc_stats        → dict: Aggregate funding statistics + lineage.

MCP Schema (search_yc_companies):
    {
        "name": "yc_search_companies",
        "description": "Search YC portfolio companies by technology keyword.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 20, "minimum": 1, "maximum": 50}
            },
            "required": ["query"]
        }
    }

MCP Schema (get_yc_stats):
    {
        "name": "yc_get_stats",
        "description": "Aggregate YC funding statistics for a technology space.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    }

Scraping notes:
    - Robots.txt is checked before every fetch session.
    - 2-second delay between requests to avoid hammering the server.
    - YC's company directory is a Next.js app; company data is embedded in
      a <script id="__NEXT_DATA__"> JSON block on the page — no JS execution
      or headless browser required.
    - No credentials required; the directory is fully public.
"""

import json
import logging
import re
import time
import urllib.parse
import urllib.robotparser
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("yc_scraper")

_BASE_URL       = "https://www.ycombinator.com"
_COMPANIES_URL  = f"{_BASE_URL}/companies"
_ROBOTS_URL     = f"{_BASE_URL}/robots.txt"
_REQUEST_DELAY  = 2  # seconds between requests — polite scraping

_HEADERS = {
    # Identify ourselves honestly — do not impersonate a real browser silently.
    # YC's directory is public data; using a descriptive UA is good practice.
    "User-Agent": (
        "TechPipelineTracker/1.0 (academic research scraper; "
        "reads public YC company directory)"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

# Batch sort order: convert YC batch codes to sortable integers.
# Format is S/W + 2-digit year (e.g. W24, S23). W = Winter (Q1), S = Summer (Q3).
# We encode as year * 10 + season_offset so batches sort chronologically.
_SEASON_OFFSET = {"W": 1, "S": 2}  # W comes before S in the same year


# ---------------------------------------------------------------------------
# Robots.txt check
# ---------------------------------------------------------------------------

def _is_scraping_allowed(url: str) -> bool:
    """
    Checks whether scraping the target URL is permitted by robots.txt.

    Uses stdlib urllib.robotparser — fetches and parses /robots.txt from
    the target domain, then checks our User-Agent against the requested path.

    Returns True if allowed or if robots.txt is unreachable (fail-open,
    since an inaccessible robots.txt means we cannot determine restrictions).
    """
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(_ROBOTS_URL)
    try:
        rp.read()
        allowed = rp.can_fetch(_HEADERS["User-Agent"], url)
        if not allowed:
            logger.warning("robots.txt disallows scraping %s", url)
        return allowed
    except Exception as e:
        # Network error fetching robots.txt — log and proceed cautiously
        logger.warning("Could not fetch robots.txt (%s) — proceeding with caution", e)
        return True


# ---------------------------------------------------------------------------
# W3C PROV lineage helper
# ---------------------------------------------------------------------------

def _build_prov_record(activity_label: str, query: str, **kwargs) -> dict:
    """
    Builds a W3C PROV-inspired lineage record for a YC scrape operation.

    activity_label distinguishes between search_yc_companies and
    get_yc_stats in the lineage trail.
    """
    return {
        "prov:type": "prov:Entity",
        "prov:wasGeneratedBy": {
            "prov:type": "prov:Activity",
            "prov:label": activity_label,
            "prov:startedAtTime": datetime.now(timezone.utc).isoformat(),
            "prov:used": {
                "source": "Y Combinator company directory (scraped)",
                "base_url": _COMPANIES_URL,
                "query": query,
                **kwargs,
            },
        },
        "prov:wasAttributedTo": {
            "prov:type": "prov:Agent",
            "prov:label": "TechPipelineTracker:yc_scraper",
        },
    }


# ---------------------------------------------------------------------------
# Fetch and parse helpers
# ---------------------------------------------------------------------------

def _extract_algolia_credentials(html: str) -> tuple[str, str] | None:
    """
    YC's company directory loads search via Algolia. The HTML embeds
    ``window.AlgoliaOpts = {"app":"...","key":"..."}`` (search-only key).
    """
    m = re.search(r"window\.AlgoliaOpts\s*=\s*(\{[^;]+\});", html)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
        app, key = obj.get("app"), obj.get("key")
        if app and key:
            return str(app), str(key)
    except json.JSONDecodeError:
        pass
    return None


def _search_yc_via_algolia(
    query: str,
    limit: int,
    app_id: str,
    api_key: str,
) -> list[dict]:
    """
    Queries the same Algolia index the YC website uses (public search-only key).
    """
    endpoint = f"https://{app_id}-dsn.algolia.net/1/indexes/YCCompany_production/query"
    payload = json.dumps({
        "params": f"query={urllib.parse.quote(query)}&hitsPerPage={min(limit, 50)}",
    }).encode("utf-8")

    headers = {
        "X-Algolia-Application-Id": app_id,
        "X-Algolia-API-Key": api_key,
        "Content-Type": "application/json",
        "User-Agent": _HEADERS["User-Agent"],
        "Accept": "application/json",
    }

    try:
        resp = requests.post(endpoint, data=payload, headers=headers, timeout=20)
    except requests.exceptions.RequestException as e:
        logger.error("Algolia request failed: %s", e)
        return []

    if not resp.ok:
        logger.warning("Algolia HTTP %s: %s", resp.status_code, resp.text[:200])
        return []

    try:
        data = resp.json()
    except json.JSONDecodeError:
        return []

    hits = data.get("hits") or []
    logger.debug("Algolia returned %d hits for query=%r", len(hits), query)
    return hits


def _fetch_yc_page(query: str) -> str:
    """
    Fetches the YC companies search page for a given query and returns
    the raw HTML string.

    URL pattern: https://www.ycombinator.com/companies?q={query}
    This triggers YC's server-side rendering of matching companies,
    which embeds the results in __NEXT_DATA__ JSON.

    Raises RuntimeError on HTTP or network errors.
    """
    url = f"{_COMPANIES_URL}?{urllib.parse.urlencode({'q': query})}"
    logger.debug("Fetching YC page: %s", url)

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=15)
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(f"Network error reaching YC: {e}") from e
    except requests.exceptions.Timeout:
        raise RuntimeError("YC request timed out after 15 seconds")

    if resp.status_code == 403:
        raise RuntimeError(
            "YC returned 403 Forbidden. The scraper may have been blocked. "
            "Try again later or check User-Agent."
        )
    if not resp.ok:
        raise RuntimeError(f"YC returned HTTP {resp.status_code}")

    return resp.text


def _extract_next_data(html: str) -> dict:
    """
    Extracts the __NEXT_DATA__ JSON block embedded in a Next.js page.

    Next.js apps embed their server-side rendered state as a JSON object
    inside a <script id="__NEXT_DATA__" type="application/json"> tag.
    This contains the full page props — including the company list —
    without requiring JavaScript execution.

    Returns the parsed JSON dict, or an empty dict if not found.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Next.js always uses id="__NEXT_DATA__" — this is a stable contract
    script_tag = soup.find("script", {"id": "__NEXT_DATA__"})
    if not script_tag or not script_tag.string:
        logger.warning("__NEXT_DATA__ script tag not found — page structure may have changed")
        return {}

    try:
        return json.loads(script_tag.string)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse __NEXT_DATA__ JSON: %s", e)
        return {}


def _extract_companies_from_data(next_data: dict) -> list[dict]:
    """
    Navigates the __NEXT_DATA__ structure to find the companies list.

    YC's page props structure can vary between deploys. We try the most
    commonly observed paths in order, failing gracefully if none match.

    Known paths observed in YC's Next.js data:
        props.pageProps.companies
        props.pageProps.companiesForDisplay
        props.pageProps.initialCompanies

    Returns a list of raw company dicts, or [] if none found.
    """
    page_props = (
        next_data
        .get("props", {})
        .get("pageProps", {})
    )

    # Try each known key in priority order
    for key in ("companies", "companiesForDisplay", "initialCompanies"):
        companies = page_props.get(key)
        if isinstance(companies, list) and companies:
            logger.debug("Found companies under pageProps.%s (%d entries)", key, len(companies))
            return companies

    # Last resort: recursively scan pageProps values for a list of dicts
    # that look like company objects (have a "name" and "batch" field)
    for value in page_props.values():
        if isinstance(value, list) and value and isinstance(value[0], dict):
            if "name" in value[0] and "batch" in value[0]:
                logger.debug("Found companies via fallback scan (%d entries)", len(value))
                return value

    logger.warning(
        "Could not find company list in __NEXT_DATA__. "
        "pageProps keys: %s", list(page_props.keys())
    )
    return []


def _parse_company(raw: dict, fetched_at: str, lineage: dict) -> dict:
    """
    Normalises a raw company dict from __NEXT_DATA__ into a consistent
    output shape, handling missing or differently-named fields gracefully.

    YC uses several field name variants across different page versions:
        description: "one_liner" or "tagline" or "short_description"
        status:      "status" (values: "Active", "Acquired", "Inactive")
        tags:        "tags" (list of strings) or "industries" or "verticals"
    """
    name = raw.get("name", "")
    slug = raw.get("slug", "")

    # Description — try multiple known field names
    description = (
        raw.get("one_liner")
        or raw.get("tagline")
        or raw.get("short_description")
        or raw.get("long_description", "")[:200]  # Truncate long descriptions
        or ""
    )

    # Batch code e.g. "W24", "S23" — used for temporal analysis
    batch = raw.get("batch", "")

    # Status normalisation — YC uses "Active", "Acquired", "Inactive"
    status_raw = raw.get("status", "")
    status = status_raw.lower() if status_raw else "unknown"

    # Tags / industry categories
    tags = raw.get("tags") or raw.get("industries") or raw.get("verticals") or []
    if isinstance(tags, list):
        tags = [str(t) for t in tags]

    # Company website URL
    website = raw.get("website") or raw.get("url") or ""

    # YC profile URL
    yc_url = f"{_BASE_URL}/companies/{slug}" if slug else ""

    return {
        "name":        name,
        "description": description.strip(),
        "batch":       batch,
        "status":      status,
        "tags":        tags,
        "website":     website,
        "yc_url":      yc_url,
        # Provenance fields
        "source":      "ycombinator",
        "fetched_at":  fetched_at,
        "lineage":     lineage,
    }


def _batch_sort_key(batch: str) -> int:
    """
    Converts a YC batch code (e.g. "W24", "S23") to a sortable integer.

    Encoding: year * 10 + season_offset (W=1, S=2)
    W24 → 241, S24 → 242 (S comes after W in the same calendar year)
    Unknown batch codes sort to 0 (oldest).
    """
    if not batch or len(batch) < 3:
        return 0
    season = batch[0].upper()
    try:
        year = int(batch[1:])
    except ValueError:
        return 0
    return year * 10 + _SEASON_OFFSET.get(season, 0)


# ---------------------------------------------------------------------------
# Core tool functions
# ---------------------------------------------------------------------------

def search_yc_companies(query: str, limit: int = 20) -> list[dict]:
    """
    Searches the YC company directory for companies matching `query` and
    returns structured company metadata.

    Stage 2 signal: YC-backed companies are the canonical "this is real"
    startup signal. Finding 10+ YC companies in a technology space means
    that space has been validated by the world's most selective accelerator.
    The batch field shows *when* YC started believing in the space.

    Args:
        query : Technology keyword to search for in company names,
                descriptions, and tags.
        limit : Max companies to return (capped at 50).

    Returns:
        List of company dicts sorted by batch (most recent first), each
        containing: name, description, batch, status, tags, website,
        yc_url, source, fetched_at, lineage

    Raises:
        PermissionError : If robots.txt disallows scraping.
        RuntimeError    : On HTTP or network errors.
    """
    limit = max(1, min(limit, 50))

    logger.info("yc_search_companies called | query=%r | limit=%d", query, limit)

    # Always check robots.txt before scraping
    target_url = f"{_COMPANIES_URL}?q={urllib.parse.quote(query)}"
    if not _is_scraping_allowed(target_url):
        raise PermissionError(
            f"robots.txt disallows scraping {target_url}. Aborting."
        )

    lineage = _build_prov_record("yc_search_companies", query, limit=limit)
    fetched_at = datetime.now(timezone.utc).isoformat()

    html = _fetch_yc_page(query)

    # Polite delay — even for a single request, honour the configured delay
    # so rapid consecutive calls from a pipeline don't hammer the server
    time.sleep(_REQUEST_DELAY)

    raw_companies: list[dict] = []
    creds = _extract_algolia_credentials(html)
    if creds:
        app_id, api_key = creds
        raw_companies = _search_yc_via_algolia(query, limit, app_id, api_key)

    if not raw_companies:
        next_data = _extract_next_data(html)
        raw_companies = _extract_companies_from_data(next_data)

    if not raw_companies:
        logger.info("yc_search_companies: zero companies found for query=%r", query)
        return []

    companies = [
        _parse_company(raw, fetched_at, lineage)
        for raw in raw_companies
    ]

    # Sort by batch recency (newest first) so the most recent YC activity
    # for this technology appears at the top
    companies.sort(key=lambda c: _batch_sort_key(c["batch"]), reverse=True)

    companies = companies[:limit]

    logger.info(
        "yc_search_companies returned %d companies for query=%r", len(companies), query
    )
    return companies


def get_yc_stats(query: str) -> dict:
    """
    Returns aggregate statistics about YC's funding activity in a technology
    space by analysing all matching portfolio companies.

    Stage 2 signal: aggregate stats distil the YC signal into a scorable
    snapshot. total_matches shows how many bets YC has placed. earliest_batch
    / latest_batch shows the temporal span — a wide span with recent activity
    means the space is still actively funded. active_count vs acquired_count
    tells you whether companies in the space are still building (active) or
    have been absorbed by incumbents (acquired) — both are positive signals
    but at different pipeline stages.

    Args:
        query : Technology keyword to aggregate YC statistics for.

    Returns:
        dict with keys:
            query, total_matches, batches_represented (sorted list),
            earliest_batch, latest_batch, active_count, acquired_count,
            inactive_count, top_tags (most common tags in the space),
            source, fetched_at, lineage

    Raises:
        PermissionError : If robots.txt disallows scraping.
        RuntimeError    : On HTTP or network errors.
    """
    logger.info("yc_get_stats called | query=%r", query)

    # Fetch up to 50 companies for a representative stats sample
    companies = search_yc_companies(query=query, limit=50)

    lineage = _build_prov_record("yc_get_stats", query)
    fetched_at = datetime.now(timezone.utc).isoformat()

    if not companies:
        return {
            "query":               query,
            "total_matches":       0,
            "batches_represented": [],
            "earliest_batch":      None,
            "latest_batch":        None,
            "active_count":        0,
            "acquired_count":      0,
            "inactive_count":      0,
            "top_tags":            [],
            "source":              "ycombinator",
            "fetched_at":          fetched_at,
            "lineage":             lineage,
        }

    # Collect all batches and sort chronologically
    batches = sorted(
        {c["batch"] for c in companies if c["batch"]},
        key=_batch_sort_key,
    )

    # Status counts
    active_count   = sum(1 for c in companies if c["status"] == "active")
    acquired_count = sum(1 for c in companies if c["status"] == "acquired")
    inactive_count = sum(1 for c in companies if c["status"] in ("inactive", "dead"))

    # Top tags: count occurrences across all companies, return most common
    tag_counts: dict[str, int] = {}
    for company in companies:
        for tag in company["tags"]:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    top_tags = sorted(tag_counts, key=tag_counts.get, reverse=True)[:10]  # type: ignore[arg-type]

    stats = {
        "query":               query,
        "total_matches":       len(companies),
        "batches_represented": batches,
        "earliest_batch":      batches[0]  if batches else None,
        "latest_batch":        batches[-1] if batches else None,
        "active_count":        active_count,
        "acquired_count":      acquired_count,
        "inactive_count":      inactive_count,
        "top_tags":            top_tags,
        # Provenance fields
        "source":              "ycombinator",
        "fetched_at":          fetched_at,
        "lineage":             lineage,
    }

    logger.info(
        "yc_get_stats | query=%r | total=%d | batches=%d | active=%d | acquired=%d",
        query, len(companies), len(batches), active_count, acquired_count,
    )
    return stats


# ---------------------------------------------------------------------------
# MCP-style tool metadata — used by agents for tool discovery
# ---------------------------------------------------------------------------

TOOL_DEFINITION_SEARCH = {
    "name": "yc_search_companies",
    "description": (
        "Search Y Combinator's portfolio for companies in a technology space. "
        "Returns company names, descriptions, batch codes, status, and tags. "
        "Use this as a Stage 2 startup-validation signal — YC backing confirms "
        "a concept has moved from GitHub to real businesses with investor conviction."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Technology keyword to search YC portfolio for.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum companies to return.",
                "default": 20,
                "minimum": 1,
                "maximum": 50,
            },
        },
        "required": ["query"],
    },
}

TOOL_DEFINITION_STATS = {
    "name": "yc_get_stats",
    "description": (
        "Aggregate YC funding statistics for a technology space. Returns total "
        "company count, batch timeline (earliest to latest), active/acquired counts, "
        "and top industry tags. Use this for a single Stage 2 numeric signal."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Technology keyword to aggregate YC statistics for.",
            },
        },
        "required": ["query"],
    },
}

TOOL_DEFINITIONS = [TOOL_DEFINITION_SEARCH, TOOL_DEFINITION_STATS]


# ---------------------------------------------------------------------------
# Quick smoke-test — run directly to verify scraping and output shape
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    QUERY = "AI infrastructure"

    print(f"\n=== search_yc_companies: '{QUERY}' ===")
    companies = search_yc_companies(query=QUERY, limit=5)
    for i, co in enumerate(companies, 1):
        print(f"\n--- Company {i} ---")
        print(f"Name        : {co['name']}")
        print(f"Description : {co['description'][:80]}...")
        print(f"Batch       : {co['batch']}")
        print(f"Status      : {co['status']}")
        print(f"Tags        : {', '.join(co['tags'][:4])}")
        print(f"YC URL      : {co['yc_url']}")
        print(f"Lineage     : {co['lineage']['prov:wasGeneratedBy']['prov:label']}")

    print(f"\n=== get_yc_stats: '{QUERY}' ===")
    stats = get_yc_stats(query=QUERY)
    print(f"Total matches      : {stats['total_matches']}")
    print(f"Batches            : {', '.join(stats['batches_represented'])}")
    print(f"Earliest batch     : {stats['earliest_batch']}")
    print(f"Latest batch       : {stats['latest_batch']}")
    print(f"Active             : {stats['active_count']}")
    print(f"Acquired           : {stats['acquired_count']}")
    print(f"Inactive           : {stats['inactive_count']}")
    print(f"Top tags           : {', '.join(stats['top_tags'][:5])}")
    print(f"Fetched at         : {stats['fetched_at']}")
    print(f"Lineage            : {stats['lineage']['prov:wasGeneratedBy']['prov:label']}")
