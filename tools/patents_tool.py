"""
Tool Name: patents_search
Description: Searches the PatentsView database for US patents matching a query.
             Returns structured patent metadata including assignee organisations,
             filing dates, and abstracts. Used as the big-tech/industrialisation
             signal in the Technology Pipeline Tracker — a surge of patents from
             large corporations indicates a technology has moved out of the startup
             phase and into institutional adoption.

Parameters:
    query      (str) : Search term to match against patent titles.
    max_results (int): Maximum patents to return (1–25). Default: 10.
    date_from  (str) : Optional ISO date filter, e.g. "2020-01-01". Default: None.

Returns:
    List[dict]: Each dict contains patent fields plus W3C PROV lineage metadata.

MCP Schema:
    {
        "name": "patents_search",
        "description": "Search US patents via PatentsView by keyword. Returns patent titles, dates, assignee organisations, and abstracts.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query":       {"type": "string"},
                "max_results": {"type": "integer", "default": 10, "minimum": 1, "maximum": 25},
                "date_from":   {"type": "string", "description": "ISO date string, e.g. '2020-01-01'"}
            },
            "required": ["query"]
        }
    }

API Reference: PatentSearch API at search.patentsview.org (USPTO migration may
    interrupt service — see tools/patents_tool.py fallback).

Fallback: when PatentsView returns HTTP 410 or a migration error, we query
Google Patents' public XHR endpoint (same data users see in search) so the
pipeline keeps working without a separate API key.
"""

import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("patents_tool")

_BASE_URL = "https://search.patentsview.org/api/v1/patent/"
_GOOGLE_PATENTS_XHR = "https://patents.google.com/xhr/query"
# Google intermittently returns 503; browser-like UA reduces blocks.
_GOOGLE_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

# Fields to retrieve from PatentsView — we request only what we need to keep
# responses lightweight and parsing straightforward.
_RETURN_FIELDS = [
    "patent_id",
    "patent_title",
    "patent_date",
    "patent_abstract",
    "patent_type",
    "assignees",
    "inventors",
]


# ---------------------------------------------------------------------------
# W3C PROV lineage helper
# ---------------------------------------------------------------------------

def _build_prov_record(query: str, max_results: int, date_from: str | None) -> dict:
    """
    Builds a W3C PROV-inspired lineage record for this PatentsView fetch.

    Tracks Entity (fetched dataset), Activity (this search call),
    and Agent (this tool) for full data provenance.
    """
    return {
        "prov:type": "prov:Entity",
        "prov:wasGeneratedBy": {
            "prov:type": "prov:Activity",
            "prov:label": "patents_search",
            "prov:startedAtTime": datetime.now(timezone.utc).isoformat(),
            "prov:used": {
                "source": "PatentsView API v1",
                "base_url": _BASE_URL,
                "query": query,
                "max_results": max_results,
                "date_from": date_from,
            },
        },
        "prov:wasAttributedTo": {
            "prov:type": "prov:Agent",
            "prov:label": "TechPipelineTracker:patents_tool",
        },
    }


# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------

def _build_query_filter(query: str, date_from: str | None) -> dict:
    """
    Builds the PatentsView JSON query filter object.

    PatentsView uses a nested JSON query language:
    - _text_phrase searches for the phrase in a field
    - _gte / _lte for date range filters
    - _and combines multiple conditions

    We search patent_title with the query text. If a date_from is provided
    we add a _gte filter on patent_date so we only see recent filings.
    """
    title_filter = {"_text_phrase": {"patent_title": query}}

    if not date_from:
        return title_filter

    # Combine title match AND date range
    return {
        "_and": [
            title_filter,
            {"_gte": {"patent_date": date_from}},
        ]
    }


def _strip_html(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"<[^>]+>", "", text).replace("&hellip;", "…").strip()


def _search_patents_google_patents(
    query: str,
    max_results: int,
    date_from: str | None,
    fetched_at: str,
    lineage: dict,
) -> list[dict]:
    """
    Fallback when PatentsView is unavailable (HTTP 410 / USPTO migration).
    Uses Google Patents' xhr/query JSON API (public, no key).
    """
    inner = f"q={urllib.parse.quote(query)}"
    params = urllib.parse.urlencode({"url": inner, "num": min(max_results, 25)})
    url = f"{_GOOGLE_PATENTS_XHR}?{params}"

    raw = b""
    max_attempts = 8
    for attempt in range(max_attempts):
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": _GOOGLE_UA,
                "Accept": "application/json",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
            break
        except urllib.error.HTTPError as e:
            if e.code in (502, 503, 504) and attempt < max_attempts - 1:
                wait = min(30.0, 2.5 * (2 ** attempt))
                logger.warning(
                    "Google Patents HTTP %s — retrying in %.1fs (%d/%d)",
                    e.code,
                    wait,
                    attempt + 1,
                    max_attempts,
                )
                time.sleep(wait)
                continue
            logger.error("Google Patents HTTP error %d: %s", e.code, e.reason)
            raise RuntimeError(
                f"Google Patents fallback returned HTTP {e.code}: {e.reason}"
            ) from e
        except urllib.error.URLError as e:
            if attempt < max_attempts - 1:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise RuntimeError(f"Network error reaching Google Patents: {e.reason}") from e

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Google Patents returned invalid JSON: {e}") from e

    patents: list[dict] = []
    for cluster in data.get("results", {}).get("cluster", []):
        for item in cluster.get("result", []):
            pat = item.get("patent") or {}
            pub = pat.get("publication_number") or pat.get("id", "")
            title = _strip_html(pat.get("title", ""))
            snippet = _strip_html(pat.get("snippet", ""))
            assignee = pat.get("assignee") or ""
            inventor = pat.get("inventor") or ""
            pub_date = pat.get("publication_date") or pat.get("grant_date") or pat.get("filing_date") or ""

            if date_from and pub_date and len(pub_date) >= 10 and len(date_from) >= 10:
                if pub_date[:10] < date_from[:10]:
                    continue

            patents.append({
                "patent_id":      pub,
                "title":          title,
                "date":           pub_date,
                "abstract":       snippet,
                "patent_type":    "",
                "assignee_orgs":  [assignee] if assignee else [],
                "inventor_names": [inventor] if inventor else [],
                "source":         "google_patents",
                "fetched_at":     fetched_at,
                "lineage":        lineage,
            })
            if len(patents) >= max_results:
                return patents

    return patents


def _patentsview_unavailable(body: bytes) -> bool:
    if not body:
        return False
    try:
        err = json.loads(body)
    except json.JSONDecodeError:
        return False
    if err.get("error") is True:
        return True
    msg = (err.get("message") or "").lower()
    return "migrating" in msg or "open data portal" in msg or "discontinued" in str(
        err.get("reason", "")
    ).lower()


# ---------------------------------------------------------------------------
# Core tool function
# ---------------------------------------------------------------------------

def search_patents(
    query: str,
    max_results: int = 10,
    date_from: str | None = None,
) -> list[dict]:
    """
    Searches PatentsView for US patents matching `query` and returns structured
    results.

    Each result includes patent metadata plus a W3C PROV lineage block so
    downstream pipeline stages can trace data provenance.

    PatentsView is a free, open API from USPTO — no authentication required.
    Rate limit: be polite; avoid hammering (no explicit limit published).

    Args:
        query       : Search term matched against patent titles.
        max_results : Max patents to return.
        date_from   : Optional ISO date string (YYYY-MM-DD) to filter patents
                      filed on or after this date.

    Returns:
        List of patent dicts, each containing:
            patent_id, title, date, abstract, patent_type,
            assignee_orgs, inventor_names, source, fetched_at, lineage

    Raises:
        ValueError  : On invalid parameters.
        RuntimeError: On HTTP or network errors.
    """
    max_results = max(1, min(max_results, 25))

    if date_from:
        # Validate date format
        try:
            datetime.strptime(date_from, "%Y-%m-%d")
        except ValueError:
            raise ValueError(
                f"date_from must be in YYYY-MM-DD format, got '{date_from}'"
            )

    logger.info(
        "patents_search called | query=%r | max_results=%d | date_from=%s",
        query, max_results, date_from,
    )

    lineage = _build_prov_record(query, max_results, date_from)
    fetched_at = datetime.now(timezone.utc).isoformat()

    # PatentsView accepts GET requests with URL-encoded JSON parameters.
    # 'q' is the query filter, 'f' is the list of fields to return,
    # 'o' contains pagination/sort options.
    q_filter = _build_query_filter(query, date_from)

    params = urllib.parse.urlencode({
        "q": json.dumps(q_filter),
        "f": json.dumps(_RETURN_FIELDS),
        "o": json.dumps({"per_page": max_results, "page": 1}),
        "s": json.dumps([{"patent_date": "desc"}]),  # Newest patents first
    })
    url = f"{_BASE_URL}?{params}"
    logger.debug("PatentsView request URL: %s", url)

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "TechPipelineTracker/1.0 (research tool)",
            "Accept": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read()
        except Exception:
            err_body = b""
        if e.code == 410 or _patentsview_unavailable(err_body):
            logger.warning(
                "PatentsView unavailable (HTTP %s) — using Google Patents fallback",
                e.code,
            )
            return _search_patents_google_patents(
                query, max_results, date_from, fetched_at, lineage
            )
        logger.error("PatentsView HTTP error %d: %s", e.code, e.reason)
        raise RuntimeError(
            f"PatentsView API returned HTTP {e.code}: {e.reason}"
        ) from e
    except urllib.error.URLError as e:
        logger.error("PatentsView network error: %s", e.reason)
        raise RuntimeError(
            f"Network error reaching PatentsView API: {e.reason}"
        ) from e

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse PatentsView JSON: %s", e)
        raise RuntimeError(f"PatentsView returned invalid JSON: {e}") from e

    if _patentsview_unavailable(raw):
        logger.warning("PatentsView returned migration/discontinued payload — Google Patents fallback")
        return _search_patents_google_patents(
            query, max_results, date_from, fetched_at, lineage
        )

    raw_patents = data.get("patents") or []

    patents = []
    for p in raw_patents:
        # Assignees is a list of objects; extract organisation names
        assignee_orgs = [
            a.get("assignee_organization", "")
            for a in (p.get("assignees") or [])
            if a.get("assignee_organization")
        ]

        # Inventors is a list; extract full names
        inventor_names = [
            f"{inv.get('inventor_first_name', '')} {inv.get('inventor_last_name', '')}".strip()
            for inv in (p.get("inventors") or [])
        ]

        patent = {
            "patent_id":      p.get("patent_id", ""),
            "title":          p.get("patent_title", ""),
            "date":           p.get("patent_date", ""),
            "abstract":       p.get("patent_abstract", ""),
            "patent_type":    p.get("patent_type", ""),
            "assignee_orgs":  assignee_orgs,
            "inventor_names": inventor_names,
            # Provenance fields
            "source":         "patentsview",
            "fetched_at":     fetched_at,
            "lineage":        lineage,
        }
        patents.append(patent)

    logger.info(
        "patents_search returned %d patents for query=%r", len(patents), query
    )
    return patents


# ---------------------------------------------------------------------------
# MCP-style tool metadata — used by agents for tool discovery
# ---------------------------------------------------------------------------

TOOL_DEFINITION = {
    "name": "patents_search",
    "description": (
        "Search US patents via PatentsView by keyword. Returns patent titles, "
        "filing dates, assignee organisations, and abstracts. Use this to detect "
        "institutional adoption — a rising count of patents from large corporations "
        "signals a technology has moved into the big-tech/industrialisation phase."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search term matched against patent titles.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of patents to return.",
                "default": 10,
                "minimum": 1,
                "maximum": 25,
            },
            "date_from": {
                "type": "string",
                "description": (
                    "Optional ISO date filter (YYYY-MM-DD). Only returns patents "
                    "filed on or after this date."
                ),
            },
        },
        "required": ["query"],
    },
}


# ---------------------------------------------------------------------------
# Quick smoke-test — run directly to verify connectivity and output shape
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = search_patents(query="neural network", max_results=3, date_from="2022-01-01")
    for i, patent in enumerate(results, 1):
        print(f"\n--- Patent {i} ---")
        print(f"ID        : {patent['patent_id']}")
        print(f"Title     : {patent['title']}")
        print(f"Date      : {patent['date']}")
        print(f"Assignees : {', '.join(patent['assignee_orgs']) or 'N/A'}")
        print(f"Inventors : {', '.join(patent['inventor_names'][:3]) or 'N/A'}")
        print(f"Fetched   : {patent['fetched_at']}")
        print(f"Lineage   : {patent['lineage']['prov:wasGeneratedBy']['prov:label']}")
