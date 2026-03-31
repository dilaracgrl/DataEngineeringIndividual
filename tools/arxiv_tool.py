"""
Tool Name: arxiv_search
Description: Searches the arXiv preprint repository for academic papers matching
             a query. Returns structured paper metadata for downstream pipeline
             stages. Used as the earliest-stage signal in the Technology Pipeline
             Tracker — a high paper count with recent dates indicates a concept
             is still in the academic/research phase.

Parameters:
    query      (str) : Search term or phrase (searches title + abstract).
    max_results (int): Maximum number of papers to return (1–50). Default: 10.
    sort_by    (str) : Sort order — "relevance" or "submittedDate". Default: "submittedDate".

Returns:
    List[dict]: Each dict contains paper fields plus W3C PROV lineage metadata.

MCP Schema:
    {
        "name": "arxiv_search",
        "description": "Search arXiv for academic papers by keyword. Returns titles, abstracts, authors, dates, and categories.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query":       {"type": "string"},
                "max_results": {"type": "integer", "default": 10, "minimum": 1, "maximum": 50},
                "sort_by":     {"type": "string", "enum": ["relevance", "submittedDate"], "default": "submittedDate"}
            },
            "required": ["query"]
        }
    }

API Reference: https://info.arxiv.org/help/api/basics.html#using
"""

import logging
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("arxiv_tool")

# arXiv Atom feed namespace — all tags in the response are prefixed with this
_ATOM_NS = "http://www.w3.org/2005/Atom"
_ARXIV_NS = "http://arxiv.org/schemas/atom"
_BASE_URL = "http://export.arxiv.org/api/query"


# ---------------------------------------------------------------------------
# W3C PROV lineage helper
# ---------------------------------------------------------------------------

def _build_prov_record(query: str, max_results: int, sort_by: str) -> dict:
    """
    Builds a W3C PROV-inspired lineage record for this arXiv fetch.

    Tracks Entity (fetched dataset), Activity (this search call),
    and Agent (this tool) so downstream consumers can answer
    "where did this paper list come from and when?".
    """
    return {
        "prov:type": "prov:Entity",
        "prov:wasGeneratedBy": {
            "prov:type": "prov:Activity",
            "prov:label": "arxiv_search",
            "prov:startedAtTime": datetime.now(timezone.utc).isoformat(),
            "prov:used": {
                "source": "arXiv API (Atom feed)",
                "base_url": _BASE_URL,
                "query": query,
                "max_results": max_results,
                "sort_by": sort_by,
            },
        },
        "prov:wasAttributedTo": {
            "prov:type": "prov:Agent",
            "prov:label": "TechPipelineTracker:arxiv_tool",
        },
    }


# ---------------------------------------------------------------------------
# XML parsing helpers
# ---------------------------------------------------------------------------

def _tag(local: str) -> str:
    """Returns a fully-qualified Atom tag string for ElementTree queries."""
    return f"{{{_ATOM_NS}}}{local}"


def _parse_entry(entry: ET.Element) -> dict:
    """
    Parses a single <entry> element from the arXiv Atom response into a
    flat dictionary with the fields most useful for pipeline stage scoring.
    """
    def text(tag: str) -> str:
        el = entry.find(_tag(tag))
        return el.text.strip() if el is not None and el.text else ""

    # Authors: <author><name>...</name></author> (may be multiple)
    authors = [
        a.find(_tag("name")).text.strip()
        for a in entry.findall(_tag("author"))
        if a.find(_tag("name")) is not None
    ]

    # Categories: <category term="cs.LG" scheme="..."/>
    categories = [
        cat.get("term", "")
        for cat in entry.findall(_tag("category"))
    ]

    # arXiv ID is embedded in the <id> tag as a URL, e.g.
    # http://arxiv.org/abs/2301.12345v1 — extract just the ID portion
    raw_id = text("id")
    arxiv_id = raw_id.split("/abs/")[-1] if "/abs/" in raw_id else raw_id

    # Published and updated dates
    published = text("published")   # ISO 8601, e.g. "2023-01-23T18:00:00Z"
    updated = text("updated")

    return {
        "arxiv_id":   arxiv_id,
        "title":      text("title").replace("\n", " "),
        "abstract":   text("summary").replace("\n", " "),
        "authors":    authors,
        "categories": categories,
        "published":  published,
        "updated":    updated,
        "url":        f"https://arxiv.org/abs/{arxiv_id}",
        "pdf_url":    f"https://arxiv.org/pdf/{arxiv_id}",
    }


# ---------------------------------------------------------------------------
# Core tool function
# ---------------------------------------------------------------------------

def search_arxiv(
    query: str,
    max_results: int = 10,
    sort_by: str = "submittedDate",
) -> list[dict]:
    """
    Searches arXiv for papers matching `query` and returns structured results.

    Each result includes paper metadata plus a W3C PROV lineage block so
    downstream pipeline stages can trace data provenance.

    The arXiv API uses a simple HTTP GET with query parameters and returns
    an Atom XML feed — no authentication required. Rate limit: ~3 req/sec.

    Args:
        query       : Search term. Searches across title and abstract.
                      Supports arXiv query syntax (e.g. "ti:transformer").
        max_results : Max papers to return (capped at 50 to stay polite).
        sort_by     : "submittedDate" (newest first) or "relevance".

    Returns:
        List of paper dicts, each containing:
            arxiv_id, title, abstract, authors, categories,
            published, updated, url, pdf_url, source, fetched_at, lineage

    Raises:
        ValueError  : On invalid parameters.
        RuntimeError: On HTTP or network errors.
    """
    valid_sort = {"relevance", "submittedDate"}
    if sort_by not in valid_sort:
        raise ValueError(f"sort_by must be one of {valid_sort}, got '{sort_by}'")

    max_results = max(1, min(max_results, 50))  # Stay polite to arXiv servers

    logger.info(
        "arxiv_search called | query=%r | max_results=%d | sort_by=%s",
        query, max_results, sort_by,
    )

    lineage = _build_prov_record(query, max_results, sort_by)
    fetched_at = datetime.now(timezone.utc).isoformat()

    # Build the request URL — arXiv uses search_query with field prefixes.
    # "all:" searches title + abstract + author + comments; most permissive.
    params = urllib.parse.urlencode({
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": "descending",
    })
    url = f"{_BASE_URL}?{params}"
    logger.debug("arXiv request URL: %s", url)

    try:
        # arXiv requires a descriptive User-Agent to identify the client
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "TechPipelineTracker/1.0 (research tool)"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw_xml = resp.read()

    except urllib.error.HTTPError as e:
        logger.error("arXiv HTTP error %d: %s", e.code, e.reason)
        raise RuntimeError(f"arXiv API returned HTTP {e.code}: {e.reason}") from e

    except urllib.error.URLError as e:
        logger.error("arXiv network error: %s", e.reason)
        raise RuntimeError(f"Network error reaching arXiv API: {e.reason}") from e

    # Parse the Atom XML feed
    try:
        root = ET.fromstring(raw_xml)
    except ET.ParseError as e:
        logger.error("Failed to parse arXiv XML response: %s", e)
        raise RuntimeError(f"arXiv returned invalid XML: {e}") from e

    entries = root.findall(_tag("entry"))

    # arXiv returns an error entry when no results are found — detect and skip it
    papers = []
    for entry in entries:
        title_el = entry.find(_tag("title"))
        if title_el is not None and title_el.text and "Error" in title_el.text:
            logger.warning("arXiv returned an error entry — possibly zero results")
            continue

        paper = _parse_entry(entry)
        paper["source"] = "arxiv"
        paper["fetched_at"] = fetched_at
        paper["lineage"] = lineage
        papers.append(paper)

    logger.info("arxiv_search returned %d papers for query=%r", len(papers), query)
    return papers


# ---------------------------------------------------------------------------
# MCP-style tool metadata — used by agents for tool discovery
# ---------------------------------------------------------------------------

TOOL_DEFINITION = {
    "name": "arxiv_search",
    "description": (
        "Search the arXiv preprint repository for academic papers by keyword. "
        "Returns paper titles, abstracts, authors, publication dates, and subject "
        "categories. Use this to measure the academic research activity around a "
        "technology — high recent paper volume signals it is still in the research phase."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search term or phrase. Searched across title and abstract.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of papers to return.",
                "default": 10,
                "minimum": 1,
                "maximum": 50,
            },
            "sort_by": {
                "type": "string",
                "description": "'submittedDate' returns newest papers first; 'relevance' ranks by match quality.",
                "enum": ["relevance", "submittedDate"],
                "default": "submittedDate",
            },
        },
        "required": ["query"],
    },
}


# ---------------------------------------------------------------------------
# Quick smoke-test — run directly to verify connectivity and output shape
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = search_arxiv(query="large language models", max_results=3)
    for i, paper in enumerate(results, 1):
        print(f"\n--- Paper {i} ---")
        print(f"ID         : {paper['arxiv_id']}")
        print(f"Title      : {paper['title']}")
        print(f"Authors    : {', '.join(paper['authors'][:3])}")
        print(f"Published  : {paper['published']}")
        print(f"Categories : {', '.join(paper['categories'])}")
        print(f"URL        : {paper['url']}")
        print(f"Fetched    : {paper['fetched_at']}")
        print(f"Lineage    : {paper['lineage']['prov:wasGeneratedBy']['prov:label']}")
