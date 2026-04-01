"""
Tool Name: semantic_scholar_search
Description: Searches the Semantic Scholar academic graph for papers matching a
             query and returns structured metadata including citation counts and
             influential citation counts. Used as the primary Stage 1 signal in
             the Technology Pipeline Tracker.

             Stage 1 signal — Academic Influence, not just Volume.
             arXiv tells you how many papers exist on a topic; Semantic Scholar
             tells you which ones matter. A paper with 5 000 citations that is
             still being cited rapidly is a very different signal from 5 000
             papers with 0 citations each. influential_citation_count (papers
             that cite this work AND are themselves highly cited) is Semantic
             Scholar's strongest academic-impact metric — it distinguishes
             foundational work (still in research phase, likely to spawn
             startups) from incremental follow-up work. High influential
             citations on recent papers = concept is academically hot and
             heading toward commercialisation.

Parameters (search_papers):
    query (str) : Search term or phrase for academic paper search.
    limit (int) : Maximum papers to return (1–100). Default: 10.

Parameters (get_citation_velocity):
    query (str) : Technology concept to measure citation momentum for.
    limit (int) : Papers to aggregate over (1–100). Default: 20.

Returns:
    search_papers          → List[dict]: Paper metadata + lineage.
    get_citation_velocity  → dict: Aggregated citation metrics + lineage.

MCP Schema (search_papers):
    {
        "name": "semantic_scholar_search_papers",
        "description": "Search Semantic Scholar for academic papers. Returns titles, abstracts, citation counts, and influential citation counts.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 10, "minimum": 1, "maximum": 100}
            },
            "required": ["query"]
        }
    }

MCP Schema (get_citation_velocity):
    {
        "name": "semantic_scholar_citation_velocity",
        "description": "Aggregate citation momentum metrics for a technology topic on Semantic Scholar.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 20, "minimum": 1, "maximum": 100}
            },
            "required": ["query"]
        }
    }

Optional .env keys:
    S2_API_KEY  — Semantic Scholar API key for higher rate limits.
                  Without key : 100 requests/5 min (unauthenticated).
                  With key    : 1 request/second per key (higher quota).
                  Apply at: https://www.semanticscholar.org/product/api

API Reference: https://api.semanticscholar.org/graph/v1
"""

import logging
import os
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment & logging setup
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("semantic_scholar_tool")

_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

# Fields requested from the API — we fetch only what we need to keep
# response payloads small and parsing fast.
_PAPER_FIELDS = ",".join([
    "paperId",
    "title",
    "abstract",
    "authors",
    "year",
    "citationCount",
    "influentialCitationCount",
    "externalIds",
    "publicationTypes",
    "openAccessPdf",
])


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------

def _get_headers() -> dict:
    """
    Builds HTTP headers for Semantic Scholar API requests.

    S2_API_KEY is optional but raises rate limits significantly:
    - Without key : ~100 requests per 5 minutes shared pool
    - With key    : 1 request/second dedicated quota

    The key is passed as 'x-api-key' header, not as a Bearer token.
    """
    api_key = os.getenv("S2_API_KEY")
    headers = {
        "User-Agent": "TechPipelineTracker/1.0 (research tool)",
        "Accept": "application/json",
    }
    if api_key:
        headers["x-api-key"] = api_key
    else:
        logger.debug(
            "S2_API_KEY not set — using unauthenticated Semantic Scholar rate limits."
        )
    return headers


# ---------------------------------------------------------------------------
# W3C PROV lineage helper
# ---------------------------------------------------------------------------

def _build_prov_record(activity_label: str, query: str, **kwargs) -> dict:
    """
    Builds a W3C PROV-inspired lineage record for a Semantic Scholar fetch.

    activity_label distinguishes between search_papers and
    get_citation_velocity in the lineage trail.
    """
    return {
        "prov:type": "prov:Entity",
        "prov:wasGeneratedBy": {
            "prov:type": "prov:Activity",
            "prov:label": activity_label,
            "prov:startedAtTime": datetime.now(timezone.utc).isoformat(),
            "prov:used": {
                "source": "Semantic Scholar Graph API v1",
                "endpoint": _SEARCH_URL,
                "query": query,
                **kwargs,
            },
        },
        "prov:wasAttributedTo": {
            "prov:type": "prov:Agent",
            "prov:label": "TechPipelineTracker:semantic_scholar_tool",
        },
    }


# ---------------------------------------------------------------------------
# Internal fetch helper
# ---------------------------------------------------------------------------

def _fetch_papers(query: str, limit: int) -> list[dict]:
    """
    Performs the raw Semantic Scholar search API call and returns a list of
    raw paper dicts. Shared by both public functions.

    The API paginates via offset/limit. We fetch a single page capped at
    the requested limit (max 100 per call per API constraints).
    """
    params = {
        "query": query,
        "fields": _PAPER_FIELDS,
        "limit": limit,
        "offset": 0,
    }

    try:
        resp = requests.get(
            _SEARCH_URL,
            headers=_get_headers(),
            params=params,
            timeout=15,
        )
    except requests.exceptions.ConnectionError as e:
        logger.error("Semantic Scholar network error: %s", e)
        raise RuntimeError(
            f"Network error reaching Semantic Scholar API: {e}"
        ) from e
    except requests.exceptions.Timeout:
        logger.error("Semantic Scholar API request timed out")
        raise RuntimeError(
            "Semantic Scholar API request timed out after 15 seconds"
        )

    if resp.status_code == 429:
        raise RuntimeError(
            "Semantic Scholar API rate limit exceeded. "
            "Set S2_API_KEY in .env for a higher quota."
        )
    if not resp.ok:
        raise RuntimeError(
            f"Semantic Scholar API returned HTTP {resp.status_code}: {resp.text[:200]}"
        )

    return resp.json().get("data", [])


# ---------------------------------------------------------------------------
# Core tool functions
# ---------------------------------------------------------------------------

def search_papers(query: str, limit: int = 10) -> list[dict]:
    """
    Searches Semantic Scholar for academic papers matching `query` and returns
    structured metadata for each paper.

    Stage 1 signal: citation_count and influential_citation_count distinguish
    foundational papers that are driving a field forward from low-impact
    incremental work. A cluster of recent papers with rapidly growing
    influential citations indicates a concept is academically hot and on the
    path toward startup/commercialisation activity.

    Args:
        query : Search term or phrase. Semantic Scholar searches across
                titles, abstracts, and full-text where available.
        limit : Max papers to return (capped at 100 per API constraints).

    Returns:
        List of paper dicts, each containing:
            paper_id, title, abstract, authors, year,
            citation_count, influential_citation_count,
            url, doi, pdf_url, publication_types,
            source, fetched_at, lineage

    Raises:
        ValueError  : On invalid limit.
        RuntimeError: On HTTP, network, or rate-limit errors.
    """
    limit = max(1, min(limit, 100))

    logger.info(
        "semantic_scholar_search_papers called | query=%r | limit=%d",
        query, limit,
    )

    lineage = _build_prov_record(
        "semantic_scholar_search_papers", query, limit=limit
    )
    fetched_at = datetime.now(timezone.utc).isoformat()

    raw_papers = _fetch_papers(query, limit)

    papers = []
    for p in raw_papers:
        # Authors is a list of {authorId, name} dicts
        authors = [a.get("name", "") for a in (p.get("authors") or [])]

        # External IDs: DOI, ArXiv ID, etc.
        ext_ids = p.get("externalIds") or {}
        doi = ext_ids.get("DOI", "")
        arxiv_id = ext_ids.get("ArXiv", "")

        # Open-access PDF link if available
        pdf_url = ""
        if p.get("openAccessPdf"):
            pdf_url = p["openAccessPdf"].get("url", "")

        paper_id = p.get("paperId", "")

        paper = {
            "paper_id":                   paper_id,
            "title":                      p.get("title") or "",
            "abstract":                   p.get("abstract") or "",
            "authors":                    authors,
            "year":                       p.get("year"),
            "citation_count":             p.get("citationCount", 0),
            "influential_citation_count": p.get("influentialCitationCount", 0),
            "publication_types":          p.get("publicationTypes") or [],
            "doi":                        doi,
            "arxiv_id":                   arxiv_id,
            "url":                        f"https://www.semanticscholar.org/paper/{paper_id}",
            "pdf_url":                    pdf_url,
            # Provenance fields
            "source":                     "semantic_scholar",
            "fetched_at":                 fetched_at,
            "lineage":                    lineage,
        }
        papers.append(paper)

    logger.info(
        "semantic_scholar_search_papers returned %d papers for query=%r",
        len(papers), query,
    )
    return papers


def get_citation_velocity(query: str, limit: int = 20) -> dict:
    """
    Measures how fast citation counts are accumulating for a technology topic
    by aggregating citation metadata across the top matching papers.

    Stage 1 signal: this function reduces the academic landscape to a single
    scorable snapshot. avg_citations tells you how influential the work is on
    average. total_influential_citations across the sample is the strongest
    single number — it means the field is producing work that other researchers
    consider foundational. year_range tells you whether the concept is brand-new
    or mature. most_cited_paper is the "anchor" paper for the concept.

    Args:
        query : Technology concept to measure citation momentum for.
        limit : Number of papers to aggregate over (fetched by relevance).

    Returns:
        dict with keys:
            query, papers_sampled, avg_citations, avg_influential_citations,
            total_citations, total_influential_citations, year_range,
            most_cited_paper {title, url, citation_count, year},
            source, fetched_at, lineage

    Raises:
        ValueError  : On invalid limit.
        RuntimeError: On HTTP, network, or rate-limit errors.
    """
    limit = max(1, min(limit, 100))

    logger.info(
        "semantic_scholar_citation_velocity called | query=%r | limit=%d",
        query, limit,
    )

    lineage = _build_prov_record(
        "semantic_scholar_citation_velocity", query, limit=limit
    )
    fetched_at = datetime.now(timezone.utc).isoformat()

    raw_papers = _fetch_papers(query, limit)

    if not raw_papers:
        logger.info(
            "semantic_scholar_citation_velocity: zero papers for query=%r", query
        )
        return {
            "query":                       query,
            "papers_sampled":              0,
            "avg_citations":               0,
            "avg_influential_citations":   0,
            "total_citations":             0,
            "total_influential_citations": 0,
            "year_range":                  None,
            "most_cited_paper":            None,
            "source":                      "semantic_scholar",
            "fetched_at":                  fetched_at,
            "lineage":                     lineage,
        }

    citations      = [p.get("citationCount", 0)             for p in raw_papers]
    inf_citations  = [p.get("influentialCitationCount", 0)  for p in raw_papers]
    years          = [p["year"] for p in raw_papers if p.get("year")]
    n              = len(raw_papers)

    # Most cited paper — the reference point for agents citing this analysis
    top = max(raw_papers, key=lambda p: p.get("citationCount", 0))
    top_id = top.get("paperId", "")
    most_cited = {
        "title":          top.get("title") or "",
        "url":            f"https://www.semanticscholar.org/paper/{top_id}",
        "citation_count": top.get("citationCount", 0),
        "year":           top.get("year"),
    }

    year_range = (
        {"earliest": min(years), "latest": max(years)} if years else None
    )

    velocity = {
        "query":                       query,
        "papers_sampled":              n,
        "avg_citations":               round(sum(citations) / n, 1),
        "avg_influential_citations":   round(sum(inf_citations) / n, 1),
        "total_citations":             sum(citations),
        "total_influential_citations": sum(inf_citations),
        "year_range":                  year_range,
        "most_cited_paper":            most_cited,
        # Provenance fields
        "source":                      "semantic_scholar",
        "fetched_at":                  fetched_at,
        "lineage":                     lineage,
    }

    logger.info(
        "semantic_scholar_citation_velocity | query=%r | papers=%d | "
        "avg_citations=%.1f | total_influential=%d",
        query, n, velocity["avg_citations"], velocity["total_influential_citations"],
    )
    return velocity


# ---------------------------------------------------------------------------
# MCP-style tool metadata — used by agents for tool discovery
# ---------------------------------------------------------------------------

TOOL_DEFINITION_SEARCH = {
    "name": "semantic_scholar_search_papers",
    "description": (
        "Search Semantic Scholar for academic papers by keyword. Returns titles, "
        "abstracts, authors, publication year, citation counts, and influential "
        "citation counts. Use this as a Stage 1 academic-influence signal — "
        "high influential_citation_count on recent papers indicates a concept is "
        "foundational and heading toward commercialisation."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search term or phrase for academic paper search.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of papers to return.",
                "default": 10,
                "minimum": 1,
                "maximum": 100,
            },
        },
        "required": ["query"],
    },
}

TOOL_DEFINITION_VELOCITY = {
    "name": "semantic_scholar_citation_velocity",
    "description": (
        "Aggregate citation momentum metrics for a technology topic on Semantic "
        "Scholar. Returns average citations, total influential citations, year range, "
        "and the most cited paper. Use this for a single Stage 1 numeric signal "
        "rather than a full paper list."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Technology concept to measure citation momentum for.",
            },
            "limit": {
                "type": "integer",
                "description": "Number of papers to aggregate over.",
                "default": 20,
                "minimum": 1,
                "maximum": 100,
            },
        },
        "required": ["query"],
    },
}

TOOL_DEFINITIONS = [TOOL_DEFINITION_SEARCH, TOOL_DEFINITION_VELOCITY]


# ---------------------------------------------------------------------------
# Quick smoke-test — run directly to verify connectivity and output shape
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    QUERY = "transformer attention mechanism"

    print(f"\n=== search_papers: '{QUERY}' ===")
    papers = search_papers(query=QUERY, limit=3)
    for i, paper in enumerate(papers, 1):
        print(f"\n--- Paper {i} ---")
        print(f"ID           : {paper['paper_id']}")
        print(f"Title        : {paper['title']}")
        print(f"Year         : {paper['year']}")
        print(f"Citations    : {paper['citation_count']}")
        print(f"Influential  : {paper['influential_citation_count']}")
        print(f"URL          : {paper['url']}")
        print(f"Lineage      : {paper['lineage']['prov:wasGeneratedBy']['prov:label']}")

    print(f"\n=== get_citation_velocity: '{QUERY}' ===")
    vel = get_citation_velocity(query=QUERY, limit=20)
    print(f"Papers sampled        : {vel['papers_sampled']}")
    print(f"Avg citations         : {vel['avg_citations']}")
    print(f"Avg influential       : {vel['avg_influential_citations']}")
    print(f"Total citations       : {vel['total_citations']}")
    print(f"Total influential     : {vel['total_influential_citations']}")
    print(f"Year range            : {vel['year_range']}")
    print(f"Most cited            : {vel['most_cited_paper']}")
    print(f"Fetched at            : {vel['fetched_at']}")
    print(f"Lineage               : {vel['lineage']['prov:wasGeneratedBy']['prov:label']}")
