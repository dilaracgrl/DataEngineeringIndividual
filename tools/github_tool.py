"""
Tool Name: github_search
Description: Searches GitHub repositories matching a query using the GitHub
             REST API. Returns structured repository metadata and aggregated
             developer activity metrics.

             Stage 2 signal — Developer Community Activity.
             After a concept leaves academic papers (Stage 1), the first sign
             it is gaining real traction is developers building things with it.
             GitHub is where that happens: repo count, star velocity, fork
             count, and recency of new repos are reliable indicators that a
             technology has moved from "research curiosity" to "people are
             actually building with this". A sudden spike in new repos with
             growing star counts is the canonical early-developer-adoption
             signal before a technology reaches startup or VC attention.

Parameters (search_repositories):
    query (str) : Technology or concept to search GitHub for.
    sort  (str) : Sort order — "stars", "forks", "updated", "best-match".
                  Default: "stars".
    limit (int) : Maximum repositories to return (1–30). Default: 10.

Parameters (get_repo_activity):
    query (str) : Technology or concept to measure activity for.
    limit (int) : Number of top repos to aggregate over (1–30). Default: 10.

Returns:
    search_repositories → List[dict]: Each dict contains repo fields plus
                                      W3C PROV lineage metadata.
    get_repo_activity   → dict: Aggregated activity metrics plus lineage.

MCP Schema (search_repositories):
    {
        "name": "github_search_repositories",
        "description": "Search GitHub repositories by keyword. Returns repo metadata including stars, forks, language, and dates.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "sort":  {"type": "string", "enum": ["stars","forks","updated","best-match"], "default": "stars"},
                "limit": {"type": "integer", "default": 10, "minimum": 1, "maximum": 30}
            },
            "required": ["query"]
        }
    }

MCP Schema (get_repo_activity):
    {
        "name": "github_repo_activity",
        "description": "Aggregate developer activity metrics for a technology on GitHub.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 10, "minimum": 1, "maximum": 30}
            },
            "required": ["query"]
        }
    }

Required .env keys:
    GITHUB_TOKEN  — Personal access token from github.com/settings/tokens
                    (read:public_repo scope is sufficient).
                    Without a token the API allows only 10 req/hour.
                    With a token: 30 req/min (search) / 5000 req/hour (core).

API Reference: https://docs.github.com/en/rest/search/search#search-repositories
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
logger = logging.getLogger("github_tool")

_SEARCH_URL = "https://api.github.com/search/repositories"
_VALID_SORT = {"stars", "forks", "updated", "best-match"}


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------

def _get_headers() -> dict:
    """
    Builds the HTTP headers for GitHub API requests.

    GITHUB_TOKEN is optional but strongly recommended:
    - Without token : 10 search requests/hour (unauthenticated limit)
    - With token    : 30 search requests/min, 5 000 core requests/hour

    The token requires only the `read:public_repo` scope — it never writes.
    """
    token = os.getenv("GITHUB_TOKEN")
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "TechPipelineTracker/1.0 (research tool)",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    else:
        logger.warning(
            "GITHUB_TOKEN not set — API rate limit is 10 req/hour. "
            "Set GITHUB_TOKEN in .env for 30 req/min."
        )
    return headers


# ---------------------------------------------------------------------------
# W3C PROV lineage helper
# ---------------------------------------------------------------------------

def _build_prov_record(activity_label: str, query: str, **kwargs) -> dict:
    """
    Builds a W3C PROV-inspired lineage record for a GitHub API fetch.

    activity_label distinguishes between the two tool operations
    (search_repositories vs get_repo_activity) in the lineage trail.
    """
    return {
        "prov:type": "prov:Entity",
        "prov:wasGeneratedBy": {
            "prov:type": "prov:Activity",
            "prov:label": activity_label,
            "prov:startedAtTime": datetime.now(timezone.utc).isoformat(),
            "prov:used": {
                "source": "GitHub REST API v3",
                "endpoint": _SEARCH_URL,
                "query": query,
                **kwargs,
            },
        },
        "prov:wasAttributedTo": {
            "prov:type": "prov:Agent",
            "prov:label": "TechPipelineTracker:github_tool",
        },
    }


# ---------------------------------------------------------------------------
# Internal fetch helper
# ---------------------------------------------------------------------------

def _fetch_repos(query: str, sort: str, limit: int) -> list[dict]:
    """
    Performs the raw GitHub search API call and returns a list of raw repo
    dicts from the response. Shared by both public functions.

    GitHub's search endpoint returns up to 100 items per page; we cap at 30
    to stay within the rate limit budget per call.
    """
    params = {
        "q": query,
        "sort": sort if sort != "best-match" else None,  # omit for best-match
        "order": "desc",
        "per_page": limit,
        "page": 1,
    }
    # Remove None values — GitHub ignores unknown params but keep it clean
    params = {k: v for k, v in params.items() if v is not None}

    try:
        resp = requests.get(
            _SEARCH_URL,
            headers=_get_headers(),
            params=params,
            timeout=15,
        )
    except requests.exceptions.ConnectionError as e:
        logger.error("GitHub network error: %s", e)
        raise RuntimeError(f"Network error reaching GitHub API: {e}") from e
    except requests.exceptions.Timeout:
        logger.error("GitHub API request timed out")
        raise RuntimeError("GitHub API request timed out after 15 seconds")

    if resp.status_code == 401:
        raise RuntimeError(
            "GitHub API authentication failed. Check GITHUB_TOKEN in .env."
        )
    if resp.status_code == 403:
        # GitHub returns 403 for rate limit exhaustion, not 429
        reset_ts = resp.headers.get("X-RateLimit-Reset", "unknown")
        raise RuntimeError(
            f"GitHub API rate limit exceeded. Resets at Unix timestamp {reset_ts}. "
            "Set GITHUB_TOKEN in .env for higher limits."
        )
    if resp.status_code == 422:
        raise ValueError(
            f"GitHub rejected the query '{query}'. "
            "Check for unsupported characters or syntax."
        )
    if not resp.ok:
        raise RuntimeError(
            f"GitHub API returned HTTP {resp.status_code}: {resp.text[:200]}"
        )

    data = resp.json()
    return data.get("items", [])


# ---------------------------------------------------------------------------
# Core tool functions
# ---------------------------------------------------------------------------

def search_repositories(
    query: str,
    sort: str = "stars",
    limit: int = 10,
) -> list[dict]:
    """
    Searches GitHub for repositories matching `query` and returns structured
    metadata for each repo.

    Stage 2 signal: repo count and star growth indicate whether the developer
    community has started building around a technology concept. A cluster of
    recently created repos with rising stars is the clearest early-adoption
    signal before the technology appears in startup or VC press.

    Args:
        query : Technology or concept to search for. GitHub searches across
                repo name, description, and README content.
        sort  : "stars"      — highest star count first (popularity)
                "forks"      — most forked first (community engagement)
                "updated"    — most recently active first (momentum)
                "best-match" — GitHub's relevance ranking
        limit : Max repos to return (capped at 30).

    Returns:
        List of repo dicts, each containing:
            name, full_name, description, stars, forks, watchers,
            language, created_at, updated_at, url, owner,
            topics, open_issues, is_fork, source, fetched_at, lineage

    Raises:
        ValueError  : On invalid sort value or rejected query.
        RuntimeError: On HTTP, network, or rate-limit errors.
    """
    if sort not in _VALID_SORT:
        raise ValueError(
            f"sort must be one of {_VALID_SORT}, got '{sort}'"
        )
    limit = max(1, min(limit, 30))

    logger.info(
        "github_search_repositories called | query=%r | sort=%s | limit=%d",
        query, sort, limit,
    )

    lineage = _build_prov_record(
        "github_search_repositories", query, sort=sort, limit=limit
    )
    fetched_at = datetime.now(timezone.utc).isoformat()

    raw_items = _fetch_repos(query, sort, limit)

    repos = []
    for item in raw_items:
        repo = {
            "name":         item.get("name", ""),
            "full_name":    item.get("full_name", ""),
            "description":  item.get("description") or "",
            "stars":        item.get("stargazers_count", 0),
            "forks":        item.get("forks_count", 0),
            "watchers":     item.get("watchers_count", 0),
            "language":     item.get("language") or "",
            "created_at":   item.get("created_at", ""),
            "updated_at":   item.get("updated_at", ""),
            "url":          item.get("html_url", ""),
            "owner":        item.get("owner", {}).get("login", ""),
            "owner_type":   item.get("owner", {}).get("type", ""),  # "User" or "Organization"
            "topics":       item.get("topics", []),
            "open_issues":  item.get("open_issues_count", 0),
            "is_fork":      item.get("fork", False),
            # Provenance fields
            "source":       "github",
            "fetched_at":   fetched_at,
            "lineage":      lineage,
        }
        repos.append(repo)

    logger.info(
        "github_search_repositories returned %d repos for query=%r",
        len(repos), query,
    )
    return repos


def get_repo_activity(
    query: str,
    limit: int = 10,
) -> dict:
    """
    Measures the overall developer activity around a technology by aggregating
    metadata across the top GitHub repositories matching `query`.

    Stage 2 signal: rather than returning individual repos, this function
    distils the signal into a single activity snapshot — useful for pipeline
    stage scoring. High average stars + many repos + recent creation date =
    strong developer-phase signal. Low counts with old dates = either too
    early (concept hasn't been built yet) or too late (superseded).

    Metrics computed:
        total_repos_found : Number of repos GitHub estimates match the query.
                            (GitHub reports this even when paginated to limit.)
        repos_sampled     : Actual repos retrieved for aggregation.
        avg_stars         : Mean star count across sampled repos.
        avg_forks         : Mean fork count across sampled repos.
        total_stars       : Sum of all stars in the sample.
        newest_repo_date  : ISO date of the most recently created repo.
        oldest_repo_date  : ISO date of the earliest repo in the sample.
        most_starred_repo : {name, url, stars} of the top repo.
        languages         : Sorted list of programming languages represented.

    Args:
        query : Technology or concept to measure activity for.
        limit : Number of top repos to aggregate over.

    Returns:
        dict with activity metrics plus source, fetched_at, lineage.

    Raises:
        ValueError  : On invalid limit.
        RuntimeError: On HTTP, network, or rate-limit errors.
    """
    limit = max(1, min(limit, 30))

    logger.info(
        "github_repo_activity called | query=%r | limit=%d", query, limit
    )

    lineage = _build_prov_record(
        "github_repo_activity", query, limit=limit
    )
    fetched_at = datetime.now(timezone.utc).isoformat()

    # Fetch by stars to get the most influential repos for aggregation
    raw_items = _fetch_repos(query, sort="stars", limit=limit)

    if not raw_items:
        logger.info("github_repo_activity: zero repos found for query=%r", query)
        return {
            "query":              query,
            "total_repos_found":  0,
            "repos_sampled":      0,
            "avg_stars":          0,
            "avg_forks":          0,
            "total_stars":        0,
            "newest_repo_date":   None,
            "oldest_repo_date":   None,
            "most_starred_repo":  None,
            "languages":          [],
            "source":             "github",
            "fetched_at":         fetched_at,
            "lineage":            lineage,
        }

    stars_list  = [r.get("stargazers_count", 0) for r in raw_items]
    forks_list  = [r.get("forks_count", 0)      for r in raw_items]
    dates       = sorted(
        [r.get("created_at", "") for r in raw_items if r.get("created_at")],
    )
    languages   = sorted({
        r.get("language") for r in raw_items if r.get("language")
    })

    # Most starred repo summary — the "face" of this technology on GitHub
    top = max(raw_items, key=lambda r: r.get("stargazers_count", 0))
    most_starred = {
        "name":  top.get("full_name", ""),
        "url":   top.get("html_url", ""),
        "stars": top.get("stargazers_count", 0),
    }

    n = len(raw_items)

    activity = {
        "query":             query,
        "repos_sampled":     n,
        "avg_stars":         round(sum(stars_list) / n, 1),
        "avg_forks":         round(sum(forks_list) / n, 1),
        "total_stars":       sum(stars_list),
        "newest_repo_date":  dates[-1] if dates else None,
        "oldest_repo_date":  dates[0]  if dates else None,
        "most_starred_repo": most_starred,
        "languages":         languages,
        # Provenance fields
        "source":            "github",
        "fetched_at":        fetched_at,
        "lineage":           lineage,
    }

    logger.info(
        "github_repo_activity | query=%r | repos=%d | avg_stars=%.1f | total_stars=%d",
        query, n, activity["avg_stars"], activity["total_stars"],
    )
    return activity


# ---------------------------------------------------------------------------
# MCP-style tool metadata — used by agents for tool discovery
# ---------------------------------------------------------------------------

TOOL_DEFINITION_SEARCH = {
    "name": "github_search_repositories",
    "description": (
        "Search GitHub repositories by keyword. Returns repo metadata including "
        "name, description, star count, fork count, programming language, creation "
        "date, and owner. Use this as a Stage 2 developer-activity signal — a high "
        "count of recently created, starred repos indicates a technology has left "
        "the research phase and developers are actively building with it."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Technology or concept to search GitHub for.",
            },
            "sort": {
                "type": "string",
                "description": (
                    "'stars' = most popular first, 'forks' = most forked, "
                    "'updated' = most recently active, 'best-match' = relevance."
                ),
                "enum": ["stars", "forks", "updated", "best-match"],
                "default": "stars",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of repositories to return.",
                "default": 10,
                "minimum": 1,
                "maximum": 30,
            },
        },
        "required": ["query"],
    },
}

TOOL_DEFINITION_ACTIVITY = {
    "name": "github_repo_activity",
    "description": (
        "Aggregate developer activity metrics for a technology on GitHub. Returns "
        "total estimated repo count, average stars, average forks, newest/oldest "
        "repo dates, most starred repo, and programming languages in use. Use this "
        "for a single numeric Stage 2 signal rather than a full repo list."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Technology or concept to measure GitHub activity for.",
            },
            "limit": {
                "type": "integer",
                "description": "Number of top repos to aggregate over.",
                "default": 10,
                "minimum": 1,
                "maximum": 30,
            },
        },
        "required": ["query"],
    },
}

# Expose both definitions as a list for agents that iterate over all tools
TOOL_DEFINITIONS = [TOOL_DEFINITION_SEARCH, TOOL_DEFINITION_ACTIVITY]


# ---------------------------------------------------------------------------
# Quick smoke-test — run directly to verify token and output shape
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    QUERY = "diffusion models"

    print(f"\n=== search_repositories: '{QUERY}' ===")
    repos = search_repositories(query=QUERY, sort="stars", limit=3)
    for i, repo in enumerate(repos, 1):
        print(f"\n--- Repo {i} ---")
        print(f"Name      : {repo['full_name']}")
        print(f"Stars     : {repo['stars']}")
        print(f"Forks     : {repo['forks']}")
        print(f"Language  : {repo['language']}")
        print(f"Created   : {repo['created_at']}")
        print(f"URL       : {repo['url']}")
        print(f"Lineage   : {repo['lineage']['prov:wasGeneratedBy']['prov:label']}")

    print(f"\n=== get_repo_activity: '{QUERY}' ===")
    activity = get_repo_activity(query=QUERY, limit=10)
    print(f"Repos sampled     : {activity['repos_sampled']}")
    print(f"Avg stars         : {activity['avg_stars']}")
    print(f"Avg forks         : {activity['avg_forks']}")
    print(f"Total stars       : {activity['total_stars']}")
    print(f"Newest repo       : {activity['newest_repo_date']}")
    print(f"Oldest repo       : {activity['oldest_repo_date']}")
    print(f"Most starred      : {activity['most_starred_repo']}")
    print(f"Languages         : {', '.join(activity['languages'])}")
    print(f"Fetched at        : {activity['fetched_at']}")
    print(f"Lineage           : {activity['lineage']['prov:wasGeneratedBy']['prov:label']}")
