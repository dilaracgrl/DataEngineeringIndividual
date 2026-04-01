"""
Tool Name: reddit_search
Description: Searches Reddit for posts matching a query within a given subreddit
             and time window. Returns structured post data for downstream
             analysis and storage. Follows MCP-style tool definition so agents
             can discover and invoke it programmatically.

Parameters:
    query       (str)  : The search term or phrase to look up on Reddit.
    subreddit   (str)  : Target subreddit slug, e.g. "technology". Use "all"
                         to search across all subreddits. Default: "all".
    limit       (int)  : Maximum number of posts to return (1–100). Default: 10.
    time_filter (str)  : Time window for results — one of "hour", "day",
                         "week", "month", "year", "all". Default: "week".

Returns:
    List[dict]: Each dict contains post fields plus W3C PROV lineage metadata.

MCP Schema:
    {
        "name": "reddit_search",
        "description": "Search Reddit posts by keyword within a subreddit and time window.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query":       {"type": "string"},
                "subreddit":   {"type": "string", "default": "all"},
                "limit":       {"type": "integer", "default": 10, "minimum": 1, "maximum": 100},
                "time_filter": {"type": "string", "enum": ["hour","day","week","month","year","all"],
                                "default": "week"}
            },
            "required": ["query"]
        }
    }
"""

import os
import logging
from datetime import datetime, timezone

import praw
import prawcore
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment & logging setup
# ---------------------------------------------------------------------------

load_dotenv()  # Pull API credentials from .env — never hardcode secrets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("reddit_tool")


# ---------------------------------------------------------------------------
# Reddit client initialisation
# ---------------------------------------------------------------------------

def _get_reddit_client() -> praw.Reddit:
    """
    Initialises and returns an authenticated read-only Reddit client using
    credentials stored in .env. Read-only is sufficient for search — we never
    write to Reddit.

    Required .env keys:
        REDDIT_CLIENT_ID
        REDDIT_CLIENT_SECRET
        REDDIT_USER_AGENT   (e.g. "MyApp/0.1 by u/yourname")
    """
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "SocialIntelligenceAgent/1.0")

    if not client_id or not client_secret:
        raise EnvironmentError(
            "Missing Reddit credentials. Ensure REDDIT_CLIENT_ID and "
            "REDDIT_CLIENT_SECRET are set in your .env file."
        )

    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        # read_only=True prevents accidental writes and simplifies auth
        read_only=True,
    )


# ---------------------------------------------------------------------------
# W3C PROV lineage helper
# ---------------------------------------------------------------------------

def _build_prov_record(query: str, subreddit: str, limit: int, time_filter: str) -> dict:
    """
    Builds a W3C PROV-inspired lineage record for this data fetch.

    W3C PROV tracks three core concepts:
        - Entity   : the thing being described (the fetched dataset)
        - Activity : the action that produced it (this search call)
        - Agent    : who/what performed the action (this tool)

    Storing this alongside every fetched record lets the system answer
    "where did this data come from and when?" — essential for auditability
    and hallucination checking.
    """
    return {
        "prov:type": "prov:Entity",
        "prov:wasGeneratedBy": {
            "prov:type": "prov:Activity",
            "prov:label": "reddit_search",
            "prov:startedAtTime": datetime.now(timezone.utc).isoformat(),
            "prov:used": {
                "source": "Reddit API (via PRAW)",
                "query": query,
                "subreddit": subreddit,
                "limit": limit,
                "time_filter": time_filter,
            },
        },
        "prov:wasAttributedTo": {
            "prov:type": "prov:Agent",
            "prov:label": "SocialIntelligenceAgent:reddit_tool",
        },
    }


# ---------------------------------------------------------------------------
# Core tool function
# ---------------------------------------------------------------------------

def search_reddit(
    query: str,
    subreddit: str = "all",
    limit: int = 10,
    time_filter: str = "week",
) -> list[dict]:
    """
    Searches Reddit for posts matching `query` and returns structured results.

    Each result includes the post's core fields plus a W3C PROV lineage block
    so downstream pipeline stages and agents can trace data provenance.

    Args:
        query       : Search term or phrase.
        subreddit   : Subreddit slug to search within, or "all" for global.
        limit       : Max posts to return (capped at 100 by Reddit's API).
        time_filter : Recency filter — "hour" | "day" | "week" (default)
                      | "month" | "year" | "all".

    Returns:
        List of post dicts, each containing:
            title, body, score, url, subreddit, created_utc,
            num_comments, source, fetched_at, lineage

    Raises:
        EnvironmentError  : If Reddit credentials are missing from .env.
        RuntimeError      : On unrecoverable API or network errors.
    """

    # Validate inputs before hitting the API — fail fast with clear messages
    valid_time_filters = {"hour", "day", "week", "month", "year", "all"}
    if time_filter not in valid_time_filters:
        raise ValueError(
            f"Invalid time_filter '{time_filter}'. "
            f"Must be one of: {valid_time_filters}"
        )

    limit = max(1, min(limit, 100))  # Clamp to Reddit's allowed range

    logger.info(
        "reddit_search called | query=%r | subreddit=%r | limit=%d | time_filter=%s",
        query, subreddit, limit, time_filter,
    )

    # Build the lineage record once — attached to every post returned
    lineage = _build_prov_record(query, subreddit, limit, time_filter)
    fetched_at = datetime.now(timezone.utc).isoformat()

    try:
        reddit = _get_reddit_client()
        target = reddit.subreddit(subreddit)

        # praw's search() maps to Reddit's native search endpoint.
        # sort="relevance" ranks by match quality; time_filter scopes recency.
        raw_results = target.search(
            query=query,
            sort="relevance",
            time_filter=time_filter,
            limit=limit,
        )

        posts = []
        for submission in raw_results:
            post = {
                # Core content fields
                "title":        submission.title,
                "body":         submission.selftext or "",  # Empty for link posts
                "score":        submission.score,           # Net upvotes
                "url":          submission.url,
                "subreddit":    submission.subreddit.display_name,
                "created_utc":  datetime.fromtimestamp(
                                    submission.created_utc, tz=timezone.utc
                                ).isoformat(),
                "num_comments": submission.num_comments,
                # Provenance fields — every record knows its origin
                "source":       "reddit",
                "fetched_at":   fetched_at,
                "lineage":      lineage,
            }
            posts.append(post)

        logger.info("reddit_search returned %d posts for query=%r", len(posts), query)
        return posts

    except prawcore.exceptions.ResponseException as e:
        # HTTP-level errors: 403 (forbidden), 404 (subreddit not found), etc.
        logger.error("Reddit API HTTP error: %s", e)
        raise RuntimeError(f"Reddit API returned an error: {e}") from e

    except prawcore.exceptions.RequestException as e:
        # Network-level failures: timeouts, DNS errors, etc.
        logger.error("Reddit network error: %s", e)
        raise RuntimeError(f"Network error reaching Reddit API: {e}") from e

    except prawcore.exceptions.TooManyRequests as e:
        # Reddit rate-limits at ~60 requests/minute for OAuth apps.
        # Surface this clearly so the caller can implement back-off.
        logger.warning("Reddit rate limit hit: %s", e)
        raise RuntimeError(
            "Reddit API rate limit exceeded. Wait before retrying."
        ) from e

    except Exception as e:
        # Catch-all: log the full exception for observability, then re-raise
        # as a RuntimeError so callers get a consistent error type.
        logger.exception("Unexpected error in reddit_search: %s", e)
        raise RuntimeError(f"Unexpected error during Reddit search: {e}") from e


# ---------------------------------------------------------------------------
# MCP-style tool metadata — used by agents for tool discovery
# ---------------------------------------------------------------------------

TOOL_DEFINITION = {
    "name": "reddit_search",
    "description": (
        "Search Reddit posts by keyword within a specified subreddit and time "
        "window. Returns post titles, bodies, scores, URLs, and metadata. "
        "Use this to find community discussions, opinions, or trending topics."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search term or phrase to look up on Reddit.",
            },
            "subreddit": {
                "type": "string",
                "description": "Subreddit slug to search within. Use 'all' for global search.",
                "default": "all",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of posts to return.",
                "default": 10,
                "minimum": 1,
                "maximum": 100,
            },
            "time_filter": {
                "type": "string",
                "description": "Time window to filter results by recency.",
                "enum": ["hour", "day", "week", "month", "year", "all"],
                "default": "week",
            },
        },
        "required": ["query"],
    },
}


# ---------------------------------------------------------------------------
# Quick smoke-test — run directly to verify credentials and output shape
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = search_reddit(query="artificial intelligence", subreddit="technology", limit=3)
    for i, post in enumerate(results, 1):
        print(f"\n--- Post {i} ---")
        print(f"Title     : {post['title']}")
        print(f"Subreddit : r/{post['subreddit']}")
        print(f"Score     : {post['score']}")
        print(f"Comments  : {post['num_comments']}")
        print(f"URL       : {post['url']}")
        print(f"Fetched   : {post['fetched_at']}")
        print(f"Lineage   : {post['lineage']['prov:wasGeneratedBy']['prov:label']}")
