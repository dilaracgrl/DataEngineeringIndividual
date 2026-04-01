"""
Tool Name: producthunt_search
Description: Searches Product Hunt for product posts matching a query using the
             Product Hunt GraphQL API v2. Returns structured post data including
             vote counts, topics, and launch dates. Used as the startup-phase
             signal in the Technology Pipeline Tracker — products appearing on
             Product Hunt indicate a concept has moved from research to early
             product/market validation.

Parameters:
    query  (str): Search term or phrase to look up on Product Hunt.
    limit  (int): Maximum number of posts to return (1–20). Default: 10.

Returns:
    List[dict]: Each dict contains post fields plus W3C PROV lineage metadata.

MCP Schema:
    {
        "name": "producthunt_search",
        "description": "Search Product Hunt for product launches by keyword. Returns product names, taglines, vote counts, topics, and launch dates.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 10, "minimum": 1, "maximum": 20}
            },
            "required": ["query"]
        }
    }

Required .env keys (either):
    PRODUCTHUNT_ACCESS_TOKEN — Developer token from https://www.producthunt.com/v2/oauth/applications
    PRODUCTHUNT_CLIENT_ID + PRODUCTHUNT_CLIENT_SECRET — OAuth app credentials; used automatically
        if the developer token is missing or returns HTTP 401 (client_credentials grant).

API Reference: https://api.producthunt.com/v2/docs
"""

import json
import logging
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment & logging setup
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("producthunt_tool")

_GRAPHQL_ENDPOINT = "https://api.producthunt.com/v2/api/graphql"
_OAUTH_TOKEN_URL = "https://api.producthunt.com/v2/oauth/token"


def _env_ph(name: str) -> str:
    """
    Read Product Hunt env vars. Accepts a common typo: ``PR0DUCTHUNT_*`` (digit
    zero) instead of ``PRODUCTHUNT_*`` (letter O) — easy to miss in monospace fonts.
    """
    v = os.getenv(name, "").strip()
    if v:
        return v
    if name.startswith("PRODUCTHUNT"):
        typo = "PR0" + name[3:]
        v = os.getenv(typo, "").strip()
        if v:
            logger.warning(
                "Using %r — rename to %r (the word PRODUCTHUNT uses the letter O, not 0)",
                typo,
                name,
            )
    return v


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------

def _oauth_client_credentials_token() -> str | None:
    """
    Exchanges client_id + client_secret for a short-lived access token.
    See: https://api.producthunt.com/v2/docs/oauth_client_only_authentication/
    """
    cid = _env_ph("PRODUCTHUNT_CLIENT_ID")
    csec = _env_ph("PRODUCTHUNT_CLIENT_SECRET")
    if not cid or not csec:
        return None

    body = json.dumps({
        "client_id": cid,
        "client_secret": csec,
        "grant_type": "client_credentials",
    }).encode("utf-8")

    req = urllib.request.Request(
        _OAUTH_TOKEN_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "TechPipelineTracker/1.0 (research tool)",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        logger.warning(
            "Product Hunt OAuth token exchange failed: HTTP %s — %s",
            e.code,
            (e.read() or b"")[:300],
        )
        return None
    except (urllib.error.URLError, json.JSONDecodeError, OSError) as e:
        logger.warning("Product Hunt OAuth token exchange failed: %s", e)
        return None

    token = data.get("access_token")
    if token:
        logger.info("Using Product Hunt OAuth client_credentials token")
    return token


def _get_access_token() -> str:
    """
    Retrieves a Product Hunt bearer token.

    If both ``PRODUCTHUNT_CLIENT_ID`` and ``PRODUCTHUNT_CLIENT_SECRET`` are set,
    we **prefer** the OAuth ``client_credentials`` token first. That avoids a
    common pitfall: an old or wrong ``PRODUCTHUNT_ACCESS_TOKEN`` being tried
    first and masking valid API Key + Secret from the same app.

    Required .env (one of):
        PRODUCTHUNT_CLIENT_ID + PRODUCTHUNT_CLIENT_SECRET, and/or
        PRODUCTHUNT_ACCESS_TOKEN (developer token from the app page)
    """
    cid = _env_ph("PRODUCTHUNT_CLIENT_ID")
    csec = _env_ph("PRODUCTHUNT_CLIENT_SECRET")
    if cid and csec:
        oauth = _oauth_client_credentials_token()
        if oauth:
            return oauth
        logger.warning(
            "OAuth client_credentials failed (check CLIENT_ID / CLIENT_SECRET); "
            "falling back to PRODUCTHUNT_ACCESS_TOKEN if set"
        )

    raw = _env_ph("PRODUCTHUNT_ACCESS_TOKEN")
    if raw:
        token = raw.strip()
        if token.lower().startswith("bearer "):
            token = token[7:].strip()
        return token

    if cid or csec:
        raise EnvironmentError(
            "PRODUCTHUNT_CLIENT_ID and PRODUCTHUNT_CLIENT_SECRET must both be set "
            "and valid, or set PRODUCTHUNT_ACCESS_TOKEN. "
            "https://www.producthunt.com/v2/oauth/applications"
        )

    raise EnvironmentError(
        "Missing Product Hunt credentials. Set PRODUCTHUNT_ACCESS_TOKEN (developer token) "
        "or PRODUCTHUNT_CLIENT_ID + PRODUCTHUNT_CLIENT_SECRET in .env. "
        "Create an app at https://www.producthunt.com/v2/oauth/applications"
    )


# ---------------------------------------------------------------------------
# W3C PROV lineage helper
# ---------------------------------------------------------------------------

def _build_prov_record(query: str, limit: int) -> dict:
    """
    Builds a W3C PROV-inspired lineage record for this Product Hunt fetch.

    Tracks Entity (fetched dataset), Activity (this GraphQL query),
    and Agent (this tool) so downstream consumers can trace data provenance.
    """
    return {
        "prov:type": "prov:Entity",
        "prov:wasGeneratedBy": {
            "prov:type": "prov:Activity",
            "prov:label": "producthunt_search",
            "prov:startedAtTime": datetime.now(timezone.utc).isoformat(),
            "prov:used": {
                "source": "Product Hunt API v2 (GraphQL)",
                "endpoint": _GRAPHQL_ENDPOINT,
                "query": query,
                "limit": limit,
            },
        },
        "prov:wasAttributedTo": {
            "prov:type": "prov:Agent",
            "prov:label": "TechPipelineTracker:producthunt_tool",
        },
    }


# ---------------------------------------------------------------------------
# GraphQL query builder
# ---------------------------------------------------------------------------

# Product Hunt removed ``posts(query:)`` from the public schema. We map the user
# search string to a topic slug via ``topics(query:)``, then fetch
# ``posts(topic: $slug, ...)``. If no topic matches, we fall back to trending
# posts (``posts(first:, order: VOTES)`` without a topic filter).

_POST_NODE_FIELDS = """
        id
        name
        tagline
        description
        slug
        url
        votesCount
        commentsCount
        createdAt
        featuredAt
        topics {
          edges {
            node {
              name
              slug
            }
          }
        }
        makers {
          name
        }
"""

_TOPIC_SLUG_QUERY = """
query TopicSlug($q: String!) {
  topics(query: $q, first: 1) {
    edges {
      node {
        slug
      }
    }
  }
}
"""

_POSTS_BY_TOPIC_QUERY = f"""
query PostsByTopic($slug: String!, $first: Int!) {{
  posts(topic: $slug, first: $first, order: VOTES) {{
    edges {{
      node {{
        {_POST_NODE_FIELDS.strip()}
      }}
    }}
  }}
}}
"""

_POSTS_TRENDING_QUERY = f"""
query TrendingPosts($first: Int!) {{
  posts(first: $first, order: VOTES) {{
    edges {{
      node {{
        {_POST_NODE_FIELDS.strip()}
      }}
    }}
  }}
}}
"""


def _graphql_request_bytes(bearer_token: str, query: str, variables: dict) -> bytes:
    payload = json.dumps({"query": query, "variables": variables}).encode("utf-8")
    req = urllib.request.Request(
        _GRAPHQL_ENDPOINT,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {bearer_token}",
            "User-Agent": "TechPipelineTracker/1.0 (research tool)",
            "Accept": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read()


def _topic_slug_candidates(user_query: str) -> list[str]:
    """Try several strings so ``topics(query:)`` can find a slug (PH has no full-text post search)."""
    q = user_query.strip()
    if not q:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for cand in (q, *[w for w in q.replace(",", " ").split() if len(w) >= 2]):
        cl = cand.lower()
        if cl not in seen:
            seen.add(cl)
            out.append(cand)
    low = q.lower()
    if "ai" in low.split() or low == "ai":
        for extra in ("artificial intelligence", "developer tools", "saas"):
            if extra not in seen:
                seen.add(extra)
                out.append(extra)
    return out[:20]


def _fetch_topic_slug(token: str, user_query: str) -> str | None:
    for cand in _topic_slug_candidates(user_query):
        try:
            raw = _graphql_request_bytes(
                token, _TOPIC_SLUG_QUERY, {"q": cand}
            )
        except urllib.error.HTTPError:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        edges = (data.get("data") or {}).get("topics", {}).get("edges") or []
        if edges:
            slug = (edges[0].get("node") or {}).get("slug")
            if slug:
                logger.info(
                    "Product Hunt topic slug %r matched from search phrase %r",
                    slug,
                    cand,
                )
                return str(slug)
    return None


def _fetch_posts_graphql(token: str, user_query: str, limit: int) -> bytes:
    slug = _fetch_topic_slug(token, user_query)
    if slug:
        return _graphql_request_bytes(
            token, _POSTS_BY_TOPIC_QUERY, {"slug": slug, "first": limit}
        )
    logger.warning(
        "No Product Hunt topic match for %r — returning trending posts (vote order)",
        user_query,
    )
    return _graphql_request_bytes(
        token, _POSTS_TRENDING_QUERY, {"first": limit}
    )


def _graphql_invalid_oauth_token(errors: list) -> bool:
    """True when PH returns HTTP 200 but GraphQL reports bad bearer (common with stale dev tokens)."""
    for err in errors:
        if isinstance(err, dict) and err.get("error") == "invalid_oauth_token":
            return True
    return False


# ---------------------------------------------------------------------------
# Core tool function
# ---------------------------------------------------------------------------

def search_producthunt(
    query: str,
    limit: int = 10,
) -> list[dict]:
    """
    Fetches Product Hunt posts related to ``query`` and returns structured results.

    The public GraphQL API no longer supports full-text ``posts(query:)``. This
    tool resolves ``query`` to a **topic slug** via ``topics(query:)`` and loads
    ``posts(topic:)``. If no topic matches, it returns **trending** posts (vote order).

    Each result includes the product's core fields plus a W3C PROV lineage block
    so downstream pipeline stages can trace data provenance.

    Product Hunt API v2 uses GraphQL over HTTPS POST. Authentication is via
    an OAuth2 bearer token passed in the Authorization header.

    Args:
        query : Phrase used to match a Product Hunt **topic** (not full-text post search).
        limit : Max posts to return (capped at 20 — PH API page limit).

    Returns:
        List of post dicts, each containing:
            id, name, tagline, description, slug, url, votes_count,
            comments_count, created_at, featured_at, topics, makers,
            source, fetched_at, lineage

    Raises:
        EnvironmentError : If neither developer token nor OAuth client credentials are set.
        RuntimeError     : On HTTP or API-level errors.
    """
    limit = max(1, min(limit, 20))  # PH GraphQL caps first: at 20 per request

    logger.info(
        "producthunt_search called | query=%r | limit=%d",
        query, limit,
    )

    token = _get_access_token()
    lineage = _build_prov_record(query, limit)
    fetched_at = datetime.now(timezone.utc).isoformat()

    try:
        raw = _fetch_posts_graphql(token, query, limit)
    except urllib.error.HTTPError as e:
        err_txt = ""
        try:
            err_txt = (e.read() or b"").decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        if e.code == 401:
            oauth = _oauth_client_credentials_token()
            if oauth and oauth != token:
                logger.warning(
                    "Product Hunt rejected bearer token (401); retrying with client_credentials"
                )
                try:
                    raw = _fetch_posts_graphql(oauth, query, limit)
                except urllib.error.HTTPError as e2:
                    logger.error("Product Hunt API HTTP error %d: %s", e2.code, e2.reason)
                    raise RuntimeError(
                        f"Product Hunt API returned HTTP {e2.code}: {e2.reason} "
                        f"(after OAuth retry). Regenerate your developer token or check "
                        f"CLIENT_ID / CLIENT_SECRET at "
                        f"https://www.producthunt.com/v2/oauth/applications"
                    ) from e2
            else:
                logger.error("Product Hunt API HTTP error 401: %s", err_txt or e.reason)
                raise RuntimeError(
                    "Product Hunt API returned HTTP 401: Unauthorized. "
                    "Regenerate the developer token at "
                    "https://www.producthunt.com/v2/oauth/applications or set "
                    "PRODUCTHUNT_CLIENT_ID and PRODUCTHUNT_CLIENT_SECRET (OAuth app). "
                    f"Details: {err_txt or e.reason}"
                ) from e
        else:
            logger.error("Product Hunt API HTTP error %d: %s", e.code, e.reason)
            raise RuntimeError(
                f"Product Hunt API returned HTTP {e.code}: {e.reason}"
            ) from e
    except urllib.error.URLError as e:
        logger.error("Product Hunt network error: %s", e.reason)
        raise RuntimeError(
            f"Network error reaching Product Hunt API: {e.reason}"
        ) from e

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse Product Hunt JSON response: %s", e)
        raise RuntimeError(f"Product Hunt returned invalid JSON: {e}") from e

    # HTTP 200 with invalid_oauth_token in body — retry with client_credentials if possible
    if isinstance(data.get("errors"), list) and _graphql_invalid_oauth_token(data["errors"]):
        oauth = _oauth_client_credentials_token()
        if oauth and oauth != token:
            logger.warning(
                "Product Hunt GraphQL invalid_oauth_token; retrying with client_credentials"
            )
            try:
                raw = _fetch_posts_graphql(oauth, query, limit)
                data = json.loads(raw)
            except urllib.error.HTTPError as e2:
                logger.error("Product Hunt API HTTP error %d: %s", e2.code, e2.reason)
                raise RuntimeError(
                    f"Product Hunt API returned HTTP {e2.code}: {e2.reason} "
                    f"(after OAuth retry). Check CLIENT_ID / CLIENT_SECRET at "
                    f"https://www.producthunt.com/v2/oauth/applications"
                ) from e2

    # GraphQL surfaces other errors inside the response body
    if "errors" in data:
        errors = data["errors"]
        logger.error("Product Hunt GraphQL errors: %s", errors)
        raise RuntimeError(
            "Product Hunt GraphQL errors (invalid token or query). "
            "Regenerate PRODUCTHUNT_ACCESS_TOKEN or set valid "
            "PRODUCTHUNT_CLIENT_ID + PRODUCTHUNT_CLIENT_SECRET. "
            f"Details: {errors}"
        )

    edges = data.get("data", {}).get("posts", {}).get("edges", [])

    posts = []
    for edge in edges:
        node = edge.get("node", {})

        # Flatten nested topics list
        topics = [
            t["node"]["name"]
            for t in node.get("topics", {}).get("edges", [])
        ]

        # Flatten makers list
        makers = [m.get("name", "") for m in node.get("makers", [])]

        post = {
            "id":             node.get("id"),
            "name":           node.get("name", ""),
            "tagline":        node.get("tagline", ""),
            "description":    node.get("description", ""),
            "slug":           node.get("slug", ""),
            "url":            node.get("url", ""),
            "votes_count":    node.get("votesCount", 0),
            "comments_count": node.get("commentsCount", 0),
            "created_at":     node.get("createdAt", ""),
            "featured_at":    node.get("featuredAt"),   # None if not featured
            "topics":         topics,
            "makers":         makers,
            # Provenance fields
            "source":         "producthunt",
            "fetched_at":     fetched_at,
            "lineage":        lineage,
        }
        posts.append(post)

    logger.info(
        "producthunt_search returned %d posts for query=%r", len(posts), query
    )
    return posts


# ---------------------------------------------------------------------------
# MCP-style tool metadata — used by agents for tool discovery
# ---------------------------------------------------------------------------

TOOL_DEFINITION = {
    "name": "producthunt_search",
    "description": (
        "Search Product Hunt for product launches by keyword. Returns product "
        "names, taglines, vote counts, topics, and launch dates. Use this to "
        "detect whether a technology concept has been commercialised — products "
        "appearing on Product Hunt signal the early startup/product phase."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search term or phrase to find on Product Hunt.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of posts to return.",
                "default": 10,
                "minimum": 1,
                "maximum": 20,
            },
        },
        "required": ["query"],
    },
}


# ---------------------------------------------------------------------------
# Quick smoke-test — run directly to verify credentials and output shape
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = search_producthunt(query="AI assistant", limit=3)
    for i, post in enumerate(results, 1):
        print(f"\n--- Post {i} ---")
        print(f"Name      : {post['name']}")
        print(f"Tagline   : {post['tagline']}")
        print(f"Votes     : {post['votes_count']}")
        print(f"Topics    : {', '.join(post['topics'])}")
        print(f"Launched  : {post['created_at']}")
        print(f"URL       : {post['url']}")
        print(f"Fetched   : {post['fetched_at']}")
        print(f"Lineage   : {post['lineage']['prov:wasGeneratedBy']['prov:label']}")
