"""
Tool Name: techcrunch_search
Description: Scrapes TechCrunch search results to find articles about a
             technology, with focus on funding and startup coverage. Returns
             structured article metadata and a temporal coverage timeline.

             Stage 3 signal — VC Funding Press Coverage.
             TechCrunch is the canonical venue where VC-backed startups
             announce funding rounds. A Series A or B announcement almost
             always appears on TechCrunch within days of closing. This makes
             TechCrunch article volume a direct proxy for investment activity
             in a space. The coverage *timeline* is more informative than
             volume alone: the date of the earliest TechCrunch article about
             a technology tells you when serious money first entered the space.
             A rising coverage trend (more articles per quarter) confirms the
             space is still attracting new investment. A plateau or decline
             means the investment phase may be ending — the technology is
             either failing or being absorbed by incumbents (which don't need
             TechCrunch to announce acquisitions the same way startups do).

Parameters (search_funding_articles):
    query (str) : Technology keyword to search TechCrunch for.
    limit (int) : Maximum articles to return (1–20). Default: 10.

Parameters (get_coverage_timeline):
    query (str) : Technology keyword to build a coverage timeline for.
    limit (int) : Articles to analyse for timeline (1–40). Default: 20.

Returns:
    search_funding_articles → List[dict]: Article metadata + lineage.
    get_coverage_timeline   → dict: Timeline stats + trend + lineage.

MCP Schema (search_funding_articles):
    {
        "name": "techcrunch_search_funding",
        "description": "Search TechCrunch for funding and startup articles mentioning a technology.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 10, "minimum": 1, "maximum": 20}
            },
            "required": ["query"]
        }
    }

MCP Schema (get_coverage_timeline):
    {
        "name": "techcrunch_coverage_timeline",
        "description": "Build a temporal coverage timeline for a technology on TechCrunch.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 20, "minimum": 1, "maximum": 40}
            },
            "required": ["query"]
        }
    }

Scraping notes:
    - Robots.txt is checked before every fetch session.
    - TechCrunch disallows ``/?s=`` (HTML search) in robots.txt — when that
      applies, this module uses the allowed ``/feed/`` RSS URL and filters
      items by query locally.
    - 2-second delay between paginated requests (HTML or RSS pages).
    - WordPress search HTML (when allowed) lives at /?s={query}; article cards
      follow a consistent structure across recent theme versions.
    - No credentials required.
"""

import logging
import re
import time
import urllib.parse
import urllib.robotparser
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html import unescape

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("techcrunch_scraper")

_BASE_URL      = "https://techcrunch.com"
_SEARCH_URL    = f"{_BASE_URL}/"
_RSS_FEED_URL  = f"{_BASE_URL}/feed/"
_ROBOTS_URL    = f"{_BASE_URL}/robots.txt"
_REQUEST_DELAY = 2  # seconds between paginated requests

# Funding vocabulary for client-side relevance filtering.
# TechCrunch search doesn't support boolean operators, so we fetch results
# for the technology query and then score each article against these terms
# to surface investment-phase coverage specifically.
_FUNDING_KEYWORDS = {
    "funding", "raises", "raised", "series a", "series b", "series c",
    "seed round", "venture", "vc", "investment", "investor", "acqui",
    "startup", "launch", "backed", "valuation", "unicorn",
}

_HEADERS = {
    "User-Agent": (
        "TechPipelineTracker/1.0 (academic research scraper; "
        "reads public TechCrunch search results)"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}


# ---------------------------------------------------------------------------
# Robots.txt check
# ---------------------------------------------------------------------------

def _is_scraping_allowed(url: str) -> bool:
    """
    Checks whether scraping the target URL is permitted by robots.txt.

    Returns True if allowed or if robots.txt is unreachable (fail-open).
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
        logger.warning("Could not fetch robots.txt (%s) — proceeding with caution", e)
        return True


def _strip_html(text: str) -> str:
    """Strip tags from RSS description HTML."""
    if not text:
        return ""
    plain = unescape(re.sub(r"<[^>]+>", " ", text))
    return re.sub(r"\s+", " ", plain).strip()


def _rss_pubdate_to_iso(pub_date: str) -> str:
    if not pub_date:
        return ""
    try:
        dt = parsedate_to_datetime(pub_date.strip())
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc).isoformat()
        return dt.isoformat()
    except (TypeError, ValueError, OverflowError):
        return pub_date.strip()


def _item_matches_query(title: str, summary: str, query: str) -> bool:
    """
    Match query against RSS items: prefer full phrase, then all words in body,
    with at least one significant word in the title (reduces unrelated excerpts).
    """
    title_l = title.lower()
    blob = f"{title} {summary}".lower()
    q = query.lower().strip()
    if q in blob:
        return True
    words = [w for w in q.split() if len(w) > 1]
    if not words:
        return q in blob
    if not all(w in blob for w in words):
        return False
    return any(w in title_l for w in words)


def _parse_rss_items(xml_content: str) -> list[dict]:
    """
    Parse WordPress RSS 2.0 XML into the same article dict shape as HTML search.
    """
    articles: list[dict] = []
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError:
        return articles

    channel = root.find("channel")
    if channel is None:
        return articles

    for item in channel.findall("item"):
        title_el = item.find("title")
        title = (title_el.text or "").strip() if title_el is not None else ""
        link_el = item.find("link")
        url = (link_el.text or "").strip() if link_el is not None else ""
        desc_el = item.find("description")
        raw_desc = (desc_el.text or "") if desc_el is not None else ""
        summary = _strip_html(raw_desc)[:300]
        pub = item.find("pubDate")
        pub_raw = (pub.text or "").strip() if pub is not None else ""
        date_iso = _rss_pubdate_to_iso(pub_raw)
        tags = [c.text.strip() for c in item.findall("category") if c.text]

        if title and url:
            articles.append({
                "title":   title,
                "summary": summary,
                "date":    date_iso or pub_raw,
                "url":     url,
                "tags":    tags,
            })

    return articles


def _fetch_rss_page(page: int) -> str | None:
    """Fetch one RSS page (paged WordPress feed). Returns XML text or None."""
    if page < 1:
        page = 1
    if page == 1:
        url = _RSS_FEED_URL
    else:
        url = f"{_RSS_FEED_URL}?paged={page}"

    if not _is_scraping_allowed(url):
        logger.warning("robots.txt disallows RSS fetch %s", url)
        return None

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=20)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
        logger.error("RSS fetch failed: %s", e)
        return None

    if resp.status_code != 200:
        logger.warning("RSS HTTP %s for %s", resp.status_code, url)
        return None

    return resp.text


def _search_funding_via_rss(query: str, limit: int) -> list[dict]:
    """
    TechCrunch robots.txt disallows ``/?s=`` (HTML search). The public RSS feed
    at ``/feed/`` is allowed — we pull recent items and filter by query locally.
    """
    collected: list[dict] = []
    page = 1
    max_pages = 8

    while len(collected) < limit and page <= max_pages:
        xml_text = _fetch_rss_page(page)
        if not xml_text:
            break

        raw = _parse_rss_items(xml_text)
        if not raw and page > 1:
            break

        for article in raw:
            if not _item_matches_query(
                article.get("title", ""),
                article.get("summary", ""),
                query,
            ):
                continue
            collected.append(article)
            if len(collected) >= limit:
                break

        page += 1
        if len(collected) < limit and page <= max_pages:
            time.sleep(_REQUEST_DELAY)

    return collected[:limit]


# ---------------------------------------------------------------------------
# W3C PROV lineage helper
# ---------------------------------------------------------------------------

def _build_prov_record(activity_label: str, query: str, **kwargs) -> dict:
    """
    Builds a W3C PROV-inspired lineage record for a TechCrunch scrape.

    activity_label distinguishes between search_funding_articles and
    get_coverage_timeline in the lineage trail.
    """
    return {
        "prov:type": "prov:Entity",
        "prov:wasGeneratedBy": {
            "prov:type": "prov:Activity",
            "prov:label": activity_label,
            "prov:startedAtTime": datetime.now(timezone.utc).isoformat(),
            "prov:used": {
                "source": "TechCrunch (scraped)",
                "base_url": _BASE_URL,
                "query": query,
                **kwargs,
            },
        },
        "prov:wasAttributedTo": {
            "prov:type": "prov:Agent",
            "prov:label": "TechPipelineTracker:techcrunch_scraper",
        },
    }


# ---------------------------------------------------------------------------
# Fetch and parse helpers
# ---------------------------------------------------------------------------

def _fetch_search_page(query: str, page: int = 1) -> BeautifulSoup:
    """
    Fetches a TechCrunch search results page and returns a BeautifulSoup
    parse tree.

    TechCrunch uses WordPress, which exposes search at /?s={query}.
    Pagination uses the standard WordPress ?paged={n} parameter.

    URL pattern:
        Page 1: https://techcrunch.com/?s=diffusion+models
        Page 2: https://techcrunch.com/?s=diffusion+models&paged=2

    Raises RuntimeError on HTTP or network errors.
    """
    params: dict[str, str | int] = {"s": query}
    if page > 1:
        params["paged"] = page

    url = f"{_SEARCH_URL}?{urllib.parse.urlencode(params)}"
    logger.debug("Fetching TechCrunch search page %d: %s", page, url)

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=15)
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(f"Network error reaching TechCrunch: {e}") from e
    except requests.exceptions.Timeout:
        raise RuntimeError("TechCrunch request timed out after 15 seconds")

    if resp.status_code == 403:
        raise RuntimeError(
            "TechCrunch returned 403 Forbidden — scraper may be rate-limited. "
            "Wait a few minutes before retrying."
        )
    if resp.status_code == 404:
        # WordPress returns 404 when paginating beyond available results
        return None  # type: ignore[return-value]

    if not resp.ok:
        raise RuntimeError(f"TechCrunch returned HTTP {resp.status_code}")

    return BeautifulSoup(resp.text, "html.parser")


def _parse_articles_from_soup(soup: BeautifulSoup) -> list[dict]:
    """
    Extracts article metadata from a TechCrunch search results page.

    TechCrunch's search results page structure (WordPress theme):

        <article class="post-block ...">
            <header class="post-block__header">
                <h2 class="post-block__title">
                    <a href="https://techcrunch.com/...">Article Title</a>
                </h2>
            </header>
            <div class="post-block__header">
                <span class="river-byline">
                    <time datetime="2024-01-15T10:30:00Z">January 15, 2024</time>
                </span>
            </div>
            <div class="post-block__content">
                <p>Article summary text...</p>
            </div>
        </article>

    We use multiple CSS selector fallbacks because TechCrunch has
    made minor theme changes over the years. The most important fields
    for pipeline analysis are: title, date (for timeline), and url.
    Tags are extracted from the article's category links if present.
    """
    articles = []

    # TechCrunch wraps each search result in an <article> tag —
    # this is semantically correct HTML and stable across theme versions
    article_tags = soup.find_all("article")

    for article in article_tags:
        # ── Title ──────────────────────────────────────────────────────────
        # Try post-block__title first (current theme), then generic h2/h3
        title_el = (
            article.find(class_="post-block__title")
            or article.find("h2")
            or article.find("h3")
        )
        title_link = title_el.find("a") if title_el else None
        title = title_link.get_text(strip=True) if title_link else ""
        url   = title_link.get("href", "")      if title_link else ""

        if not title or not url:
            # Skip malformed entries — likely ads or sidebar content
            continue

        # ── Date ───────────────────────────────────────────────────────────
        # WordPress outputs a <time datetime="ISO8601"> which is the most
        # reliable source. The datetime attribute is machine-readable and
        # doesn't depend on locale formatting of the visible text.
        time_el = article.find("time")
        date_raw = ""
        if time_el:
            # Prefer the machine-readable datetime attribute
            date_raw = time_el.get("datetime", "") or time_el.get_text(strip=True)

        # ── Summary ────────────────────────────────────────────────────────
        # post-block__content contains the article excerpt
        content_el = article.find(class_="post-block__content")
        if not content_el:
            # Fallback: first <p> in the article that isn't inside the title
            content_el = article.find("p")
        summary = content_el.get_text(strip=True) if content_el else ""

        # ── Tags / Categories ──────────────────────────────────────────────
        # TechCrunch renders category tags as links with class "river-byline__tags"
        # or as <a> elements with "/category/" in their href
        tags = []
        tag_container = article.find(class_="river-byline__tags")
        if tag_container:
            tags = [a.get_text(strip=True) for a in tag_container.find_all("a")]
        else:
            # Fallback: find category links by URL pattern
            tags = [
                a.get_text(strip=True)
                for a in article.find_all("a", href=True)
                if "/category/" in a["href"] or "/tag/" in a["href"]
            ]

        articles.append({
            "title":   title,
            "summary": summary[:300],   # Truncate — enough for relevance check
            "date":    date_raw,
            "url":     url,
            "tags":    tags,
        })

    return articles


def _is_funding_relevant(article: dict) -> bool:
    """
    Returns True if the article appears to be about funding, investment,
    or startup activity — the Stage 3 signal we care about.

    Checks the title, summary, and tags for any of the funding keywords.
    This is a permissive filter — we'd rather include a borderline article
    than miss a real funding announcement.
    """
    text = " ".join([
        article.get("title", ""),
        article.get("summary", ""),
        " ".join(article.get("tags", [])),
    ]).lower()

    return any(kw in text for kw in _FUNDING_KEYWORDS)


def _parse_date(date_str: str) -> datetime | None:
    """
    Parses a date string from a TechCrunch article into a datetime object.

    TechCrunch uses ISO 8601 in the datetime attribute (e.g. "2024-01-15T10:30:00Z")
    but may fall back to human-readable strings. We try ISO 8601 first.
    """
    if not date_str:
        return None
    # Try ISO 8601 with timezone
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str[:19], fmt[:len(date_str[:19])])
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Core tool functions
# ---------------------------------------------------------------------------

def search_funding_articles(query: str, limit: int = 10) -> list[dict]:
    """
    Searches TechCrunch for articles about a technology with a focus on
    funding and startup coverage, and returns structured article metadata.

    Stage 3 signal: each TechCrunch article about a funding round is a
    direct data point that VC money has entered the space. The article date
    tells you *when*, the article title often names the company and round
    size, and the URL links to the original source for citation.

    Strategy: fetch TechCrunch search results for `query`, then apply a
    client-side relevance filter for funding vocabulary. This two-pass
    approach handles TechCrunch's lack of boolean search operators while
    keeping the returned articles highly signal-relevant.

    Args:
        query : Technology keyword to search for on TechCrunch.
        limit : Max articles to return (capped at 20).

    Returns:
        List of article dicts, each containing:
            title, summary, date, url, tags, is_funding_relevant,
            source, fetched_at, lineage

    Raises:
        RuntimeError : On HTTP or network errors (HTML path only).

    Note:
        TechCrunch ``robots.txt`` disallows ``/?s=`` (HTML search). This function
        automatically uses the public ``/feed/`` RSS URL (allowed) and filters
        items by the query string when search is blocked.
    """
    limit = max(1, min(limit, 20))

    logger.info(
        "techcrunch_search_funding called | query=%r | limit=%d", query, limit
    )

    search_url = f"{_SEARCH_URL}?s={urllib.parse.quote(query)}"
    lineage = _build_prov_record("techcrunch_search_funding", query, limit=limit)
    fetched_at = datetime.now(timezone.utc).isoformat()

    collected: list[dict] = []

    # robots.txt disallows ``/?s=`` (HTML search). Use allowed RSS feed instead.
    if not _is_scraping_allowed(search_url):
        logger.info(
            "HTML search disallowed by robots.txt — using RSS feed fallback "
            "(filtering items for query=%r)",
            query,
        )
        collected = _search_funding_via_rss(query, limit)
    else:
        page = 1
        # Paginate until we have enough articles or run out of results.
        while len(collected) < limit and page <= 3:
            soup = _fetch_search_page(query, page)

            if soup is None:
                break

            raw_articles = _parse_articles_from_soup(soup)

            if not raw_articles:
                break

            for article in raw_articles:
                article["is_funding_relevant"] = _is_funding_relevant(article)
                article["source"]     = "techcrunch"
                article["fetched_at"] = fetched_at
                article["lineage"]    = lineage
                collected.append(article)

            page += 1
            if page <= 3:
                time.sleep(_REQUEST_DELAY)

        collected = collected[:limit]

    for article in collected:
        article.setdefault("is_funding_relevant", _is_funding_relevant(article))
        article.setdefault("source", "techcrunch")
        article.setdefault("fetched_at", fetched_at)
        article.setdefault("lineage", lineage)

    logger.info(
        "techcrunch_search_funding returned %d articles for query=%r "
        "(%d funding-relevant)",
        len(collected), query,
        sum(1 for a in collected if a.get("is_funding_relevant")),
    )
    return collected


def get_coverage_timeline(query: str, limit: int = 20) -> dict:
    """
    Builds a temporal coverage timeline for a technology by collecting
    TechCrunch articles and analysing their publication date distribution.

    Stage 3 signal: the timeline is the most informative dimension of
    TechCrunch data. The date of the *first* article is when the mainstream
    tech press first noticed the technology (often coincides with first
    funding round). The gap between earliest and latest article is the
    "press lifespan". coverage_trend (rising/falling/stable) shows whether
    press interest is growing or declining — a declining trend on an active
    technology often means it has moved past the "novel startup" phase into
    Big Tech product territory (where funding rounds aren't newsworthy).

    Trend calculation:
        Articles are bucketed by year-month. The trend compares the average
        monthly article count in the first half of the timeline vs the second
        half. >10% increase = rising, >10% decrease = falling.

    Args:
        query : Technology keyword to build a coverage timeline for.
        limit : Number of articles to analyse (fetched across pages).

    Returns:
        dict with keys:
            query, total_found, earliest_article {title, date, url},
            latest_article {title, date, url}, coverage_trend,
            monthly_distribution {YYYY-MM: count},
            funding_article_count, source, fetched_at, lineage

    Raises:
        RuntimeError : On HTTP or network errors from underlying fetch logic.
    """
    limit = max(1, min(limit, 40))

    logger.info(
        "techcrunch_coverage_timeline called | query=%r | limit=%d", query, limit
    )

    lineage = _build_prov_record("techcrunch_coverage_timeline", query, limit=limit)
    fetched_at = datetime.now(timezone.utc).isoformat()

    # Reuse search_funding_articles for fetching — shares all scraping logic
    articles = search_funding_articles(query=query, limit=limit)

    if not articles:
        return {
            "query":                query,
            "total_found":          0,
            "earliest_article":     None,
            "latest_article":       None,
            "coverage_trend":       "no_data",
            "monthly_distribution": {},
            "funding_article_count": 0,
            "source":               "techcrunch",
            "fetched_at":           fetched_at,
            "lineage":              lineage,
        }

    # Parse dates and sort chronologically (oldest first)
    dated = []
    for article in articles:
        dt = _parse_date(article.get("date", ""))
        if dt:
            dated.append((dt, article))

    dated.sort(key=lambda x: x[0])

    # Monthly distribution: {YYYY-MM: count}
    # This lets the analyst agent see exactly when TechCrunch coverage
    # spiked — often coincides with major funding announcements or
    # product launches that triggered multiple follow-up articles
    monthly: dict[str, int] = {}
    for dt, _ in dated:
        key = dt.strftime("%Y-%m")
        monthly[key] = monthly.get(key, 0) + 1

    # Trend: compare first-half vs second-half monthly average
    months = list(monthly.values())
    trend = "stable"
    if len(months) >= 2:
        mid = len(months) // 2
        first_avg  = sum(months[:mid])  / max(mid, 1)
        second_avg = sum(months[mid:])  / max(len(months) - mid, 1)
        if first_avg > 0:
            ratio = (second_avg - first_avg) / first_avg
            if ratio > 0.10:
                trend = "rising"
            elif ratio < -0.10:
                trend = "falling"

    def _article_summary(dt: datetime, article: dict) -> dict:
        return {
            "title": article.get("title", ""),
            "date":  article.get("date", ""),
            "url":   article.get("url", ""),
        }

    funding_count = sum(1 for a in articles if a.get("is_funding_relevant"))

    timeline = {
        "query":                query,
        "total_found":          len(articles),
        "earliest_article":     _article_summary(*dated[0])  if dated else None,
        "latest_article":       _article_summary(*dated[-1]) if dated else None,
        "coverage_trend":       trend,
        "monthly_distribution": monthly,
        "funding_article_count": funding_count,
        # Provenance fields
        "source":               "techcrunch",
        "fetched_at":           fetched_at,
        "lineage":              lineage,
    }

    logger.info(
        "techcrunch_coverage_timeline | query=%r | total=%d | trend=%s | funding=%d",
        query, len(articles), trend, funding_count,
    )
    return timeline


# ---------------------------------------------------------------------------
# MCP-style tool metadata — used by agents for tool discovery
# ---------------------------------------------------------------------------

TOOL_DEFINITION_SEARCH = {
    "name": "techcrunch_search_funding",
    "description": (
        "Search TechCrunch for articles mentioning a technology, with relevance "
        "filtering for funding and startup coverage. Returns titles, summaries, "
        "dates, URLs, and tags. Use this as a Stage 3 signal — TechCrunch is "
        "where VC-backed startups announce funding rounds."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Technology keyword to search TechCrunch for.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum articles to return.",
                "default": 10,
                "minimum": 1,
                "maximum": 20,
            },
        },
        "required": ["query"],
    },
}

TOOL_DEFINITION_TIMELINE = {
    "name": "techcrunch_coverage_timeline",
    "description": (
        "Build a temporal press coverage timeline for a technology on TechCrunch. "
        "Returns earliest/latest article, monthly distribution, and a "
        "rising/falling/stable trend. Use this to determine when VC money first "
        "entered a space and whether investment press interest is still growing."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Technology keyword to build a coverage timeline for.",
            },
            "limit": {
                "type": "integer",
                "description": "Number of articles to analyse for the timeline.",
                "default": 20,
                "minimum": 1,
                "maximum": 40,
            },
        },
        "required": ["query"],
    },
}

TOOL_DEFINITIONS = [TOOL_DEFINITION_SEARCH, TOOL_DEFINITION_TIMELINE]


# ---------------------------------------------------------------------------
# Quick smoke-test — run directly to verify scraping and output shape
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    QUERY = "AI agents"

    print(f"\n=== search_funding_articles: '{QUERY}' ===")
    articles = search_funding_articles(query=QUERY, limit=5)
    for i, article in enumerate(articles, 1):
        print(f"\n--- Article {i} ---")
        print(f"Title    : {article['title']}")
        print(f"Date     : {article['date']}")
        print(f"Funding? : {article['is_funding_relevant']}")
        print(f"Tags     : {', '.join(article['tags'][:4])}")
        print(f"URL      : {article['url']}")
        print(f"Lineage  : {article['lineage']['prov:wasGeneratedBy']['prov:label']}")

    print(f"\n=== get_coverage_timeline: '{QUERY}' ===")
    timeline = get_coverage_timeline(query=QUERY, limit=20)
    print(f"Total found         : {timeline['total_found']}")
    print(f"Earliest article    : {timeline['earliest_article']}")
    print(f"Latest article      : {timeline['latest_article']}")
    print(f"Coverage trend      : {timeline['coverage_trend']}")
    print(f"Funding articles    : {timeline['funding_article_count']}")
    print(f"Monthly dist.       : {timeline['monthly_distribution']}")
    print(f"Fetched at          : {timeline['fetched_at']}")
    print(f"Lineage             : {timeline['lineage']['prov:wasGeneratedBy']['prov:label']}")
