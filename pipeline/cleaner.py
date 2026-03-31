"""
Data normalisation layer — DataCleaner.

Why normalisation matters before embedding
──────────────────────────────────────────
Every tool in this pipeline returns data in a slightly different shape.
arXiv gives you "abstract"; Semantic Scholar gives you "abstract" too but
may include LaTeX markup. ProductHunt gives "tagline"; GitHub gives
"description"; YC gives "one_liner". NewsAPI gives "description";
TechCrunch gives "summary". All of these fields mean the same thing —
"what is this thing about?" — but they arrive in different keys, with
different formatting artifacts, and with different length distributions.

This matters critically at the embedding stage. The sentence-transformer
model (all-MiniLM-L6-v2) treats its input as plain text. If you feed it:

    "Attention Is All You Need\n\nWe propose a new simple network <b>architecture</b>..."

...the HTML tag <b> becomes part of the semantic signal, and the newlines
fragment the sentence boundary detection. The embedding you get is slightly
wrong — and "slightly wrong" embeddings across thousands of documents produce
a vector space where similar concepts are not as close as they should be,
degrading the quality of every RAG search result the analyst agent uses.

Normalisation rules applied here:
  1. Strip HTML tags — removes <b>, <p>, <a href=...>, etc.
  2. Collapse whitespace — newlines, tabs, double-spaces → single space
  3. None → "" — prevents TypeError in string operations downstream
  4. Date standardisation — YYYY-MM-DD everywhere, regardless of source format
  5. Deduplication — same URL or ID appearing twice (e.g. from overlapping
     NewsAPI + TechCrunch fetches) produces one clean record, not two
  6. Truncation to 512 tokens — all-MiniLM-L6-v2 has a 256 WordPiece token
     limit internally, but our "tokens" are whitespace-split words. 512 words
     is a conservative proxy that keeps inputs well within the model's window.
     Truncating also keeps MongoDB documents and ChromaDB payloads lean.
  7. cleaned_text — a single unified text field (title + description) that
     is the only field passed to the embedding model. This decouples the
     embedding logic from field-name variations across sources.

Architecture position
─────────────────────
    pipeline/fetcher.py  → raw documents (heterogeneous shapes)
         ↓
    pipeline/cleaner.py  → normalised documents (uniform schema)  ← THIS FILE
         ↓
    pipeline/embedder.py → embeddings generated from cleaned_text
    database/vector_store.py → embeddings stored
"""

import logging
import re
from datetime import datetime
from html.parser import HTMLParser

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("cleaner")

# Maximum word count for cleaned_text — conservative proxy for the
# all-MiniLM-L6-v2 model's internal token limit.
_MAX_TOKENS = 512

# Date format patterns tried in order when parsing raw date strings.
# Most specific formats first, most general last.
_DATE_FORMATS = [
    "%Y-%m-%dT%H:%M:%S%z",   # ISO 8601 with timezone offset: 2024-01-15T10:30:00+00:00
    "%Y-%m-%dT%H:%M:%SZ",    # ISO 8601 UTC zulu: 2024-01-15T10:30:00Z
    "%Y-%m-%dT%H:%M:%S",     # ISO 8601 no timezone: 2024-01-15T10:30:00
    "%Y-%m-%d",               # Date only: 2024-01-15
    "%B %d, %Y",              # Long month: January 15, 2024
    "%b %d, %Y",              # Short month: Jan 15, 2024
    "%d %B %Y",               # Day first: 15 January 2024
    "%Y",                     # Year only: 2024
]


# ---------------------------------------------------------------------------
# Text cleaning utilities
# ---------------------------------------------------------------------------

class _HTMLStripper(HTMLParser):
    """
    Minimal HTML tag stripper using stdlib HTMLParser.

    Using the stdlib parser (not regex) is important because HTML tag
    stripping via regex is famously unreliable — nested tags, attributes
    with > characters, and malformed markup all break naive patterns.
    HTMLParser handles these correctly.
    """

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_text(self) -> str:
        return " ".join(self._parts)


def _strip_html(text: str) -> str:
    """
    Removes HTML tags and decodes HTML entities from a string.

    Returns plain text safe for embedding. Replaces tags with spaces
    rather than empty strings so "word<b>bold</b>word" → "word bold word"
    rather than "wordboldword".
    """
    if not text:
        return ""
    stripper = _HTMLStripper()
    try:
        stripper.feed(text)
        return stripper.get_text()
    except Exception:
        # HTMLParser can raise on severely malformed input — fall back to
        # a simple regex strip as a last resort
        return re.sub(r"<[^>]+>", " ", text)


def _normalise_whitespace(text: str) -> str:
    """
    Collapses all whitespace sequences (newlines, tabs, multiple spaces)
    into a single space and strips leading/trailing whitespace.
    """
    return re.sub(r"\s+", " ", text).strip()


def _clean_text(text) -> str:
    """
    Full text cleaning pipeline: None guard → HTML strip → whitespace collapse.
    Applied to every text field before building cleaned_text.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return _normalise_whitespace(_strip_html(text))


def _truncate(text: str, max_tokens: int = _MAX_TOKENS) -> str:
    """
    Truncates text to at most max_tokens whitespace-split words and rejoins.

    This is a word-count proxy for the model's internal token limit.
    WordPiece tokenisation produces ~1.3 tokens per word on average for
    technical English, so 512 words ≈ 665 WordPiece tokens, safely within
    the all-MiniLM-L6-v2 input window.
    """
    words = text.split()
    if len(words) <= max_tokens:
        return text
    return " ".join(words[:max_tokens])


def _build_cleaned_text(*parts: str) -> str:
    """
    Joins multiple text parts (title, abstract, description, etc.) into a
    single cleaned_text field for embedding.

    Each part is individually cleaned before joining so a None or HTML-heavy
    field in one part does not corrupt the others. Parts that clean to ""
    are excluded from the join to avoid double-spaces.
    """
    cleaned_parts = [_clean_text(p) for p in parts]
    combined = ". ".join(p for p in cleaned_parts if p)
    return _truncate(combined)


def _parse_date(raw) -> str:
    """
    Parses a date value from any common format and returns YYYY-MM-DD.

    Accepts:
        - ISO 8601 strings (with or without time component)
        - Human-readable date strings (e.g. "January 15, 2024")
        - Integer or string years (e.g. 2023)
        - None or empty string → returns ""

    Returns "" rather than raising on unparseable input — a missing date
    is better than a crashed pipeline.
    """
    if raw is None or raw == "":
        return ""

    # Integer year (common in Semantic Scholar output)
    if isinstance(raw, int):
        return f"{raw}-01-01"

    raw_str = str(raw).strip()
    if not raw_str:
        return ""

    # Try each format in order
    # Strip to first 19 chars to handle sub-second precision like
    # "2024-01-15T10:30:00.000Z" without needing a separate format entry
    candidate = raw_str[:19]
    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(candidate, fmt[:len(candidate)])
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue

    # Last resort: extract a 4-digit year if present anywhere in the string
    match = re.search(r"\b(19|20)\d{2}\b", raw_str)
    if match:
        return f"{match.group()}-01-01"

    logger.debug("Could not parse date: %r", raw_str)
    return ""


def _dedup(items: list[dict], key: str) -> list[dict]:
    """
    Removes duplicate dicts from `items` based on the value of `key`.

    The first occurrence of each key value is kept; subsequent duplicates
    are dropped. Items where the key is missing or empty are kept (they
    cannot be meaningfully deduplicated).

    Deduplication is necessary because:
      - NewsAPI and TechCrunch can both return the same article URL
      - arXiv and Semantic Scholar both return the same paper (different keys)
      - Repeated pipeline runs accumulate the same data in MongoDB; after
        cleaning we want exactly one record per unique entity.
    """
    seen: set = set()
    result: list[dict] = []
    for item in items:
        val = item.get(key)
        if not val:
            result.append(item)
            continue
        if val not in seen:
            seen.add(val)
            result.append(item)
    return result


# ---------------------------------------------------------------------------
# DataCleaner
# ---------------------------------------------------------------------------

class DataCleaner:
    """
    Normalises raw documents from all pipeline tools into uniform schemas.

    Each public method accepts the raw list returned by a fetcher stage and
    returns a new list of clean dicts with predictable field names. The
    original documents are not modified — cleaning produces a fresh list.

    All methods are pure functions with no side effects (no DB writes).
    The embedder and vector store receive cleaned output from here.
    """

    # ── Stage 1: Academic papers ─────────────────────────────────────────────

    def clean_papers(self, papers: list[dict]) -> list[dict]:
        """
        Normalises arXiv and Semantic Scholar paper dicts into a uniform schema.

        Field mapping:
            arXiv            → Semantic Scholar  → cleaned field
            arxiv_id         → paper_id          → id
            title            → title             → title
            abstract         → abstract          → abstract
            authors (list)   → authors (list)    → authors (comma-separated str)
            published[:4]    → year (int)         → year (int)
            (none)           → citation_count    → citation_count (int)
            url              → url               → url
            source           → source            → source

        cleaned_text = title + ". " + abstract (truncated to 512 words).
        This is the only field passed to the embedding model.

        Args:
            papers : List of raw paper dicts from arxiv_tool or semantic_scholar_tool.

        Returns:
            Deduplicated list of clean paper dicts.
        """
        cleaned: list[dict] = []

        for raw in papers:
            # ── ID ──────────────────────────────────────────────────────────
            # arXiv uses arxiv_id; Semantic Scholar uses paper_id.
            # We normalise both to "id" and keep the original key as a hint.
            doc_id = (
                raw.get("arxiv_id")
                or raw.get("paper_id")
                or raw.get("url", "")
            )

            # ── Authors ─────────────────────────────────────────────────────
            # Both tools return a list of name strings.
            authors_raw = raw.get("authors") or []
            if isinstance(authors_raw, list):
                authors = ", ".join(_clean_text(a) for a in authors_raw if a)
            else:
                authors = _clean_text(str(authors_raw))

            # ── Year ─────────────────────────────────────────────────────────
            # arXiv stores the full ISO date in "published"; S2 stores an int.
            year_raw = raw.get("year") or raw.get("published", "")
            year_str = _parse_date(year_raw)
            year_int = int(year_str[:4]) if year_str and year_str[:4].isdigit() else None

            # ── Text fields ──────────────────────────────────────────────────
            title    = _clean_text(raw.get("title", ""))
            abstract = _clean_text(raw.get("abstract", ""))

            cleaned.append({
                "id":             str(doc_id),
                "title":          title,
                "abstract":       abstract,
                "authors":        authors,
                "year":           year_int,
                "citation_count": int(raw.get("citation_count") or 0),
                "url":            raw.get("url", ""),
                "source":         raw.get("source", ""),
                "cleaned_text":   _build_cleaned_text(title, abstract),
            })

        result = _dedup(cleaned, "id")
        logger.info("clean_papers | input=%d | output=%d", len(papers), len(result))
        return result

    # ── Stage 2: Startup signals ─────────────────────────────────────────────

    def clean_startups(self, items: list[dict]) -> list[dict]:
        """
        Normalises ProductHunt posts, GitHub repos, and YC company dicts into
        a uniform startup schema.

        These three sources represent different sub-signals within Stage 2 but
        share the same conceptual shape: an entity (product/repo/company) with
        a name, a description, and a date. Normalising them together lets the
        analyst agent treat them uniformly when computing a startup-phase score.

        Field mapping:
            ProductHunt   GitHub        YC            → cleaned field
            name          name          name          → name
            tagline       description   description   → description
            url           url           yc_url        → url
            created_at    created_at    (none)        → date
            topics (list) topics (list) tags (list)  → tags (comma-sep str)
            "producthunt" "github"      "ycombinator" → source
            2             2             2             → stage_signal

        stage_signal is always 2 for startup items — included so the analyst
        agent can filter by stage without inspecting the source field.

        Args:
            items : Mixed list of dicts from producthunt_tool, github_tool,
                    and yc_scraper.

        Returns:
            Deduplicated list of clean startup dicts.
        """
        cleaned: list[dict] = []

        for raw in items:
            source = raw.get("source", "")

            # ── Name ────────────────────────────────────────────────────────
            name = _clean_text(
                raw.get("name", "")
                or raw.get("full_name", "")
            )

            # ── Description ─────────────────────────────────────────────────
            # ProductHunt: tagline or description
            # GitHub: description
            # YC: description (already normalised by yc_scraper._parse_company)
            description = _clean_text(
                raw.get("tagline")
                or raw.get("description")
                or raw.get("one_liner")
                or ""
            )

            # ── URL ──────────────────────────────────────────────────────────
            url = (
                raw.get("url")
                or raw.get("yc_url")
                or raw.get("html_url")
                or ""
            )

            # ── Date ─────────────────────────────────────────────────────────
            date_raw = (
                raw.get("created_at")
                or raw.get("date")
                or ""
            )
            date = _parse_date(date_raw)

            # ── Tags ─────────────────────────────────────────────────────────
            # Tags may be a list of strings (all three sources) or already a
            # comma-separated string — handle both.
            tags_raw = raw.get("tags") or raw.get("topics") or []
            if isinstance(tags_raw, list):
                tags = ", ".join(_clean_text(str(t)) for t in tags_raw if t)
            else:
                tags = _clean_text(str(tags_raw))

            # ── Extra context for embedding ──────────────────────────────────
            # GitHub repos have a language field that is useful semantic context
            extra = _clean_text(raw.get("language") or raw.get("batch") or "")

            cleaned.append({
                "id":           url or name,
                "name":         name,
                "description":  description,
                "source":       source,
                "stage_signal": 2,
                "url":          url,
                "date":         date,
                "tags":         tags,
                "cleaned_text": _build_cleaned_text(name, description, tags, extra),
            })

        result = _dedup(cleaned, "id")
        logger.info("clean_startups | input=%d | output=%d", len(items), len(result))
        return result

    # ── Stage 3: Investment articles ─────────────────────────────────────────

    def clean_articles(self, articles: list[dict]) -> list[dict]:
        """
        Normalises NewsAPI and TechCrunch article dicts into a uniform schema.

        Both sources produce articles with a title, a text snippet, a URL,
        and a date. The key difference is the field name for the text snippet:
        NewsAPI uses "description"; TechCrunch uses "summary". Both are short
        excerpts (not full article text), so they get the same treatment.

        is_funding_relevant is preserved from the techcrunch_scraper's
        client-side filtering and set to False for NewsAPI articles by default
        (since news_tool already filters for funding vocabulary in its query).

        Args:
            articles : Mixed list of dicts from news_tool and techcrunch_scraper.

        Returns:
            Deduplicated list of clean article dicts.
        """
        cleaned: list[dict] = []

        for raw in articles:
            title = _clean_text(raw.get("title", ""))

            # Summary field varies by source
            summary = _clean_text(
                raw.get("summary")
                or raw.get("description")
                or raw.get("content_snippet")
                or ""
            )

            url  = raw.get("url", "")
            date = _parse_date(
                raw.get("published_at")
                or raw.get("date")
                or raw.get("publishedAt")
                or ""
            )
            # api_source: the tool/scraper that fetched this article ("newsapi"
            # or "techcrunch"). Preserved separately because source_name holds
            # the human-readable publication name (e.g. "TechCrunch") which
            # would otherwise overwrite the API tag and break source filtering.
            api_source = raw.get("source", "")
            source = raw.get("source_name") or raw.get("source", "")

            # is_funding_relevant: TechCrunch scraper sets this explicitly;
            # NewsAPI articles are already filtered for funding terms so we
            # default to True for newsapi source.
            is_funding = raw.get("is_funding_relevant")
            if is_funding is None:
                is_funding = api_source == "newsapi"

            cleaned.append({
                "id":                 url,
                "title":              title,
                "summary":            summary,
                "source":             str(source),
                "api_source":         api_source,
                "url":                url,
                "date":               date,
                "is_funding_relevant": bool(is_funding),
                "cleaned_text":       _build_cleaned_text(title, summary),
            })

        result = _dedup(cleaned, "id")
        logger.info("clean_articles | input=%d | output=%d", len(articles), len(result))
        return result

    # ── Stage 4: Patents ─────────────────────────────────────────────────────

    def clean_patents(self, patents: list[dict]) -> list[dict]:
        """
        Normalises PatentsView patent dicts into a uniform schema.

        Patents have two especially important fields for Stage 4 scoring:
          assignee_orgs — which corporations filed the patent. If the assignees
                          are all large tech companies (Google, Microsoft, Apple),
                          this confirms big-tech institutional adoption.
          date          — when the patent was filed. Patent filing dates lag
                          real R&D activity by 12–24 months, so a 2022 patent
                          date on a technology suggests R&D started ~2020–2021.

        assignee_orgs is joined to a comma-separated string because the
        embedding model needs plain text, and because having multiple assignees
        on one patent is a graph relationship (handled by graph_client.py),
        not a text embedding concern.

        Args:
            patents : List of patent dicts from patents_tool.

        Returns:
            Deduplicated list of clean patent dicts.
        """
        cleaned: list[dict] = []

        for raw in patents:
            patent_id = raw.get("patent_id", "")
            title     = _clean_text(raw.get("title", ""))
            abstract  = _clean_text(raw.get("abstract", ""))
            date      = _parse_date(raw.get("date", ""))

            # assignee_orgs is already a list of strings from patents_tool
            assignees_raw = raw.get("assignee_orgs") or []
            if isinstance(assignees_raw, list):
                assignees = ", ".join(_clean_text(a) for a in assignees_raw if a)
            else:
                assignees = _clean_text(str(assignees_raw))

            # USPTO patent URL pattern — PatentsView doesn't always return a URL
            url = (
                raw.get("url")
                or (f"https://patents.google.com/patent/US{patent_id}" if patent_id else "")
            )

            cleaned.append({
                "id":           patent_id,
                "title":        title,
                "abstract":     abstract,
                "assignees":    assignees,
                "date":         date,
                "url":          url,
                "source":       raw.get("source", "patentsview"),
                "cleaned_text": _build_cleaned_text(title, abstract, assignees),
            })

        result = _dedup(cleaned, "id")
        logger.info("clean_patents | input=%d | output=%d", len(patents), len(result))
        return result

    # ── Full pipeline clean ───────────────────────────────────────────────────

    def clean_all(self, raw_data: dict) -> dict:
        """
        Runs the appropriate cleaner on each section of a fetch_all() result.

        Mirrors the structure of fetch_all() so the output can be passed
        directly to the embedder and analyst agent without reshaping.

        Input structure (from DataFetcher.fetch_all()):
            raw_data["academic"]              → list[dict]
            raw_data["startup"]["github_repos"]
            raw_data["startup"]["producthunt"]
            raw_data["startup"]["yc_companies"]
            raw_data["investment"]["news_articles"]
            raw_data["investment"]["techcrunch_articles"]
            raw_data["bigtech"]               → list[dict]
            raw_data["mainstream"]            → dict (not cleaned — point-in-time)

        mainstream signals (Wikipedia, Google Trends) are passed through
        unchanged — they are already structured dicts with no embedded HTML
        or field-name variance, and they are not sent to the embedding model.

        Args:
            raw_data : The dict returned by DataFetcher.fetch_all().

        Returns:
            dict with the same top-level keys but cleaned values:
                query, fetched_at, academic, startup, investment,
                bigtech, mainstream, summary
        """
        if not isinstance(raw_data, dict):
            raise ValueError("raw_data must be a dict from DataFetcher.fetch_all()")

        query      = raw_data.get("query", "")
        fetched_at = raw_data.get("fetched_at", "")

        # ── Academic ─────────────────────────────────────────────────────────
        academic_clean = self.clean_papers(raw_data.get("academic") or [])

        # ── Startup ──────────────────────────────────────────────────────────
        startup_raw  = raw_data.get("startup") or {}
        startup_items = (
            (startup_raw.get("github_repos")  or [])
            + (startup_raw.get("producthunt") or [])
            + (startup_raw.get("yc_companies") or [])
        )
        startup_clean = self.clean_startups(startup_items)

        # ── Investment ───────────────────────────────────────────────────────
        investment_raw = raw_data.get("investment") or {}
        article_items  = (
            (investment_raw.get("news_articles")       or [])
            + (investment_raw.get("techcrunch_articles") or [])
        )
        investment_clean = self.clean_articles(article_items)

        # ── Big Tech ─────────────────────────────────────────────────────────
        bigtech_clean = self.clean_patents(raw_data.get("bigtech") or [])

        # ── Mainstream — pass through unchanged ──────────────────────────────
        mainstream = raw_data.get("mainstream") or {}

        # ── Summary counts ────────────────────────────────────────────────────
        summary = {
            "academic_papers":     len(academic_clean),
            "startup_entities":    len(startup_clean),
            "investment_articles": len(investment_clean),
            "patents":             len(bigtech_clean),
            "wikipedia_exists":    mainstream.get("wikipedia", {}).get("exists", False),
            "trending_on_google":  mainstream.get("trends", {}).get("query_found", False),
        }

        logger.info(
            "clean_all | query=%r | academic=%d startup=%d investment=%d patents=%d",
            query,
            summary["academic_papers"],
            summary["startup_entities"],
            summary["investment_articles"],
            summary["patents"],
        )

        return {
            "query":      query,
            "fetched_at": fetched_at,
            "academic":   academic_clean,
            "startup":    startup_clean,
            "investment": investment_clean,
            "bigtech":    bigtech_clean,
            "mainstream": mainstream,
            "summary":    summary,
        }


# ---------------------------------------------------------------------------
# Module-level convenience instance
# ---------------------------------------------------------------------------

# Import pattern:
#     from pipeline.cleaner import cleaner
#     cleaned = cleaner.clean_all(raw_data)
cleaner = DataCleaner()


# ---------------------------------------------------------------------------
# Quick smoke-test — run directly with synthetic data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    c = DataCleaner()

    # ── Papers ───────────────────────────────────────────────────────────────
    print("\n=== clean_papers ===")
    raw_papers = [
        {
            "arxiv_id": "1706.03762",
            "title": "Attention Is All You Need",
            "abstract": "<p>We propose a new simple network <b>architecture</b>, the Transformer.\nBased solely on attention mechanisms.</p>",
            "authors": ["Vaswani", "Shazeer", None],
            "published": "2017-06-12T00:00:00Z",
            "citation_count": 90000,
            "url": "https://arxiv.org/abs/1706.03762",
            "source": "arxiv",
            "fetched_at": "2026-03-30T10:00:00Z",
            "lineage": {},
        },
        {
            "paper_id": "abc123",
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "abstract": None,
            "authors": ["Devlin", "Chang"],
            "year": 2019,
            "citation_count": None,
            "url": "https://www.semanticscholar.org/paper/abc123",
            "source": "semantic_scholar",
            "fetched_at": "2026-03-30T10:00:00Z",
            "lineage": {},
        },
        # Duplicate — should be dropped
        {
            "arxiv_id": "1706.03762",
            "title": "Attention Is All You Need (duplicate)",
            "abstract": "...",
            "authors": [],
            "published": "2017-06-12",
            "citation_count": 90000,
            "url": "https://arxiv.org/abs/1706.03762",
            "source": "arxiv",
            "fetched_at": "2026-03-30T10:00:00Z",
            "lineage": {},
        },
    ]
    papers = c.clean_papers(raw_papers)
    for p in papers:
        print(f"  id={p['id']:<20s} year={p['year']} citations={p['citation_count']}")
        print(f"  cleaned_text[:80]: {p['cleaned_text'][:80]}")

    # ── Startups ─────────────────────────────────────────────────────────────
    print("\n=== clean_startups ===")
    raw_startups = [
        {   # ProductHunt
            "name": "Midjourney",
            "tagline": "Generate <b>stunning</b> AI art",
            "url": "https://www.producthunt.com/posts/midjourney",
            "created_at": "2022-07-12T08:00:00Z",
            "topics": ["AI", "Design", "Creativity"],
            "source": "producthunt",
            "fetched_at": "2026-03-30T10:00:00Z",
            "lineage": {},
        },
        {   # GitHub
            "name": "AUTOMATIC1111/stable-diffusion-webui",
            "description": "Stable Diffusion web UI\n\nA browser interface based on Gradio.",
            "url": "https://github.com/AUTOMATIC1111/stable-diffusion-webui",
            "created_at": "2022-08-22T00:00:00Z",
            "topics": ["stable-diffusion", "deep-learning"],
            "language": "Python",
            "source": "github",
            "fetched_at": "2026-03-30T10:00:00Z",
            "lineage": {},
        },
        {   # YC
            "name": "Stability AI",
            "description": "Foundation models for images, language and more",
            "yc_url": "https://www.ycombinator.com/companies/stability-ai",
            "batch": "W23",
            "tags": ["Generative AI", "Deep Learning"],
            "source": "ycombinator",
            "fetched_at": "2026-03-30T10:00:00Z",
            "lineage": {},
        },
    ]
    startups = c.clean_startups(raw_startups)
    for s in startups:
        print(f"  source={s['source']:<14s} name={s['name']:<40s} date={s['date']}")
        print(f"  cleaned_text[:80]: {s['cleaned_text'][:80]}")

    # ── Articles ─────────────────────────────────────────────────────────────
    print("\n=== clean_articles ===")
    raw_articles = [
        {
            "title": "Stability AI raises $101M <b>Series A</b>",
            "description": "The company behind Stable Diffusion closed a $101M round.",
            "url": "https://techcrunch.com/stability-ai-raises",
            "publishedAt": "2022-10-17T13:00:00Z",
            "source_name": "TechCrunch",
            "source": "newsapi",
            "fetched_at": "2026-03-30T10:00:00Z",
            "lineage": {},
        },
        {
            "title": "Midjourney funding and growth",
            "summary": "Midjourney grew to 1M users\twithout  VC funding.",
            "url": "https://techcrunch.com/midjourney-growth",
            "date": "2023-03-22",
            "source": "techcrunch",
            "is_funding_relevant": True,
            "fetched_at": "2026-03-30T10:00:00Z",
            "lineage": {},
        },
    ]
    articles = c.clean_articles(raw_articles)
    for a in articles:
        print(f"  source={a['source']:<12s} date={a['date']} funding={a['is_funding_relevant']}")
        print(f"  cleaned_text[:80]: {a['cleaned_text'][:80]}")

    # ── Patents ──────────────────────────────────────────────────────────────
    print("\n=== clean_patents ===")
    raw_patents = [
        {
            "patent_id": "US11423293",
            "title": "System and method for\ngenerating images using diffusion  models",
            "abstract": "<p>A neural network-based image generation system...</p>",
            "date": "2022-08-23",
            "assignee_orgs": ["Google LLC", None, "Alphabet Inc."],
            "source": "patentsview",
            "fetched_at": "2026-03-30T10:00:00Z",
            "lineage": {},
        },
    ]
    patents = c.clean_patents(raw_patents)
    for p in patents:
        print(f"  id={p['id']} date={p['date']} assignees={p['assignees']}")
        print(f"  cleaned_text[:80]: {p['cleaned_text'][:80]}")

    print("\nSmoke-test complete.")
