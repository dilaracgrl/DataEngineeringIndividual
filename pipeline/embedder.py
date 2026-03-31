"""
Embedding and storage orchestration layer — DataEmbedder.

This is the final stage of the pipeline before the analyst agent reads results.
It takes the cleaned, normalised data from DataCleaner, distributes it across
all four storage layers, computes pipeline stage scores, and returns a complete
result dict ready for agent consumption.

Architecture position
─────────────────────
    pipeline/fetcher.py   → raw documents
    pipeline/cleaner.py   → normalised documents
         ↓
    pipeline/embedder.py  → storage + scoring          ← THIS FILE
         ↓
    database/vector_store.py   ← semantic search (RAG)
    database/graph_client.py   ← relationship graph (GraphRAG)
    database/sqlite_client.py  ← stage scores + timeline
         ↓
    agents/                    ← analyst agent reads all layers

Scoring design philosophy
──────────────────────────
Each of the 10 tool scores is computed independently on a 0–100 scale using
the count of cleaned documents as the primary signal, augmented by secondary
signals where available (recency, citation count, funding relevance).

The scores are deliberately simple. The alternative — a complex ML scoring
model — would be opaque, require labelled training data, and would be a black
box when the system assigns an unexpected stage. Simple count-based scores
with documented thresholds let the analyst agent explain its reasoning:
"arxiv_score is 72 because we found 14 papers, 8 of which were published
in the last 2 years."

Overall stage derivation
────────────────────────
Each pipeline stage "activates" when at least one of its tool scores exceeds
an activation threshold (default: 30/100). The overall_stage is the highest
activated stage. This reflects the real-world pattern: a technology does not
leave Stage 2 until it has *both* developer activity AND some Stage 2 signal —
it is not purely additive.

Stage 1 — Academic    : arxiv_score > 30 OR semantic_scholar_score > 30
Stage 2 — Startup     : github_score > 30 OR producthunt_score > 30 OR yc_score > 30
Stage 3 — Investment  : news_score > 30 OR techcrunch_score > 30
Stage 4 — Big Tech    : patents_score > 30
Stage 5 — Mainstream  : wikipedia_score > 50 OR trends_score > 60

Overall score (0–100)
──────────────────────
Each stage contributes a maximum of 20 points to the overall score:
  Stage 1 contribution = stage_1_combined * 0.20  → 0–20
  Stage 2 contribution = stage_2_combined * 0.20  → 0–20
  Stage 3 contribution = stage_3_combined * 0.20  → 0–20
  Stage 4 contribution = stage_4_combined * 0.20  → 0–20
  Stage 5 contribution = stage_5_combined * 0.20  → 0–20
  overall_score = sum of contributions             → 0–100

A purely academic technology (Stage 1 only) scores at most 20.
A fully mainstream technology (all stages active) can score 100.
This makes overall_score directly readable as "percentage through the pipeline."
"""

import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.fetcher  import DataFetcher
from pipeline.cleaner  import DataCleaner
from database.vector_store  import VectorStore
from database.graph_client  import GraphClient
from database.sqlite_client import SQLiteClient
from database.mongo_client  import MongoDBClient

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("embedder")

# ── Scoring constants ────────────────────────────────────────────────────────

# Count saturation points: the document count at which a tool score reaches 100.
# Chosen empirically: 20 arXiv papers is "plenty" for a well-researched tech;
# 10 YC companies is a well-funded space; 5 patents is significant big-tech activity.
_SCALE = {
    "arxiv":            20,   # papers → 100 at 20
    "semantic_scholar": 20,   # papers → 100 at 20
    "github":           10,   # repos  → 100 at 10
    "producthunt":       8,   # posts  → 100 at 8
    "yc":               10,   # companies → 100 at 10
    "news":             10,   # articles → 100 at 10
    "techcrunch":        8,   # articles → 100 at 8
    "patents":           5,   # patents → 100 at 5
}

# Recency window: papers published within this many years receive a recency bonus
_RECENCY_YEARS = 3

# Stage activation threshold: a stage is "active" (contributes to overall_stage)
# when at least one of its tool scores exceeds this value.
_ACTIVATION_THRESHOLD = 30.0

# Wikipedia activation threshold is higher than the general threshold because
# merely having a Wikipedia article (exists=True) scores 40 — we want the
# article to have meaningful view volume before calling it Stage 5.
_WIKI_ACTIVATION_THRESHOLD = 50.0
_TRENDS_ACTIVATION_THRESHOLD = 60.0


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _score_from_count(count: int, scale: int) -> float:
    """
    Maps a document count to a 0–100 score using linear saturation.

    count=0       → 0
    count=scale/2 → 50
    count=scale   → 100
    count>scale   → 100 (saturates)

    Linear is intentional — it is predictable and explainable. A log scale
    would compress differences at the high end, making it hard to distinguish
    a technology with 30 papers from one with 100.
    """
    if scale <= 0 or count <= 0:
        return 0.0
    return min((count / scale) * 100.0, 100.0)


def _recency_bonus(items: list[dict], bonus: float = 30.0) -> float:
    """
    Returns a recency bonus if any item has a date within _RECENCY_YEARS of now.

    Applied to academic papers and patents where recency is a strong signal:
    a paper from 2022 is much more relevant to current pipeline stage than
    one from 2009. The bonus is all-or-nothing (not proportional) to keep
    scoring transparent.
    """
    if not items:
        return 0.0
    current_year = datetime.now(timezone.utc).year
    cutoff_year  = current_year - _RECENCY_YEARS
    for item in items:
        year = item.get("year")
        if year and isinstance(year, int) and year >= cutoff_year:
            return bonus
        date = item.get("date", "")
        if date and len(date) >= 4 and date[:4].isdigit():
            if int(date[:4]) >= cutoff_year:
                return bonus
    return 0.0


def _citation_bonus(items: list[dict], bonus: float = 20.0) -> float:
    """
    Returns a citation bonus if the average citation count across items is
    above a meaningful threshold (50 citations = academically influential).

    Applied to Semantic Scholar papers specifically, where citation_count is
    a reliable quality signal. Gives the semantic_scholar_score an edge over
    the raw arxiv_score for well-cited foundational work.
    """
    if not items:
        return 0.0
    counts = [item.get("citation_count", 0) or 0 for item in items]
    if not counts:
        return 0.0
    avg = sum(counts) / len(counts)
    return bonus if avg >= 50 else (bonus * avg / 50)


# ---------------------------------------------------------------------------
# DataEmbedder
# ---------------------------------------------------------------------------

class DataEmbedder:
    """
    Orchestrates the full post-cleaning storage and scoring pipeline.

    Takes output from DataCleaner.clean_all() and:
      1. Writes to all four storage layers (vector, graph, SQLite timeline)
      2. Computes per-tool and overall pipeline stage scores
      3. Persists scores to SQLite via sqlite_client.save_trend_score()

    Can be used standalone (embed_and_store + compute_stage_scores) or as a
    complete end-to-end runner via run_full_pipeline().
    """

    def __init__(
        self,
        fetcher:       DataFetcher   | None = None,
        cleaner:       DataCleaner   | None = None,
        vector_store:  VectorStore   | None = None,
        graph_client:  GraphClient   | None = None,
        sqlite_client: SQLiteClient  | None = None,
        mongo_client:  MongoDBClient | None = None,
    ) -> None:
        self._fetcher = fetcher       or DataFetcher()
        self._cleaner = cleaner       or DataCleaner()
        self._vs      = vector_store  or VectorStore()
        self._graph   = graph_client  or GraphClient()
        self._sqlite  = sqlite_client or SQLiteClient()
        self._mongo   = mongo_client  or MongoDBClient()

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _safe(self, label: str, fn, *args, **kwargs):
        """
        Calls fn(*args, **kwargs) with fault isolation.

        Storage layer failures (Neo4j down, ChromaDB full) must not abort
        the scoring pipeline — the agent still needs scores even if one
        storage layer is unavailable.

        Returns the function's result, or None on any exception.
        """
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            logger.error("[%s] %s: %s", label, type(e).__name__, e)
            return None

    def _papers_for_vector_store(self, papers: list[dict]) -> list[dict]:
        """
        Adapts cleaned paper dicts for VectorStore.add_papers().

        VectorStore expects either an arxiv_id or paper_id field (from raw
        tool output). Cleaned papers use "id" instead. This shim adds the
        correct key back so the vector store can deduplicate correctly,
        while passing cleaned_text through as "abstract" so the embedding
        model uses the pre-cleaned text rather than re-cleaning raw abstracts.
        """
        adapted = []
        for p in papers:
            entry = dict(p)  # shallow copy — do not mutate the input
            # Restore the source-appropriate identity key
            if p.get("source") == "semantic_scholar":
                entry["paper_id"] = p["id"]
            else:
                entry["arxiv_id"] = p["id"]
            # Override abstract with cleaned_text so the vector store embeds
            # the pre-cleaned version, avoiding duplicate HTML stripping.
            entry["abstract"] = p.get("cleaned_text") or p.get("abstract", "")
            adapted.append(entry)
        return adapted

    def _articles_for_vector_store(self, articles: list[dict]) -> list[dict]:
        """
        Adapts cleaned article dicts for VectorStore.add_articles().

        VectorStore uses "url" as the article identity key — already present
        in cleaned articles. We override "description" with cleaned_text for
        the same reason as papers: use the pre-cleaned text for embedding.
        """
        adapted = []
        for a in articles:
            entry = dict(a)
            entry["description"] = a.get("cleaned_text") or a.get("summary", "")
            adapted.append(entry)
        return adapted

    # ── 1. Storage layer ─────────────────────────────────────────────────────

    def embed_and_store(self, cleaned_data: dict, query: str) -> dict:
        """
        Distributes cleaned data across all storage layers.

        Storage operations are attempted independently — a Neo4j failure does
        not block ChromaDB writes, and vice versa. Each operation is wrapped
        in _safe() and its outcome is recorded in the returned summary dict.

        Layers written:
          a) Vector store  — papers and articles for semantic RAG search
          b) Graph store   — Technology, Company, Paper nodes + relationships
          c) SQLite        — Timeline events (first detection per stage)

        Args:
            cleaned_data : Output of DataCleaner.clean_all().
            query        : The technology name (used as the Technology node name).

        Returns:
            dict with "vector", "graph", "timeline" sub-dicts showing counts
            of items written to each layer.
        """
        papers     = cleaned_data.get("academic",   [])
        startups   = cleaned_data.get("startup",    [])
        articles   = cleaned_data.get("investment", [])
        patents    = cleaned_data.get("bigtech",    [])
        mainstream = cleaned_data.get("mainstream", {})

        summary = {"vector": {}, "graph": {}, "timeline": []}

        # ── a) Vector store ──────────────────────────────────────────────────
        # Papers: embed title + abstract for semantic search of academic evidence
        if papers:
            vs_papers = self._safe(
                "vector/papers",
                self._vs.add_papers,
                self._papers_for_vector_store(papers),
            )
            summary["vector"]["papers"] = (vs_papers or {}).get("added", 0)
            logger.info("[vector] papers added=%s", summary["vector"]["papers"])

        # Articles: embed title + summary for semantic search of investment evidence
        if articles:
            vs_articles = self._safe(
                "vector/articles",
                self._vs.add_articles,
                self._articles_for_vector_store(articles),
            )
            summary["vector"]["articles"] = (vs_articles or {}).get("added", 0)
            logger.info("[vector] articles added=%s", summary["vector"]["articles"])

        # ── b) Graph store ───────────────────────────────────────────────────
        # Always ensure the central Technology node exists first.
        # All other nodes will be linked to it.
        tech_added = self._safe(
            "graph/technology",
            self._graph.add_technology,
            name=query,
            description=f"Technology tracked by pipeline: {query}",
            first_seen_stage=self._infer_first_stage(papers, startups, articles, patents, mainstream),
        )
        summary["graph"]["technology"] = query if tech_added else None

        # Papers → Paper nodes linked to Technology
        paper_links = 0
        for paper in papers[:20]:   # Cap at 20 to keep graph traversals fast
            paper_id = paper.get("id", "")
            if not paper_id:
                continue
            added = self._safe(
                "graph/paper",
                self._graph.add_paper,
                arxiv_id=paper_id,
                title=paper.get("title", ""),
                year=paper.get("year"),
            )
            if added:
                linked = self._safe(
                    "graph/paper_link",
                    self._graph.link_paper_to_technology,
                    arxiv_id=paper_id,
                    technology_name=query,
                )
                if linked:
                    paper_links += 1
        summary["graph"]["paper_links"] = paper_links
        logger.info("[graph] paper→technology links: %d", paper_links)

        # Startups → Company nodes linked to Technology
        company_links = 0
        for startup in startups[:30]:   # Cap at 30
            name = startup.get("name", "")
            if not name:
                continue
            source = startup.get("source", "unknown")
            added = self._safe(
                "graph/company",
                self._graph.add_company,
                name=name,
                stage=2,
                source=source,
            )
            if added:
                linked = self._safe(
                    "graph/company_link",
                    self._graph.link_company_to_technology,
                    company_name=name,
                    technology_name=query,
                )
                if linked:
                    company_links += 1
        summary["graph"]["company_links"] = company_links
        logger.info("[graph] company→technology links: %d", company_links)

        # ── c) Timeline ──────────────────────────────────────────────────────
        # Record first-detection milestones in SQLite. These use INSERT OR IGNORE
        # so re-running the pipeline preserves the original detection date.

        # Determine which sources contributed data at each stage
        arxiv_papers = [p for p in papers if p.get("source") == "arxiv"]
        s2_papers    = [p for p in papers if p.get("source") == "semantic_scholar"]

        stage_events = []

        if arxiv_papers:
            stage_events.append((1, "arxiv_search", f"{len(arxiv_papers)} arXiv papers found"))
        if s2_papers:
            stage_events.append((1, "semantic_scholar_search", f"{len(s2_papers)} S2 papers found"))

        gh_startups = [s for s in startups if s.get("source") == "github"]
        ph_startups = [s for s in startups if s.get("source") == "producthunt"]
        yc_startups = [s for s in startups if s.get("source") == "ycombinator"]

        if gh_startups:
            stage_events.append((2, "github_search_repositories", f"{len(gh_startups)} repos found"))
        if ph_startups:
            stage_events.append((2, "producthunt_search", f"{len(ph_startups)} products found"))
        if yc_startups:
            stage_events.append((2, "yc_search_companies", f"{len(yc_startups)} YC companies found"))

        news_art = [a for a in articles if "newsapi" in a.get("source", "").lower()]
        tc_art   = [a for a in articles if "techcrunch" in a.get("source", "").lower()]

        if news_art:
            stage_events.append((3, "news_search_funding", f"{len(news_art)} funding articles found"))
        if tc_art:
            stage_events.append((3, "techcrunch_search_funding", f"{len(tc_art)} TechCrunch articles found"))

        if patents:
            stage_events.append((4, "patents_search", f"{len(patents)} patents filed"))

        wiki = mainstream.get("wikipedia", {})
        if wiki.get("exists"):
            views = wiki.get("page_views", {}).get("average_daily_views", 0)
            stage_events.append((5, "wikipedia_search", f"Article exists; avg {views:.0f} daily views"))

        trends = mainstream.get("trends", {})
        if trends.get("query_found"):
            stage_events.append((5, "trends_search", f"Matched: {trends.get('matched_trends', [])}"))

        for stage, source, notes in stage_events:
            result = self._safe(
                f"timeline/stage{stage}",
                self._sqlite.save_timeline_event,
                technology=query,
                stage=stage,
                source=source,
                notes=notes,
            )
            if result:
                summary["timeline"].append({"stage": stage, "source": source, "action": result.get("action")})

        logger.info("[timeline] %d events recorded", len(summary["timeline"]))
        return summary

    def _infer_first_stage(self, papers, startups, articles, patents, mainstream) -> int:
        """
        Infers the first pipeline stage at which this technology was detected,
        used as the first_seen_stage on the Technology graph node.

        Returns the lowest numbered stage that has any data — not the highest,
        because first_seen_stage records where the technology *started*, not
        where it currently is.
        """
        if papers:
            return 1
        if startups:
            return 2
        if articles:
            return 3
        if patents:
            return 4
        if mainstream.get("wikipedia", {}).get("exists") or mainstream.get("trends", {}).get("query_found"):
            return 5
        return 1  # Default — assume academic origin

    # ── 2. Scoring layer ─────────────────────────────────────────────────────

    def compute_stage_scores(self, cleaned_data: dict, query: str = "") -> dict:
        """
        Analyses cleaned data and returns a complete scores dict.

        Each per-tool score is 0–100. The overall_stage is 1–5.
        The overall_score is 0–100 (sum of five 0–20 stage contributions).

        Scoring is designed to be transparent and explainable — every number
        traces back to a count of real documents from a real source.

        Args:
            cleaned_data : Output of DataCleaner.clean_all().
            query        : Original search query — used to load cached patents
                           from MongoDB when the current fetch returned nothing.

        Returns:
            dict with keys matching the trend_scores SQLite table columns.
        """
        papers     = cleaned_data.get("academic",   [])
        startups   = cleaned_data.get("startup",    [])
        articles   = cleaned_data.get("investment", [])
        patents    = cleaned_data.get("bigtech",    [])
        mainstream = cleaned_data.get("mainstream", {})

        # ── Patent fallback: if the current fetch returned nothing (PatentsView
        # down, both fallbacks blocked), score using previously-stored patents
        # from MongoDB so Stage 4 isn't silently zeroed on API outages.
        if not patents and query:
            try:
                stored = self._mongo.find_by_query("patents", query, limit=20)
                if stored:
                    patents = self._cleaner.clean_patents(stored)
                    logger.info(
                        "patents | API fetch empty — scoring with %d cached patents "
                        "from MongoDB for query=%r",
                        len(patents), query,
                    )
            except Exception as exc:
                logger.warning("patents | MongoDB cache fallback failed: %s", exc)

        # ── Split by source ───────────────────────────────────────────────────
        arxiv_papers = [p for p in papers    if p.get("source") == "arxiv"]
        s2_papers    = [p for p in papers    if p.get("source") == "semantic_scholar"]
        gh_repos     = [s for s in startups  if s.get("source") == "github"]
        ph_posts     = [s for s in startups  if s.get("source") == "producthunt"]
        yc_cos       = [s for s in startups  if s.get("source") == "ycombinator"]
        # Use api_source (added by cleaner to preserve the tool tag) so that
        # a NewsAPI article whose source_name is "TechCrunch" is not
        # misattributed to tc_arts.  Falls back to source for older records.
        news_arts    = [a for a in articles  if a.get("api_source", a.get("source", "")) == "newsapi"]
        tc_arts      = [a for a in articles  if a.get("api_source", a.get("source", "")) == "techcrunch"]

        # ── Per-tool scores ───────────────────────────────────────────────────

        # Stage 1 — arXiv
        # Count score (max 70) + recency bonus (up to 30)
        arxiv_count  = _score_from_count(len(arxiv_papers), _SCALE["arxiv"]) * 0.70
        arxiv_recent = _recency_bonus(arxiv_papers, bonus=30.0)
        arxiv_score  = round(min(arxiv_count + arxiv_recent, 100.0), 1)

        # Stage 1 — Semantic Scholar
        # Count score (max 60) + citation quality bonus (max 20) + recency (max 20)
        s2_count   = _score_from_count(len(s2_papers), _SCALE["semantic_scholar"]) * 0.60
        s2_cite    = _citation_bonus(s2_papers, bonus=20.0)
        s2_recent  = _recency_bonus(s2_papers, bonus=20.0)
        s2_score   = round(min(s2_count + s2_cite + s2_recent, 100.0), 1)

        # Stage 2 — GitHub
        # Count score only (star counts not preserved in cleaned schema)
        github_score = round(_score_from_count(len(gh_repos), _SCALE["github"]), 1)

        # Stage 2 — ProductHunt
        producthunt_score = round(_score_from_count(len(ph_posts), _SCALE["producthunt"]), 1)

        # Stage 2 — YC
        # Count score (max 80) + batch recency bonus (max 20)
        # Recent YC batches (within last 3 years) indicate the space is still
        # being actively funded — more signal than historical YC activity.
        yc_count = _score_from_count(len(yc_cos), _SCALE["yc"]) * 0.80
        yc_score = round(min(yc_count + _recency_bonus(yc_cos, bonus=20.0), 100.0), 1)

        # Stage 3 — NewsAPI
        # Count score + funding relevance bonus
        # funding-relevant articles are worth more than generic news mentions
        news_base     = _score_from_count(len(news_arts), _SCALE["news"]) * 0.70
        news_funding  = sum(1 for a in news_arts if a.get("is_funding_relevant")) / max(len(news_arts), 1)
        news_score    = round(min(news_base + news_funding * 30, 100.0), 1)

        # Stage 3 — TechCrunch
        tc_base      = _score_from_count(len(tc_arts), _SCALE["techcrunch"]) * 0.70
        tc_funding   = sum(1 for a in tc_arts if a.get("is_funding_relevant")) / max(len(tc_arts), 1)
        tc_score     = round(min(tc_base + tc_funding * 30, 100.0), 1)

        # Stage 4 — Patents
        # Count score (max 70) + recency bonus (max 30)
        # Recent patents are a stronger Stage 4 signal than old ones —
        # a 2024 patent filing means current R&D, not legacy IP protection
        pat_count   = _score_from_count(len(patents), _SCALE["patents"]) * 0.70
        pat_recent  = _recency_bonus(patents, bonus=30.0)
        patents_score = round(min(pat_count + pat_recent, 100.0), 1)

        # Stage 5 — Wikipedia
        # Existence alone = 40; page view trend adds up to 40; high view volume adds 20
        wiki = mainstream.get("wikipedia", {})
        wiki_score = 0.0
        if wiki.get("exists"):
            wiki_score += 40.0
            views_data  = wiki.get("page_views", {})
            avg_views   = views_data.get("average_daily_views", 0) or 0
            trend       = views_data.get("trend", "no_data")
            # View volume: 1k daily views = moderate awareness, 10k = mainstream
            wiki_score += min((avg_views / 10_000) * 40, 40.0)
            if trend == "rising":
                wiki_score += 20.0
            elif trend == "stable":
                wiki_score += 10.0
        wiki_score = round(min(wiki_score, 100.0), 1)

        # Stage 5 — Google Trends
        # Binary: either the technology is trending right now or it isn't.
        # No partial credit — appearing in Google Trends realtime feed is a
        # high bar that means mass public interest *right now*.
        trends_data = mainstream.get("trends", {})
        trends_score = 100.0 if trends_data.get("query_found") else 0.0

        # ── Stage-combined scores (each 0–100, used for stage activation) ────
        stage_1 = round((arxiv_score + s2_score) / 2, 1)
        stage_2 = round(max(github_score, producthunt_score, yc_score), 1)
        stage_3 = round((news_score + tc_score) / 2, 1)         # newsapi + tc
        stage_4 = patents_score
        stage_5 = round(wiki_score * 0.6 + trends_score * 0.4, 1)

        # ── Overall stage ────────────────────────────────────────────────────
        # Walk stages highest to lowest; return the first one that is "active".
        # Stage 5 has higher thresholds because a Wikipedia article alone
        # doesn't mean mainstream — it needs meaningful view volume.
        overall_stage = 1  # always at least academic if any data was found
        if stage_5 >= _WIKI_ACTIVATION_THRESHOLD:
            overall_stage = 5
        elif stage_4 >= _ACTIVATION_THRESHOLD:
            overall_stage = 4
        elif stage_3 >= _ACTIVATION_THRESHOLD:
            overall_stage = 3
        elif stage_2 >= _ACTIVATION_THRESHOLD:
            overall_stage = 2
        elif stage_1 >= _ACTIVATION_THRESHOLD:
            overall_stage = 1

        # ── Overall score (0–100) ────────────────────────────────────────────
        # Each stage contributes exactly 20 points maximum.
        # A technology advances its overall_score by demonstrating signal at
        # progressively later pipeline stages.
        overall_score = round(
            (stage_1 / 100) * 20 +
            (stage_2 / 100) * 20 +
            (stage_3 / 100) * 20 +
            (stage_4 / 100) * 20 +
            (stage_5 / 100) * 20,
            1,
        )

        scores = {
            # Per-tool scores
            "arxiv_score":            arxiv_score,
            "semantic_scholar_score": s2_score,
            "github_score":           github_score,
            "producthunt_score":      producthunt_score,
            "yc_score":               yc_score,
            "news_score":             news_score,
            "techcrunch_score":       tc_score,
            "patents_score":          patents_score,
            "wikipedia_score":        wiki_score,
            "trends_score":           trends_score,
            # Stage-combined (not stored in SQLite but useful for agents)
            "_stage_1": stage_1,
            "_stage_2": stage_2,
            "_stage_3": stage_3,
            "_stage_4": stage_4,
            "_stage_5": stage_5,
            # Summary
            "overall_stage": overall_stage,
            "overall_score": overall_score,
        }

        logger.info(
            "compute_stage_scores | stage=%d | score=%.1f | "
            "arxiv=%.0f s2=%.0f gh=%.0f ph=%.0f yc=%.0f "
            "news=%.0f tc=%.0f pat=%.0f wiki=%.0f trends=%.0f",
            overall_stage, overall_score,
            arxiv_score, s2_score, github_score, producthunt_score, yc_score,
            news_score, tc_score, patents_score, wiki_score, trends_score,
        )
        return scores

    # ── 3. Velocity analysis ──────────────────────────────────────────────────

    def calculate_velocity(self, query: str) -> dict:
        """
        Measures how fast a technology is moving through the pipeline stages
        by comparing data volumes across time windows stored in MongoDB and
        SQLite.

        Four independent signals are computed:

          academic_velocity  — year-on-year paper count change in MongoDB
                               (arxiv + S2 combined, using published / year field)
          startup_velocity   — older-half vs newer-half monthly repo creation
                               rate in MongoDB github_repos
          news_velocity      — article count: last 30 days vs prior 30 days
                               in MongoDB news collection
          overall_velocity   — overall_score progression across SQLite
                               trend_scores history rows

        Each signal is labelled:
          "accelerating"  — growth > +20 %
          "stable"        — growth within ±20 %
          "decelerating"  — growth < −20 %

        estimated_next_stage_months uses the SPECIFIC signal scores for the
        next stage (e.g. news_score if current stage is 2) and their growth
        rate from SQLite history to estimate when they will cross the stage
        activation threshold.  Returns None when there is insufficient
        history, the signal is decelerating, or the estimate exceeds 5 years.
        """
        _ACCEL = "accelerating"
        _STABLE = "stable"
        _DECEL  = "decelerating"

        def _label(pct: float) -> str:
            return _ACCEL if pct > 20 else (_DECEL if pct < -20 else _STABLE)

        now = datetime.now(timezone.utc)

        # ── 1. Academic velocity (MongoDB papers, group by year) ─────────────
        academic_velocity  = _STABLE
        academic_growth_rate = 0.0
        try:
            papers = self._mongo.find_by_query("papers", query, limit=300)
            by_year: dict[int, int] = {}
            for p in papers:
                yr: int | None = None
                if isinstance(p.get("year"), int):
                    yr = p["year"]
                else:
                    raw_date = p.get("published") or p.get("date") or ""
                    if len(str(raw_date)) >= 4:
                        try:
                            yr = int(str(raw_date)[:4])
                        except ValueError:
                            pass
                if yr and 2015 <= yr <= now.year:
                    by_year[yr] = by_year.get(yr, 0) + 1

            y_old = now.year - 2   # e.g. 2023 for current year 2025
            y_new = now.year - 1   # e.g. 2024
            c_old = by_year.get(y_old, 0)
            c_new = by_year.get(y_new, 0)
            if c_old > 0:
                academic_growth_rate = round(((c_new - c_old) / c_old) * 100, 1)
            elif c_new > 0:
                academic_growth_rate = 100.0   # appeared from nothing
            academic_velocity = _label(academic_growth_rate)
            logger.debug("velocity | academic by_year=%s growth=%.1f%%", by_year, academic_growth_rate)
        except Exception as exc:
            logger.warning("velocity | academic query failed: %s", exc)

        # ── 2. Startup velocity (MongoDB github_repos, monthly repo creation) ─
        startup_velocity = _STABLE
        try:
            repos = self._mongo.find_by_query("github_repos", query, limit=200)
            months: dict[str, int] = {}
            for r in repos:
                created = r.get("created_at") or r.get("date") or ""
                if len(str(created)) >= 7:
                    ym = str(created)[:7]   # "YYYY-MM"
                    months[ym] = months.get(ym, 0) + 1

            if len(months) >= 4:
                sorted_ym = sorted(months.keys())
                mid = len(sorted_ym) // 2
                older_avg = sum(months[m] for m in sorted_ym[:mid]) / mid
                newer_avg = sum(months[m] for m in sorted_ym[mid:]) / (len(sorted_ym) - mid)
                if older_avg > 0:
                    startup_growth = ((newer_avg - older_avg) / older_avg) * 100
                    startup_velocity = _label(startup_growth)
                elif newer_avg > 0:
                    startup_velocity = _ACCEL
        except Exception as exc:
            logger.warning("velocity | startup query failed: %s", exc)

        # ── 3. News velocity (last 30 days vs prior 30 days) ─────────────────
        news_velocity = _STABLE
        try:
            articles = self._mongo.find_by_query("news", query, limit=300)
            cutoff_recent = now - timedelta(days=30)
            cutoff_prior  = now - timedelta(days=60)
            recent_count = prior_count = 0
            for a in articles:
                pub_raw = (
                    a.get("publishedAt") or a.get("published_at")
                    or a.get("date") or a.get("fetched_at") or ""
                )
                if not pub_raw or len(str(pub_raw)) < 10:
                    continue
                try:
                    pub_dt = datetime.fromisoformat(str(pub_raw)[:19].replace("Z", "+00:00"))
                    if pub_dt.tzinfo is None:
                        pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
                if pub_dt >= cutoff_recent:
                    recent_count += 1
                elif pub_dt >= cutoff_prior:
                    prior_count += 1

            if prior_count > 0:
                news_growth = ((recent_count - prior_count) / prior_count) * 100
                news_velocity = _label(news_growth)
            elif recent_count > 0:
                news_velocity = _ACCEL
        except Exception as exc:
            logger.warning("velocity | news query failed: %s", exc)

        # ── 4. Overall velocity (SQLite trend_scores history) ────────────────
        overall_velocity      = _STABLE
        score_growth_per_month = 0.0
        months_elapsed        = 1.0
        history: list[dict]   = []
        current_stage         = 1
        current_score         = 0.0

        try:
            history = self._sqlite.get_trend_history(query, limit=10)
            if len(history) >= 2:
                newest = history[0]
                oldest = history[-1]
                current_stage = newest.get("overall_stage", 1)
                current_score = float(newest.get("overall_score", 0))
                score_delta   = current_score - float(oldest.get("overall_score", 0))

                ts_new = newest.get("timestamp", "")
                ts_old = oldest.get("timestamp", "")
                if ts_new and ts_old:
                    try:
                        dt_new = datetime.fromisoformat(str(ts_new)[:19])
                        dt_old = datetime.fromisoformat(str(ts_old)[:19])
                        days = max(1, (dt_new - dt_old).days)
                        months_elapsed = max(0.5, days / 30.0)
                    except ValueError:
                        pass

                score_growth_per_month = score_delta / months_elapsed
                overall_velocity = (
                    _ACCEL if score_growth_per_month > 2
                    else _DECEL if score_growth_per_month < -2
                    else _STABLE
                )
            elif len(history) == 1:
                current_stage = history[0].get("overall_stage", 1)
                current_score = float(history[0].get("overall_score", 0))
        except Exception as exc:
            logger.warning("velocity | SQLite history query failed: %s", exc)

        # ── 5. Estimated months to next stage ────────────────────────────────
        # Uses the specific signal columns for the NEXT stage and their
        # growth rate from SQLite history, so the estimate is tied to the
        # actual signals that must activate — not just the overall_score.
        #
        # Columns to watch per current stage:
        #   Stage 1 (academic)   → watch github_score, producthunt_score, yc_score (threshold 30)
        #   Stage 2 (developer)  → watch news_score, techcrunch_score (threshold 30)
        #   Stage 3 (investment) → watch patents_score (threshold 30)
        #   Stage 4 (big tech)   → watch wikipedia_score (threshold 50), trends_score (60)
        _NEXT_STAGE_SIGNALS: dict[int, list[str]] = {
            1: ["github_score", "producthunt_score", "yc_score"],
            2: ["news_score", "techcrunch_score"],
            3: ["patents_score"],
            4: ["wikipedia_score", "trends_score"],
        }
        _NEXT_STAGE_THRESHOLD: dict[int, float] = {
            1: 30.0, 2: 30.0, 3: 30.0, 4: 50.0,
        }

        estimated_next_stage_months: int | None = None
        try:
            if len(history) >= 2 and current_stage < 5:
                sig_cols  = _NEXT_STAGE_SIGNALS.get(current_stage, [])
                threshold = _NEXT_STAGE_THRESHOLD.get(current_stage, 30.0)

                newest_sig = max(
                    (float(history[0].get(c, 0) or 0) for c in sig_cols),
                    default=0.0,
                )
                oldest_sig = max(
                    (float(history[-1].get(c, 0) or 0) for c in sig_cols),
                    default=0.0,
                )
                sig_growth_per_month = (newest_sig - oldest_sig) / months_elapsed
                points_needed = max(0.0, threshold - newest_sig)

                if points_needed == 0:
                    estimated_next_stage_months = 0
                elif sig_growth_per_month > 0:
                    raw_months = points_needed / sig_growth_per_month
                    estimated_next_stage_months = (
                        round(raw_months) if raw_months <= 60 else None
                    )
        except Exception as exc:
            logger.warning("velocity | next-stage estimation failed: %s", exc)

        # ── 6. One-sentence summary ───────────────────────────────────────────
        _DIRECTION = {
            _ACCEL: "rapidly gaining momentum",
            _STABLE: "holding steady",
            _DECEL:  "losing momentum",
        }
        # Dominant direction: bias toward acceleration when any signal is up
        dominant = overall_velocity
        if overall_velocity == _STABLE and (
            academic_velocity == _ACCEL or news_velocity == _ACCEL
        ):
            dominant = _ACCEL

        parts = [f"This technology is {_DIRECTION[dominant]}"]
        if academic_velocity == _ACCEL and academic_growth_rate > 0:
            parts.append(
                f"academic output up {academic_growth_rate:.0f}% year-over-year"
            )
        if news_velocity == _ACCEL:
            parts.append("press coverage accelerating")
        elif news_velocity == _DECEL:
            parts.append("press coverage cooling")
        if estimated_next_stage_months is not None and estimated_next_stage_months > 0:
            parts.append(
                f"estimated Stage {current_stage + 1} transition in "
                f"~{estimated_next_stage_months} months"
            )
        velocity_summary = " — ".join(parts[:3]) + "."

        result = {
            "academic_velocity":           academic_velocity,
            "academic_growth_rate":        academic_growth_rate,
            "startup_velocity":            startup_velocity,
            "news_velocity":               news_velocity,
            "overall_velocity":            overall_velocity,
            "score_growth_per_month":      round(score_growth_per_month, 2),
            "estimated_next_stage_months": estimated_next_stage_months,
            "velocity_summary":            velocity_summary,
        }
        logger.info(
            "calculate_velocity | query=%r | overall=%s | academic=%s(%.0f%%) "
            "startup=%s | news=%s | next_stage=%s months",
            query, overall_velocity, academic_velocity, academic_growth_rate,
            startup_velocity, news_velocity, str(estimated_next_stage_months),
        )
        return result

    # ── 4. Full pipeline ──────────────────────────────────────────────────────

    def run_full_pipeline(self, query: str, progress_callback=None) -> dict:
        """
        End-to-end pipeline: fetch → clean → embed/store → score.

        This is the single method the analyst agent calls. It returns a complete
        result dict containing raw data, cleaned data, storage summaries, and
        scores — everything the agent needs to write a grounded pipeline
        assessment with citations.

        Execution order:
          1. DataFetcher.fetch_all()       — hits all APIs and scrapers
          2. DataCleaner.clean_all()       — normalises and deduplicates
          3. embed_and_store()             — writes to vector/graph/sqlite
          4. compute_stage_scores()        — scores each stage 0–100
          5. sqlite_db.save_trend_score()  — persists scores for history

        Args:
            query             : Technology or concept to analyse.
            progress_callback : Optional callable(str) for streaming progress
                                messages to the UI during the pipeline run.

        Returns:
            dict:
                query          (str)  : Original query.
                fetched_at     (str)  : ISO 8601 UTC timestamp.
                raw            (dict) : Output of fetch_all() for full provenance.
                cleaned        (dict) : Output of clean_all() — normalised docs.
                storage        (dict) : Storage layer write summaries.
                scores         (dict) : All 10 per-tool scores + overall stage/score.
                score_row_id   (int)  : SQLite rowid of the saved score row.

        Raises:
            ValueError: If query is empty.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        def _cb(msg: str) -> None:
            if progress_callback:
                progress_callback(msg)

        query = query.strip()
        started = datetime.now(timezone.utc).isoformat()
        logger.info("=" * 60)
        logger.info("run_full_pipeline | query=%r | started %s", query, started)
        logger.info("=" * 60)

        # 1. Fetch
        raw = self._fetcher.fetch_all(query, progress_callback=progress_callback)

        # 2. Clean
        _cb("Computing embeddings...")
        cleaned = self._cleaner.clean_all(raw)

        # 3. Embed and store
        storage = self.embed_and_store(cleaned, query)

        # 4. Score
        scores = self.compute_stage_scores(cleaned, query=query)

        # 5. Persist scores to SQLite
        score_row_id = self._safe(
            "sqlite/save_trend_score",
            self._sqlite.save_trend_score,
            query=query,
            scores_dict=scores,
        )

        # 6. Velocity — runs AFTER save_trend_score so the row we just wrote
        #    is available in the SQLite history for growth-rate comparison.
        velocity = self._safe(
            "calculate_velocity",
            self.calculate_velocity,
            query,
        ) or {
            "academic_velocity":           "stable",
            "academic_growth_rate":        0.0,
            "startup_velocity":            "stable",
            "news_velocity":               "stable",
            "overall_velocity":            "stable",
            "score_growth_per_month":      0.0,
            "estimated_next_stage_months": None,
            "velocity_summary":            "Insufficient history to calculate velocity.",
        }

        finished = datetime.now(timezone.utc).isoformat()
        logger.info(
            "run_full_pipeline DONE | query=%r | stage=%d | score=%.1f | "
            "velocity=%s | finished %s",
            query, scores["overall_stage"], scores["overall_score"],
            velocity["overall_velocity"], finished,
        )

        return {
            "query":        query,
            "fetched_at":   started,
            "finished_at":  finished,
            "raw":          raw,
            "cleaned":      cleaned,
            "storage":      storage,
            "scores":       scores,
            "velocity":     velocity,
            "score_row_id": score_row_id,
        }


# ---------------------------------------------------------------------------
# Module-level convenience instance
# ---------------------------------------------------------------------------

embedder = DataEmbedder()


# ---------------------------------------------------------------------------
# Quick smoke-test — uses synthetic cleaned data (no live API calls)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    # Synthetic cleaned data mirroring DataCleaner.clean_all() output
    SYNTHETIC = {
        "query":      "diffusion models",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "academic": [
            {"id": "2006.11239", "title": "DDPM",   "abstract": "Denoising diffusion...",
             "year": 2020, "citation_count": 8000, "source": "arxiv",            "url": "https://arxiv.org/abs/2006.11239", "cleaned_text": "DDPM denoising diffusion"},
            {"id": "2112.10752", "title": "LDM",    "abstract": "Latent diffusion...",
             "year": 2021, "citation_count": 5000, "source": "arxiv",            "url": "https://arxiv.org/abs/2112.10752", "cleaned_text": "LDM latent diffusion"},
            {"id": "s2abc123",   "title": "Score",  "abstract": "Score-based...",
             "year": 2019, "citation_count": 3000, "source": "semantic_scholar", "url": "https://semanticscholar.org/s2abc123", "cleaned_text": "score based generative"},
        ],
        "startup": [
            {"id": "https://github.com/CompVis/stable-diffusion", "name": "CompVis/stable-diffusion",
             "description": "SD implementation", "source": "github",       "stage_signal": 2,
             "url": "https://github.com/CompVis/stable-diffusion", "date": "2022-08-22", "tags": "diffusion, generative", "cleaned_text": "stable diffusion"},
            {"id": "https://www.producthunt.com/posts/midjourney", "name": "Midjourney",
             "description": "AI art generation", "source": "producthunt",  "stage_signal": 2,
             "url": "https://www.producthunt.com/posts/midjourney", "date": "2022-07-01", "tags": "AI, art", "cleaned_text": "midjourney ai art"},
            {"id": "https://www.ycombinator.com/companies/stability-ai", "name": "Stability AI",
             "description": "Foundation models",  "source": "ycombinator", "stage_signal": 2,
             "url": "https://www.ycombinator.com/companies/stability-ai", "date": "2023-01-01", "tags": "AI, generative", "cleaned_text": "stability ai foundation"},
        ],
        "investment": [
            {"id": "https://tc.com/stability-raises", "title": "Stability AI raises $101M",
             "summary": "Series A round closes",  "source": "techcrunch", "url": "https://tc.com/stability-raises",
             "date": "2022-10-17", "is_funding_relevant": True, "cleaned_text": "stability raises series a"},
            {"id": "https://news.com/diffusion-vc",   "title": "VC money flows into diffusion",
             "summary": "Investors pour into AI art", "source": "newsapi", "url": "https://news.com/diffusion-vc",
             "date": "2023-03-01", "is_funding_relevant": True, "cleaned_text": "vc diffusion ai investment"},
        ],
        "bigtech": [
            {"id": "US11423293", "title": "Image generation via diffusion", "abstract": "Neural network image gen",
             "assignees": "Google LLC", "date": "2023-01-15", "url": "https://patents.google.com/patent/US11423293",
             "source": "patentsview", "cleaned_text": "image generation diffusion google"},
        ],
        "mainstream": {
            "wikipedia": {
                "exists": True, "title": "Diffusion model",
                "page_url": "https://en.wikipedia.org/wiki/Diffusion_model",
                "page_views": {"average_daily_views": 4500, "total_views": 135000, "trend": "rising"},
            },
            "trends": {"query_found": False, "matched_trends": [], "all_trends": []},
        },
        "summary": {},
    }

    e = DataEmbedder()

    print("\n=== compute_stage_scores (synthetic data) ===")
    scores = e.compute_stage_scores(SYNTHETIC)
    print(f"  arxiv_score            : {scores['arxiv_score']}")
    print(f"  semantic_scholar_score : {scores['semantic_scholar_score']}")
    print(f"  github_score           : {scores['github_score']}")
    print(f"  producthunt_score      : {scores['producthunt_score']}")
    print(f"  yc_score               : {scores['yc_score']}")
    print(f"  news_score             : {scores['news_score']}")
    print(f"  techcrunch_score       : {scores['techcrunch_score']}")
    print(f"  patents_score          : {scores['patents_score']}")
    print(f"  wikipedia_score        : {scores['wikipedia_score']}")
    print(f"  trends_score           : {scores['trends_score']}")
    print(f"  ─────────────────────────────")
    print(f"  overall_stage          : {scores['overall_stage']}")
    print(f"  overall_score          : {scores['overall_score']}")

    print("\n=== embed_and_store (vector only — Neo4j/SQLite optional) ===")
    storage = e.embed_and_store(SYNTHETIC, "diffusion models")
    print(f"  vector : {storage['vector']}")
    print(f"  graph  : {storage['graph']}")
    print(f"  timeline events: {len(storage['timeline'])}")
    for ev in storage["timeline"]:
        print(f"    stage={ev['stage']} source={ev['source']} action={ev['action']}")

    print("\nSmoke-test complete.")
