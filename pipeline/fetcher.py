"""
Pipeline orchestration layer — DataFetcher.

This module is the single entry point for running a full pipeline fetch.
It imports every tool and scraper, calls them in the correct order for each
pipeline stage, stores results in MongoDB, and returns a unified result dict.

Architecture position
─────────────────────
    tools/          ← individual data-source tools (one API each)
    scrapers/       ← web scrapers (YC, TechCrunch)
         ↓
    pipeline/fetcher.py   ← THIS FILE — orchestrates all tools
         ↓
    database/mongo_client.py  ← stores raw documents
    database/sqlite_client.py ← stores scores (written by analyst agent)
    database/vector_store.py  ← stores embeddings (written here for papers/articles)

Stage mapping
─────────────
  Stage 1 — Academic       : fetch_academic()    → arXiv + Semantic Scholar
  Stage 2 — Developer      : fetch_startup_signals() → GitHub + ProductHunt + YC
  Stage 3 — Investment     : fetch_investment_signals() → NewsAPI + TechCrunch
  Stage 4 — Big Tech       : fetch_bigtech_signals()  → PatentsView
  Stage 5 — Mainstream     : fetch_mainstream_signals() → Wikipedia + Google Trends
                              (not stored in MongoDB — these are point-in-time
                              live signals, not discrete historical documents)

Fault tolerance
───────────────
Individual tool failures are isolated: if the ProductHunt API is down,
GitHub and YC still run and their results are still stored. Each tool call
is wrapped in _run_tool(), which catches all exceptions, logs them with the
tool name, and returns an empty result so the pipeline continues.

This matters because the system must be able to produce *some* answer even
when one or two data sources are temporarily unavailable. An empty list from
a tool is a valid signal (no data found) — a crash is not.

Usage:
    from pipeline.fetcher import DataFetcher
    fetcher = DataFetcher()
    results = fetcher.fetch_all("diffusion models")
    # results["academic"]    → list of papers
    # results["startup"]     → dict with github_repos, producthunt, yc_companies
    # results["investment"]  → dict with news, techcrunch
    # results["bigtech"]     → list of patents
    # results["mainstream"]  → dict with wikipedia, trends
"""

import logging
import sys
import os
from datetime import datetime, timezone
from pathlib import Path

# Make project root importable regardless of how this module is invoked
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── Tool imports ─────────────────────────────────────────────────────────────
from tools.arxiv_tool             import search_arxiv
from tools.semantic_scholar_tool  import search_papers          as search_semantic_scholar
from tools.github_tool            import search_repositories    as search_github_repos
from tools.github_tool            import get_repo_activity
from tools.producthunt_tool       import search_producthunt
from tools.news_tool              import search_funding_news
from tools.patents_tool           import search_patents
from tools.wikipedia_tool         import search_wikipedia, get_page_views
from tools.trends_tool            import search_trends

# ── Scraper imports ──────────────────────────────────────────────────────────
from scrapers.yc_scraper          import search_yc_companies
from scrapers.techcrunch_scraper  import search_funding_articles as search_techcrunch

# ── Database imports ─────────────────────────────────────────────────────────
from database.mongo_client        import MongoDBClient
from database.vector_store        import VectorStore

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fetcher")


# ---------------------------------------------------------------------------
# DataFetcher
# ---------------------------------------------------------------------------

class DataFetcher:
    """
    Orchestrates all tool and scraper calls for a given technology query,
    stores raw results in MongoDB, populates the vector store for RAG,
    and returns a unified result dict.

    Each public fetch method corresponds to one pipeline stage and can be
    called independently (for partial refreshes) or together via fetch_all().
    """

    def __init__(
        self,
        mongo_client: MongoDBClient | None = None,
        vector_store: VectorStore  | None = None,
    ) -> None:
        """
        Args:
            mongo_client : Optional pre-constructed MongoDBClient.
                           Defaults to a new client using MONGO_URI from .env.
            vector_store : Optional pre-constructed VectorStore.
                           Defaults to a new store using CHROMA_DB_PATH from .env.
        """
        self._mongo = mongo_client or MongoDBClient()
        self._vs    = vector_store or VectorStore()

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _run_tool(
        self,
        tool_name: str,
        fn,
        *args,
        **kwargs,
    ):
        """
        Calls a tool function with fault isolation.

        Wraps every tool call so that a single failing API or scraper does
        not abort the entire fetch. Returns the tool's result on success,
        or None on any exception. The caller decides what "None" means for
        its stage (usually treated as an empty list or empty dict).

        Args:
            tool_name : Human-readable name used in log messages.
            fn        : The tool function to call.
            *args     : Positional arguments forwarded to fn.
            **kwargs  : Keyword arguments forwarded to fn.

        Returns:
            The tool's return value, or None if it raised any exception.
        """
        try:
            result = fn(*args, **kwargs)
            return result
        except EnvironmentError as e:
            # Missing API key — not retryable, inform clearly
            logger.error("[%s] Missing credentials: %s", tool_name, e)
        except PermissionError as e:
            # robots.txt blocked scraping
            logger.warning("[%s] Scraping not allowed: %s", tool_name, e)
        except RuntimeError as e:
            # Network errors, HTTP errors, API rate limits
            logger.error("[%s] Runtime error: %s", tool_name, e)
        except Exception as e:
            # Catch-all — log with traceback for unexpected failures
            logger.exception("[%s] Unexpected error: %s", tool_name, e)
        return None

    def _store(self, collection: str, docs: list[dict], tool_name: str) -> int:
        """
        Stores a list of documents in MongoDB and returns the count stored.

        Wraps MongoDBClient.insert_documents() with fault isolation so a
        MongoDB write failure does not abort the in-memory results.

        Returns:
            Number of documents passed to insert_documents (not upserted —
            the caller cares about how much data was fetched, not dedup counts).
        """
        if not docs:
            return 0
        try:
            summary = self._mongo.insert_documents(collection, docs)
            logger.debug(
                "Stored %d docs in '%s' (inserted=%d updated=%d skipped=%d)",
                len(docs), collection,
                summary["inserted"], summary["updated"], summary["skipped"],
            )
        except Exception as e:
            logger.error("MongoDB write failed for '%s': %s", collection, e)
        return len(docs)

    # ── Stage 1: Academic ────────────────────────────────────────────────────

    def fetch_academic(self, query: str) -> list[dict]:
        """
        Stage 1 — Academic research signal.

        Calls arXiv and Semantic Scholar, merges results into a single list,
        stores everything in the "papers" MongoDB collection, and adds papers
        to the vector store so the analyst agent can do semantic RAG search
        over abstracts.

        Why both sources?
          arXiv provides breadth — it indexes virtually every CS/physics/bio
          preprint. Semantic Scholar provides depth — it adds citation counts
          and influential citation counts that arXiv lacks. Together they give
          both volume (are papers being published?) and influence (are those
          papers being cited?).

        MongoDB collection: "papers"
        Vector store collection: "papers"

        Returns:
            Combined list of paper dicts from both sources, with source field
            set to "arxiv" or "semantic_scholar" respectively.
        """
        logger.info("fetch_academic | query=%r", query)
        papers: list[dict] = []

        # ── arXiv ────────────────────────────────────────────────────────────
        arxiv_results = self._run_tool(
            "arxiv_search",
            search_arxiv,
            query=query,
            max_results=15,
            sort_by="submittedDate",
        )
        if arxiv_results:
            papers.extend(arxiv_results)
            logger.info("[arxiv] %d papers returned", len(arxiv_results))
        else:
            logger.warning("[arxiv] No results or tool failed for query=%r", query)

        # ── Semantic Scholar ─────────────────────────────────────────────────
        s2_results = self._run_tool(
            "semantic_scholar_search",
            search_semantic_scholar,
            query=query,
            limit=15,
        )
        if s2_results:
            papers.extend(s2_results)
            logger.info("[semantic_scholar] %d papers returned", len(s2_results))
        else:
            logger.warning("[semantic_scholar] No results or tool failed for query=%r", query)

        # ── Store ────────────────────────────────────────────────────────────
        stored = self._store("papers", papers, "fetch_academic")
        logger.info("fetch_academic | total=%d | stored=%d", len(papers), stored)

        # ── Embed for RAG ────────────────────────────────────────────────────
        # Papers are the primary RAG source — the analyst agent queries the
        # vector store to cite relevant abstracts in its pipeline assessment.
        if papers:
            try:
                vs_result = self._vs.add_papers(papers)
                logger.info(
                    "fetch_academic | vector store: added=%d skipped=%d",
                    vs_result["added"], vs_result["skipped"],
                )
            except Exception as e:
                logger.error("Vector store write failed for papers: %s", e)

        return papers

    # ── Stage 2: Developer / Startup ─────────────────────────────────────────

    def fetch_startup_signals(self, query: str) -> dict:
        """
        Stage 2 — Developer community and early startup signals.

        Calls GitHub (repo activity), ProductHunt (product launches), and
        the YC scraper (portfolio companies). Each result set goes into its
        own MongoDB collection — the collections are distinct because they
        represent different sub-signals within Stage 2:

          github_repos   → developer community is building with the technology
          startups       → commercial products exist (ProductHunt validation)
          yc_companies   → accelerator-backed companies (strongest startup signal)

        Returns:
            dict with keys "github_repos", "producthunt", "yc_companies",
            each containing the respective list of results (or [] on failure).
        """
        logger.info("fetch_startup_signals | query=%r", query)

        # ── GitHub ───────────────────────────────────────────────────────────
        github_results = self._run_tool(
            "github_search_repositories",
            search_github_repos,
            query=query,
            sort="stars",
            limit=20,
        )
        github_results = github_results or []
        stored = self._store("github_repos", github_results, "github_search")
        logger.info("[github] %d repos returned | %d stored", len(github_results), stored)

        # Also fetch the aggregate activity summary — logged but not stored
        # since it's a derived dict, not a storable document list.
        github_activity = self._run_tool(
            "github_repo_activity",
            get_repo_activity,
            query=query,
            limit=20,
        )
        if github_activity:
            logger.info(
                "[github_activity] avg_stars=%.1f | total_stars=%d | repos_sampled=%d",
                github_activity.get("avg_stars", 0),
                github_activity.get("total_stars", 0),
                github_activity.get("repos_sampled", 0),
            )

        # ── ProductHunt ──────────────────────────────────────────────────────
        ph_results = self._run_tool(
            "producthunt_search",
            search_producthunt,
            query=query,
            limit=15,
        )
        ph_results = ph_results or []
        stored = self._store("startups", ph_results, "producthunt_search")
        logger.info("[producthunt] %d posts returned | %d stored", len(ph_results), stored)

        # ── YC Scraper ───────────────────────────────────────────────────────
        yc_results = self._run_tool(
            "yc_search_companies",
            search_yc_companies,
            query=query,
            limit=30,
        )
        yc_results = yc_results or []
        stored = self._store("yc_companies", yc_results, "yc_search")
        logger.info("[yc] %d companies returned | %d stored", len(yc_results), stored)

        total = len(github_results) + len(ph_results) + len(yc_results)
        logger.info(
            "fetch_startup_signals | total=%d (github=%d ph=%d yc=%d)",
            total, len(github_results), len(ph_results), len(yc_results),
        )

        return {
            "github_repos":  github_results,
            "producthunt":   ph_results,
            "yc_companies":  yc_results,
            "github_activity": github_activity or {},
        }

    # ── Stage 3: Investment ───────────────────────────────────────────────────

    def fetch_investment_signals(self, query: str) -> dict:
        """
        Stage 3 — VC investment and press coverage signals.

        Calls NewsAPI (funding vocabulary search) and the TechCrunch scraper
        (funding article search). Both target investment-phase press coverage
        but from different angles:

          news_tool       → broad funding news (all tech press via NewsAPI)
          techcrunch_scraper → TechCrunch-specific (the canonical VC announcement
                               venue — richer funding signal per article)

        Articles are also embedded in the vector store so the analyst agent
        can semantically search for investment evidence during RAG.

        MongoDB collections: "news", "techcrunch_articles"

        Returns:
            dict with keys "news_articles" and "techcrunch_articles".
        """
        logger.info("fetch_investment_signals | query=%r", query)

        # ── NewsAPI ──────────────────────────────────────────────────────────
        news_results = self._run_tool(
            "news_search_funding",
            search_funding_news,
            query=query,
            days_back=30,
            page_size=15,
        )
        news_results = news_results or []
        stored = self._store("news", news_results, "news_search")
        logger.info("[newsapi] %d articles returned | %d stored", len(news_results), stored)

        # ── TechCrunch ───────────────────────────────────────────────────────
        tc_results = self._run_tool(
            "techcrunch_search_funding",
            search_techcrunch,
            query=query,
            limit=15,
        )
        tc_results = tc_results or []
        stored = self._store("techcrunch_articles", tc_results, "techcrunch_search")
        logger.info("[techcrunch] %d articles returned | %d stored", len(tc_results), stored)

        # ── Embed articles for RAG ────────────────────────────────────────────
        all_articles = news_results + tc_results
        if all_articles:
            try:
                vs_result = self._vs.add_articles(all_articles)
                logger.info(
                    "fetch_investment_signals | vector store: added=%d skipped=%d",
                    vs_result["added"], vs_result["skipped"],
                )
            except Exception as e:
                logger.error("Vector store write failed for articles: %s", e)

        total = len(news_results) + len(tc_results)
        logger.info(
            "fetch_investment_signals | total=%d (news=%d tc=%d)",
            total, len(news_results), len(tc_results),
        )

        return {
            "news_articles":      news_results,
            "techcrunch_articles": tc_results,
        }

    # ── Stage 4: Big Tech / Institutional ────────────────────────────────────

    def fetch_bigtech_signals(self, query: str) -> list[dict]:
        """
        Stage 4 — Institutional adoption signal via patent filings.

        Calls PatentsView for US patent data. Patent filings from large
        corporations (Google, Microsoft, Apple, Meta, Amazon) on a technology
        are the clearest Stage 4 signal: these companies only file patents on
        technology they plan to productise or defensively protect.

        A surge of corporate patent filings 12–18 months after startup funding
        activity is the canonical pattern indicating a technology has entered
        the big-tech absorption phase.

        MongoDB collection: "patents"

        Returns:
            List of patent dicts.
        """
        logger.info("fetch_bigtech_signals | query=%r", query)

        patents = self._run_tool(
            "patents_search",
            search_patents,
            query=query,
            max_results=20,
        )
        patents = patents or []
        stored = self._store("patents", patents, "patents_search")
        logger.info("[patents] %d patents returned | %d stored", len(patents), stored)

        return patents

    # ── Stage 5: Mainstream ───────────────────────────────────────────────────

    def fetch_mainstream_signals(self, query: str) -> dict:
        """
        Stage 5 — Mainstream public awareness signals.

        Calls Wikipedia (article existence + page views) and trendspyg
        (Google Trends real-time trending check).

        These signals are NOT stored in MongoDB — they are ephemeral,
        point-in-time snapshots of current public interest. Storing them
        would create a misleading historical record (Wikipedia view counts
        change daily; Google Trends data expires). Instead, the analyst
        agent reads them directly from this return value.

        The SQLite score table is the right place to persist the *score*
        derived from these signals — the sqlite_client.save_trend_score()
        call happens in the analyst agent, not here.

        Returns:
            dict with keys:
                "wikipedia"  → dict from search_wikipedia() +
                               dict from get_page_views() (or {} if page not found)
                "trends"     → dict from search_trends()
        """
        logger.info("fetch_mainstream_signals | query=%r", query)

        # ── Wikipedia ────────────────────────────────────────────────────────
        wiki_summary = self._run_tool(
            "wikipedia_search",
            search_wikipedia,
            query=query,
        )
        wiki_summary = wiki_summary or {"exists": False}

        wiki_views: dict = {}
        if wiki_summary.get("exists"):
            # Only fetch page view stats if the article exists — avoids a
            # ValueError from get_page_views() when no article is found
            wiki_views = self._run_tool(
                "wikipedia_page_views",
                get_page_views,
                query=query,
                days_back=30,
            ) or {}
            logger.info(
                "[wikipedia] article='%s' | avg_views=%.1f | trend=%s",
                wiki_summary.get("title", ""),
                wiki_views.get("average_daily_views", 0),
                wiki_views.get("trend", "no_data"),
            )
        else:
            logger.info("[wikipedia] No article found for query=%r", query)

        # ── Google Trends ────────────────────────────────────────────────────
        trends = self._run_tool(
            "trends_search",
            search_trends,
            query=query,
            geo="US",
        )
        trends = trends or {"query_found": False, "all_trends": []}
        logger.info(
            "[trends] query_found=%s | matched=%s",
            trends.get("query_found"),
            trends.get("matched_trends", []),
        )

        return {
            "wikipedia": {**wiki_summary, "page_views": wiki_views},
            "trends":    trends,
        }

    # ── Full pipeline fetch ───────────────────────────────────────────────────

    def fetch_all(self, query: str) -> dict:
        """
        Runs all five stage fetchers in sequence and returns a unified result.

        This is the primary entry point for the analyst agent: call fetch_all()
        once with a technology query to populate every database layer and receive
        all raw signals in a single dict ready for scoring.

        Execution order is intentionally sequential (not parallel) because:
          1. The scraper tools have polite delays between requests.
          2. MongoDB write ordering matters for provenance (earlier stages
             should appear with earlier inserted_at timestamps).
          3. Debugging sequential failures is much simpler than debugging
             concurrent ones.

        Args:
            query : Technology or concept to fetch signals for.

        Returns:
            dict:
                query      (str)  : The original query.
                fetched_at (str)  : ISO 8601 UTC timestamp of this fetch run.
                academic   (list) : Papers from arXiv + Semantic Scholar.
                startup    (dict) : GitHub repos, ProductHunt, YC companies.
                investment (dict) : NewsAPI articles, TechCrunch articles.
                bigtech    (list) : PatentsView patents.
                mainstream (dict) : Wikipedia summary/views, Google Trends.
                summary    (dict) : Record counts per stage for quick inspection.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        query = query.strip()
        fetched_at = datetime.now(timezone.utc).isoformat()
        logger.info("=" * 60)
        logger.info("fetch_all | query=%r | started at %s", query, fetched_at)
        logger.info("=" * 60)

        academic   = self.fetch_academic(query)
        startup    = self.fetch_startup_signals(query)
        investment = self.fetch_investment_signals(query)
        bigtech    = self.fetch_bigtech_signals(query)
        mainstream = self.fetch_mainstream_signals(query)

        # ── Summary counts ────────────────────────────────────────────────────
        # Gives the analyst agent a quick inventory before it starts scoring.
        # Zero counts in a stage are meaningful — they indicate the technology
        # has not yet generated signal at that level.
        summary = {
            "academic_papers":       len(academic),
            "github_repos":          len(startup.get("github_repos", [])),
            "producthunt_posts":     len(startup.get("producthunt", [])),
            "yc_companies":          len(startup.get("yc_companies", [])),
            "news_articles":         len(investment.get("news_articles", [])),
            "techcrunch_articles":   len(investment.get("techcrunch_articles", [])),
            "patents":               len(bigtech),
            "wikipedia_exists":      mainstream.get("wikipedia", {}).get("exists", False),
            "trending_on_google":    mainstream.get("trends", {}).get("query_found", False),
        }

        logger.info("fetch_all complete | query=%r | summary=%s", query, summary)

        return {
            "query":      query,
            "fetched_at": fetched_at,
            "academic":   academic,
            "startup":    startup,
            "investment": investment,
            "bigtech":    bigtech,
            "mainstream": mainstream,
            "summary":    summary,
        }


# ---------------------------------------------------------------------------
# Module-level convenience instance
# ---------------------------------------------------------------------------

# Import pattern:
#     from pipeline.fetcher import fetcher
#     results = fetcher.fetch_all("solid-state batteries")
fetcher = DataFetcher()


# ---------------------------------------------------------------------------
# Quick smoke-test — run directly to verify the full pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    QUERY = "large language models"

    print(f"\nRunning fetch_all for: '{QUERY}'\n")
    f = DataFetcher()
    results = f.fetch_all(QUERY)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for key, val in results["summary"].items():
        print(f"  {key:<30s} {val}")

    print(f"\nfetched_at : {results['fetched_at']}")
    print(f"academic   : {len(results['academic'])} papers")
    print(f"github     : {len(results['startup'].get('github_repos', []))} repos")
    print(f"producthunt: {len(results['startup'].get('producthunt', []))} posts")
    print(f"yc         : {len(results['startup'].get('yc_companies', []))} companies")
    print(f"news       : {len(results['investment'].get('news_articles', []))} articles")
    print(f"techcrunch : {len(results['investment'].get('techcrunch_articles', []))} articles")
    print(f"patents    : {len(results['bigtech'])} patents")
    print(f"wikipedia  : exists={results['mainstream']['wikipedia'].get('exists')}")
    print(f"trends     : found={results['mainstream']['trends'].get('query_found')}")
