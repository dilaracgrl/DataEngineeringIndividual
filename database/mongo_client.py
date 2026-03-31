"""
MongoDB client for the Technology Pipeline Tracker.

Centralises all MongoDB access behind a single MongoDBClient class.
Each collection maps to a distinct pipeline stage signal — inserting a
document into the correct collection is itself a classification act: it
says "this piece of data belongs to stage N of the technology lifecycle."

Collection → Pipeline stage mapping
─────────────────────────────────────────────────────────────────────
Collection              Source tools                    Stage
─────────────────────────────────────────────────────────────────────
papers                  arxiv_tool, semantic_scholar    1 — Academic
github_repos            github_tool                     2 — Developer
yc_companies            yc_scraper                      2 — Startup
startups                producthunt_tool                2 — Startup
news                    news_tool                       3 — Investment
techcrunch_articles     techcrunch_scraper              3 — Investment
patents                 patents_tool                    4 — Big Tech
─────────────────────────────────────────────────────────────────────
(Stage 5 — Mainstream — is measured live via wikipedia_tool and
trends_tool and is not stored, since it reflects real-time public
interest rather than discrete historical events.)

Schema decisions
────────────────
Every document stored here MUST have these provenance fields at the
top level (enforced by insert_documents before writing):
    source      (str)  : Tool that produced the document, e.g. "arxiv"
    fetched_at  (str)  : ISO 8601 UTC timestamp of the fetch
    inserted_at (str)  : ISO 8601 UTC timestamp of the DB insert (added here)
    lineage     (dict) : W3C PROV record from the tool

Deduplication strategy per collection:
    papers              → upsert on arxiv_id (or paper_id for S2)
    github_repos        → upsert on url (canonical HTML URL of the repo)
    yc_companies        → upsert on yc_url
    startups            → upsert on url (Product Hunt post URL)
    news                → upsert on url (article URL)
    techcrunch_articles → upsert on url
    patents             → upsert on patent_id

This means the same document fetched twice on different days updates
rather than duplicates — keeping the collection lean and the
inserted_at / fetched_at fields reflecting the latest known state.

Required .env key:
    MONGO_URI  — MongoDB connection string.
                 Local:  mongodb://localhost:27017
                 Atlas:  mongodb+srv://user:pass@cluster.mongodb.net/

Usage:
    from database.mongo_client import MongoDBClient
    db = MongoDBClient()
    db.insert_documents("papers", papers_list)
    results = db.find_by_query("papers", "transformer attention")
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from pymongo.collection import Collection
from pymongo.errors import (
    BulkWriteError,
    ConnectionFailure,
    OperationFailure,
    ServerSelectionTimeoutError,
)

# ---------------------------------------------------------------------------
# Environment & logging setup
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mongo_client")

# Database name — all collections live inside this DB
_DB_NAME = "tech_pipeline_tracker"

# Valid collection names — guards against typos in caller code
VALID_COLLECTIONS = {
    "papers",               # Stage 1: arXiv + Semantic Scholar papers
    "github_repos",         # Stage 2: GitHub repository activity
    "yc_companies",         # Stage 2: Y Combinator portfolio companies
    "startups",             # Stage 2: Product Hunt launches
    "news",                 # Stage 3: NewsAPI investment coverage
    "techcrunch_articles",  # Stage 3: TechCrunch funding articles
    "patents",              # Stage 4: PatentsView US patent filings
}

# Deduplication key per collection.
# This is the field used as the unique identifier for upsert logic.
# Choosing a natural key (URL, patent ID, arXiv ID) rather than a synthetic
# key means documents fetched from the same source on different days are
# recognised as the same entity and updated in place, not duplicated.
_DEDUP_KEYS: dict[str, str] = {
    "papers":               "arxiv_id",     # arXiv: arxiv_id; S2: paper_id
    "github_repos":         "url",          # github HTML URL is globally unique
    "yc_companies":         "yc_url",       # YC profile URL is canonical
    "startups":             "url",          # Product Hunt post URL
    "news":                 "url",          # Article URL
    "techcrunch_articles":  "url",          # Article URL
    "patents":              "patent_id",    # USPTO patent number
}

# Text index fields per collection — used by find_by_query().
# MongoDB text indexes are defined per-collection; these are the fields
# that make most sense to search for a given stage signal.
_TEXT_INDEX_FIELDS: dict[str, list[str]] = {
    "papers":               ["title", "abstract"],
    "github_repos":         ["name", "description"],
    "yc_companies":         ["name", "description"],
    "startups":             ["name", "tagline", "description"],
    "news":                 ["title", "description"],
    "techcrunch_articles":  ["title", "summary"],
    "patents":              ["title", "abstract"],
}


# ---------------------------------------------------------------------------
# MongoDBClient
# ---------------------------------------------------------------------------

class MongoDBClient:
    """
    Centralised MongoDB client for the Technology Pipeline Tracker.

    All reads and writes go through this class. Each public method
    validates the collection name, applies provenance stamps, and
    handles errors with specific exception types — consistent with
    the tool pattern used throughout the rest of the project.

    Lazy connection: the pymongo client is initialised on first use
    rather than at import time, so importing this module does not
    immediately require MongoDB to be running.
    """

    def __init__(self, uri: str | None = None, db_name: str = _DB_NAME) -> None:
        """
        Initialises the MongoDBClient.

        Args:
            uri     : MongoDB connection string. Falls back to MONGO_URI in .env,
                      then to localhost default.
            db_name : Database name to use. Defaults to "tech_pipeline_tracker".
        """
        self._uri = (
            uri
            or os.getenv("MONGO_URI")
            or "mongodb://localhost:27017"
        )
        self._db_name = db_name
        self._client: MongoClient | None = None
        logger.debug("MongoDBClient initialised (lazy — not yet connected)")

    # ── Connection management ────────────────────────────────────────────────

    def _get_db(self):
        """
        Returns the pymongo Database object, creating the MongoClient on
        first call (lazy connection pattern).

        pymongo's MongoClient maintains an internal connection pool;
        creating one client per process and reusing it is the recommended
        pattern. We do NOT create a new client on every method call.

        Raises:
            RuntimeError: If MONGO_URI is missing or connection fails.
        """
        if self._client is None:
            if not self._uri:
                raise RuntimeError(
                    "MongoDB URI not set. Add MONGO_URI to .env or pass uri= to MongoDBClient()."
                )
            try:
                # serverSelectionTimeoutMS=5000: fail fast if MongoDB is not
                # running, rather than hanging for 30 seconds
                self._client = MongoClient(
                    self._uri,
                    serverSelectionTimeoutMS=5_000,
                )
                # Force a connection check — ismaster is a lightweight ping
                self._client.admin.command("ismaster")
                logger.info("MongoDB connected to %s / %s", self._uri, self._db_name)
            except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                self._client = None
                raise RuntimeError(
                    f"Could not connect to MongoDB at '{self._uri}': {e}\n"
                    "Check that MongoDB is running and MONGO_URI is correct in .env."
                ) from e

        return self._client[self._db_name]

    def _get_collection(self, collection: str) -> Collection:
        """
        Validates the collection name and returns the pymongo Collection object.

        Raises:
            ValueError: If collection is not in VALID_COLLECTIONS.
        """
        if collection not in VALID_COLLECTIONS:
            raise ValueError(
                f"Unknown collection '{collection}'. "
                f"Valid collections: {sorted(VALID_COLLECTIONS)}"
            )
        return self._get_db()[collection]

    def get_analysis_cache(self):
        """
        Returns the analysis_cache collection for 24-hour result caching.

        This collection sits outside the pipeline data collections (VALID_COLLECTIONS)
        because it stores API-level service results, not pipeline documents.
        Schema per document:
            query      (str)  : Normalised query string — used as the upsert key.
            cached_at  (str)  : ISO 8601 UTC timestamp of when the result was cached.
            scores     (dict) : Pipeline scores dict (for the scores SSE event).
            assessment (dict) : Merged analyst + critique assessment.
            velocity   (dict) : Velocity analysis dict.
            raw_critique (str): Raw text of the self-critique Claude call.
            logged_at  (str)  : Timestamp from the original analyse() run.
        """
        return self._get_db()["analysis_cache"]

    def close(self) -> None:
        """Closes the MongoDB connection and releases the connection pool."""
        if self._client is not None:
            self._client.close()
            self._client = None
            logger.debug("MongoDB connection closed")

    # ── Text index management ────────────────────────────────────────────────

    def _ensure_text_index(self, collection: str) -> None:
        """
        Ensures a MongoDB text index exists on the appropriate fields for
        the given collection. Called lazily before find_by_query().

        MongoDB allows only one text index per collection. We build a
        compound text index across all the searchable fields defined in
        _TEXT_INDEX_FIELDS. If the index already exists, this is a no-op.

        Why text indexes here and not in a migration script?
        Because this is a research/analysis tool and we want it to work
        out of the box without a separate setup step. The createIndex
        call is idempotent — safe to call on every query.
        """
        fields = _TEXT_INDEX_FIELDS.get(collection, ["title"])
        coll = self._get_collection(collection)
        # pymongo's create_index with TEXT is idempotent if the index spec matches
        try:
            coll.create_index(
                [(f, "text") for f in fields],
                name=f"{collection}_text_idx",
                background=True,
            )
        except OperationFailure as e:
            # A conflicting index spec already exists — log and continue.
            # This can happen if the collection was created with a different
            # field set; the existing index still works for partial matches.
            logger.warning("Text index creation skipped for '%s': %s", collection, e)

    # ── Write operations ─────────────────────────────────────────────────────

    def insert_documents(
        self,
        collection: str,
        docs: list[dict],
    ) -> dict:
        """
        Bulk-inserts a list of documents into `collection`, adding an
        `inserted_at` timestamp and deduplicating via upsert on the
        collection's natural key.

        Upsert strategy:
            For each document, we issue an UpdateOne with upsert=True,
            filtering on the deduplication key (e.g. url, patent_id).
            - If no document with that key exists → insert (new data)
            - If a document already exists → update with $set (refreshed data)

            This means re-fetching the same arXiv paper two weeks later
            updates its citation_count field rather than creating a duplicate.
            The inserted_at field on the *first* insert is preserved because
            we use $setOnInsert for it, not $set.

        Provenance enforcement:
            Every document MUST have source, fetched_at, and lineage fields.
            Documents missing these are skipped with a warning — this enforces
            the W3C PROV contract defined in the tool layer.

        Args:
            collection : Target collection name (must be in VALID_COLLECTIONS).
            docs       : List of document dicts from a tool function.

        Returns:
            dict with: inserted (new docs), updated (existing docs refreshed),
                       skipped (failed provenance check), total_attempted.

        Raises:
            ValueError  : On invalid collection name.
            RuntimeError: On MongoDB connection or write errors.
        """
        if not docs:
            logger.debug("insert_documents called with empty list for '%s'", collection)
            return {"inserted": 0, "updated": 0, "skipped": 0, "total_attempted": 0}

        coll = self._get_collection(collection)
        dedup_key = _DEDUP_KEYS[collection]
        now = datetime.now(timezone.utc).isoformat()

        operations = []
        skipped = 0

        for doc in docs:
            # Provenance check — skip documents that lack required fields.
            # These would create unproveable records in the pipeline store.
            if not doc.get("source") or not doc.get("fetched_at"):
                logger.warning(
                    "Skipping document in '%s' — missing 'source' or 'fetched_at'. "
                    "Ensure the tool's _build_prov_record() was called.",
                    collection,
                )
                skipped += 1
                continue

            # Resolve the deduplication key value.
            # Semantic Scholar uses "paper_id" while arXiv uses "arxiv_id";
            # both go into the "papers" collection. Fall back to paper_id if
            # arxiv_id is absent.
            key_value = doc.get(dedup_key)
            if not key_value and collection == "papers":
                key_value = doc.get("paper_id")   # S2 fallback
            if not key_value:
                logger.warning(
                    "Skipping document in '%s' — missing dedup key '%s'.",
                    collection, dedup_key,
                )
                skipped += 1
                continue

            # Stamp the document with the DB insertion time.
            # inserted_at is set only on first insert ($setOnInsert) so it
            # always reflects when we *first* saw this entity.
            # fetched_at (from the tool) always reflects the latest fetch.
            doc_to_write = {k: v for k, v in doc.items() if k != "lineage"}
            doc_to_write["lineage"] = doc.get("lineage", {})

            operations.append(
                UpdateOne(
                    filter={dedup_key: key_value},
                    update={
                        "$set": doc_to_write,
                        "$setOnInsert": {"inserted_at": now},
                    },
                    upsert=True,
                )
            )

        if not operations:
            return {
                "inserted": 0,
                "updated": 0,
                "skipped": skipped,
                "total_attempted": len(docs),
            }

        try:
            result = coll.bulk_write(operations, ordered=False)
        except BulkWriteError as e:
            # BulkWriteError contains partial results — log details and
            # surface the error without losing the partial write count
            logger.error("Bulk write partial failure for '%s': %s", collection, e.details)
            raise RuntimeError(
                f"MongoDB bulk write had errors for collection '{collection}': "
                f"{e.details.get('writeErrors', [])}"
            ) from e

        summary = {
            "inserted":        result.upserted_count,
            "updated":         result.modified_count,
            "skipped":         skipped,
            "total_attempted": len(docs),
        }

        logger.info(
            "insert_documents '%s' | inserted=%d | updated=%d | skipped=%d",
            collection,
            summary["inserted"],
            summary["updated"],
            summary["skipped"],
        )
        return summary

    # ── Read operations ──────────────────────────────────────────────────────

    def find_by_query(
        self,
        collection: str,
        query: str,
        limit: int = 20,
    ) -> list[dict]:
        """
        Full-text search across title and description fields in `collection`.

        Uses MongoDB's $text operator against a compound text index
        (created lazily if it doesn't exist). Results are sorted by
        MongoDB's text relevance score (textScore) — highest-match first.

        This is the primary method for answering "what do we have stored
        about X?" across any pipeline stage collection.

        Args:
            collection : Collection to search (must be in VALID_COLLECTIONS).
            query      : Search string — supports multi-word phrases.
            limit      : Max documents to return.

        Returns:
            List of document dicts with an added "_score" field (relevance).
            The "_id" field is excluded from results (internal MongoDB key).

        Raises:
            ValueError  : On invalid collection name.
            RuntimeError: On MongoDB errors.
        """
        limit = max(1, min(limit, 100))
        self._ensure_text_index(collection)
        coll = self._get_collection(collection)

        logger.info(
            "find_by_query '%s' | query=%r | limit=%d", collection, query, limit
        )

        try:
            cursor = coll.find(
                filter={"$text": {"$search": query}},
                projection={
                    "_id": 0,
                    "_score": {"$meta": "textScore"},
                },
                sort=[("_score", {"$meta": "textScore"})],
                limit=limit,
            )
            return list(cursor)
        except OperationFailure as e:
            logger.error("find_by_query failed for '%s': %s", collection, e)
            raise RuntimeError(
                f"MongoDB text search failed for collection '{collection}': {e}"
            ) from e

    def get_by_source(
        self,
        collection: str,
        source: str,
        limit: int = 20,
    ) -> list[dict]:
        """
        Retrieves documents from `collection` filtered by their `source` field.

        The `source` field is set by every tool (e.g. "arxiv", "semantic_scholar",
        "producthunt", "patentsview"). This method lets the pipeline query a
        specific data origin within a stage — for example, getting only arXiv
        papers from the "papers" collection while excluding Semantic Scholar ones.

        Useful for comparing signal quality across tools at the same stage,
        or for re-processing data from one specific source.

        Args:
            collection : Target collection.
            source     : Source identifier string (e.g. "arxiv", "github",
                         "patentsview", "ycombinator", "techcrunch").
            limit      : Max documents to return.

        Returns:
            List of document dicts sorted by fetched_at descending (newest first).

        Raises:
            ValueError  : On invalid collection name.
            RuntimeError: On MongoDB errors.
        """
        limit = max(1, min(limit, 100))
        coll = self._get_collection(collection)

        logger.info(
            "get_by_source '%s' | source=%r | limit=%d", collection, source, limit
        )

        try:
            cursor = coll.find(
                filter={"source": source},
                projection={"_id": 0},
                sort=[("fetched_at", -1)],
                limit=limit,
            )
            return list(cursor)
        except OperationFailure as e:
            logger.error("get_by_source failed for '%s': %s", collection, e)
            raise RuntimeError(
                f"MongoDB query failed for collection '{collection}': {e}"
            ) from e

    def get_recent(
        self,
        collection: str,
        hours: int = 24,
        limit: int = 20,
    ) -> list[dict]:
        """
        Retrieves documents inserted into `collection` within the last `hours` hours.

        Queries on `inserted_at` (the DB write timestamp, not fetched_at).
        This answers "what new data has the pipeline collected recently?" —
        useful for monitoring pipeline health and triggering downstream
        re-analysis when new signals arrive.

        Note: `inserted_at` is only set for documents written by insert_documents()
        in this client. Documents inserted directly (e.g. via mongo shell) may
        lack this field and will not appear in results.

        Args:
            collection : Target collection.
            hours      : Look-back window in hours.
            limit      : Max documents to return.

        Returns:
            List of document dicts sorted by inserted_at descending (newest first).

        Raises:
            ValueError  : On invalid collection or hours value.
            RuntimeError: On MongoDB errors.
        """
        if hours < 1:
            raise ValueError(f"hours must be >= 1, got {hours}")
        limit = max(1, min(limit, 100))
        coll = self._get_collection(collection)

        cutoff = (
            datetime.now(timezone.utc) - timedelta(hours=hours)
        ).isoformat()

        logger.info(
            "get_recent '%s' | hours=%d | cutoff=%s | limit=%d",
            collection, hours, cutoff, limit,
        )

        try:
            cursor = coll.find(
                filter={"inserted_at": {"$gte": cutoff}},
                projection={"_id": 0},
                sort=[("inserted_at", -1)],
                limit=limit,
            )
            return list(cursor)
        except OperationFailure as e:
            logger.error("get_recent failed for '%s': %s", collection, e)
            raise RuntimeError(
                f"MongoDB query failed for collection '{collection}': {e}"
            ) from e

    def get_collection_stats(self, collection: str) -> dict:
        """
        Returns summary statistics about a collection: document count, and
        the oldest and newest document dates (by fetched_at).

        Used by the analyst agent to understand how much data has been
        gathered for a given pipeline stage before deciding whether the
        signal is reliable enough to score. A collection with 3 documents
        is not a trustworthy stage signal; one with 200 documents is.

        Args:
            collection : Target collection.

        Returns:
            dict with keys:
                collection, document_count, oldest_fetched_at,
                newest_fetched_at, sources (unique source values present)

        Raises:
            ValueError  : On invalid collection name.
            RuntimeError: On MongoDB errors.
        """
        coll = self._get_collection(collection)

        logger.info("get_collection_stats '%s'", collection)

        try:
            count = coll.count_documents({})

            if count == 0:
                return {
                    "collection":      collection,
                    "document_count":  0,
                    "oldest_fetched_at": None,
                    "newest_fetched_at": None,
                    "sources":         [],
                }

            # Oldest document: sort fetched_at ascending, take first
            oldest_doc = coll.find_one(
                filter={},
                projection={"fetched_at": 1, "_id": 0},
                sort=[("fetched_at", 1)],
            )

            # Newest document: sort fetched_at descending, take first
            newest_doc = coll.find_one(
                filter={},
                projection={"fetched_at": 1, "_id": 0},
                sort=[("fetched_at", -1)],
            )

            # Distinct sources present in this collection
            sources = coll.distinct("source")

            return {
                "collection":        collection,
                "document_count":    count,
                "oldest_fetched_at": oldest_doc.get("fetched_at") if oldest_doc else None,
                "newest_fetched_at": newest_doc.get("fetched_at") if newest_doc else None,
                "sources":           sorted(sources),
            }

        except OperationFailure as e:
            logger.error("get_collection_stats failed for '%s': %s", collection, e)
            raise RuntimeError(
                f"MongoDB stats query failed for collection '{collection}': {e}"
            ) from e

    # ── Context manager support ──────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------------
# Module-level convenience instance
# ---------------------------------------------------------------------------

# A single shared instance for use across the pipeline.
# Import pattern:
#     from database.mongo_client import mongo_db
#     mongo_db.insert_documents("papers", results)
#
# If you need a custom URI or DB name, instantiate MongoDBClient directly.
mongo_db = MongoDBClient()


# ---------------------------------------------------------------------------
# Quick smoke-test — run directly to verify connection and all methods
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    # Sample documents that mirror what arxiv_tool.py returns
    SAMPLE_PAPERS = [
        {
            "arxiv_id":                   "2301.00001",
            "title":                      "Attention Is All You Need (test)",
            "abstract":                   "We propose a transformer architecture...",
            "authors":                    ["A. Vaswani", "N. Shazeer"],
            "year":                       2017,
            "citation_count":             90000,
            "influential_citation_count": 5000,
            "url":                        "https://arxiv.org/abs/2301.00001",
            "source":                     "arxiv",
            "fetched_at":                 datetime.now(timezone.utc).isoformat(),
            "lineage": {
                "prov:type": "prov:Entity",
                "prov:wasGeneratedBy": {
                    "prov:label": "arxiv_search",
                    "prov:startedAtTime": datetime.now(timezone.utc).isoformat(),
                },
            },
        },
        {
            "arxiv_id":                   "2301.00002",
            "title":                      "BERT: Pre-training of Deep Bidirectional Transformers (test)",
            "abstract":                   "We introduce a new language representation model BERT...",
            "authors":                    ["J. Devlin", "M. Chang"],
            "year":                       2019,
            "citation_count":             60000,
            "influential_citation_count": 3200,
            "url":                        "https://arxiv.org/abs/2301.00002",
            "source":                     "arxiv",
            "fetched_at":                 datetime.now(timezone.utc).isoformat(),
            "lineage": {
                "prov:type": "prov:Entity",
                "prov:wasGeneratedBy": {
                    "prov:label": "arxiv_search",
                    "prov:startedAtTime": datetime.now(timezone.utc).isoformat(),
                },
            },
        },
    ]

    with MongoDBClient() as db:
        print("\n=== insert_documents: 'papers' ===")
        result = db.insert_documents("papers", SAMPLE_PAPERS)
        print(json.dumps(result, indent=2))

        print("\n=== find_by_query: 'papers', 'transformer' ===")
        docs = db.find_by_query("papers", "transformer", limit=5)
        for d in docs:
            print(f"  [{d.get('_score', 0):.2f}] {d.get('title', '')[:60]}")

        print("\n=== get_by_source: 'papers', 'arxiv' ===")
        docs = db.get_by_source("papers", "arxiv", limit=5)
        for d in docs:
            print(f"  {d.get('arxiv_id')} — {d.get('title', '')[:60]}")

        print("\n=== get_recent: 'papers', 24h ===")
        docs = db.get_recent("papers", hours=24, limit=5)
        print(f"  {len(docs)} documents inserted in last 24h")

        print("\n=== get_collection_stats: 'papers' ===")
        stats = db.get_collection_stats("papers")
        print(json.dumps(stats, indent=2))
