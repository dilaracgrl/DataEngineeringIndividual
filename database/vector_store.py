"""
Vector store for semantic RAG search across academic papers and articles.

What is RAG and why does it matter here?
─────────────────────────────────────────
RAG (Retrieval-Augmented Generation) is the pattern of giving an LLM agent
relevant context from a database *before* it writes its answer, rather than
relying on the model's training data alone. In this pipeline:

  1. A user asks "where is solid-state batteries in the pipeline?"
  2. The agent calls search_papers("solid-state batteries") here
  3. This returns the 5 most semantically similar papers we have stored
  4. The agent reads those abstracts and cites them in its answer

Without RAG the agent either hallucinates citations or gives a generic answer.
With RAG it gives a grounded, source-cited answer from our actual fetched data.

Why semantic search instead of keyword matching (MongoDB $text)?
────────────────────────────────────────────────────────────────
MongoDB's text index matches exact words. A query for "neural net" won't match
a paper titled "Deep Learning via Gradient Descent" even though they're about
the same thing. Semantic search converts both the query and every stored document
into dense vector embeddings — points in a high-dimensional space where similar
*meanings* are close together, regardless of the exact words used.

Concrete example relevant to this pipeline:
  Query: "making AI think step by step"
  Keyword match: 0 results (no exact token overlap)
  Semantic match: returns papers on "chain-of-thought prompting",
                  "scratchpad reasoning", "tree-of-thought decoding"
  → The analyst agent gets the right context even with an informal query.

Model choice: all-MiniLM-L6-v2
────────────────────────────────
A 22M-parameter sentence transformer that runs locally (no API key, no cost).
It produces 384-dimensional embeddings and is one of the best
speed/accuracy tradeoffs available. Encode time is ~5ms per sentence on CPU.
The model is downloaded once by sentence-transformers and cached in ~/.cache.

ChromaDB
────────
ChromaDB is an open-source vector database that persists to disk (./chroma_db).
It handles embedding storage, indexing (HNSW), and approximate nearest-neighbour
search. We supply the raw text and it calls our embedding function — no manual
numpy required.

Collections
───────────
  papers   — arXiv and Semantic Scholar documents (Stage 1 signal)
  articles — NewsAPI and TechCrunch documents (Stage 3 signal)

Usage:
    from database.vector_store import VectorStore
    vs = VectorStore()
    vs.add_papers(arxiv_results)
    similar = vs.search_papers("attention mechanism transformers", n_results=5)
"""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Avoid PostHog telemetry bugs/noise (Chroma 0.5.x + posthog incompatibility).
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment & logging setup
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vector_store")

# Persist ChromaDB to a directory next to this module by default.
# Override with CHROMA_DB_PATH in .env.
_DEFAULT_CHROMA_PATH = Path(__file__).parent / "chroma_db"

# Embedding model — runs fully locally, no API key required.
# Downloaded once to ~/.cache/torch/sentence_transformers on first use.
_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ChromaDB collection names
_COLLECTION_PAPERS   = "papers"
_COLLECTION_ARTICLES = "articles"

# Metadata fields we persist per document type.
# ChromaDB metadata values must be str, int, or float — no lists or dicts.
# Fields that are lists in the tool output (authors, tags) are joined to
# a single comma-separated string.
_PAPER_METADATA_KEYS   = {"source", "year", "citation_count", "url", "arxiv_id", "paper_id"}
_ARTICLE_METADATA_KEYS = {"source", "date", "url"}


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------

class VectorStore:
    """
    Semantic vector store for papers and articles using ChromaDB and
    sentence-transformers.

    Lazy initialisation: the ChromaDB client and sentence-transformer model
    are loaded on first use, not at import time. This keeps import cost low
    and avoids downloading the model if the vector store is never called.

    Deduplication: ChromaDB uses the document ID to deduplicate. We derive
    IDs from the document's URL or arxiv_id, so re-adding the same paper
    is a no-op (ChromaDB upserts by ID).
    """

    def __init__(self, chroma_path: str | Path | None = None) -> None:
        """
        Args:
            chroma_path : Directory for ChromaDB persistence.
                          Falls back to CHROMA_DB_PATH in .env, then to
                          database/chroma_db/ next to this module.
        """
        self._chroma_path = Path(
            chroma_path
            or os.getenv("CHROMA_DB_PATH")
            or _DEFAULT_CHROMA_PATH
        )
        self._client: chromadb.PersistentClient | None = None
        self._embedding_fn = None
        self._papers_col   = None
        self._articles_col = None
        logger.debug("VectorStore targeting %s", self._chroma_path)

    # ── Lazy initialisation ──────────────────────────────────────────────────

    def _get_embedding_fn(self):
        """
        Returns the SentenceTransformer embedding function, loading it on
        first call.

        ChromaDB's SentenceTransformerEmbeddingFunction wraps the
        sentence-transformers library and handles batching automatically.
        The model weights (~90 MB) are downloaded once and cached locally.
        """
        if self._embedding_fn is None:
            logger.info("Loading embedding model '%s' (first use — may download)", _EMBEDDING_MODEL)
            self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=_EMBEDDING_MODEL
            )
            logger.info("Embedding model ready")
        return self._embedding_fn

    def _get_client(self) -> chromadb.PersistentClient:
        """
        Returns the ChromaDB persistent client, creating it on first call.

        PersistentClient saves all data to disk at self._chroma_path.
        The directory is created automatically if it does not exist.
        """
        if self._client is None:
            self._chroma_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self._chroma_path))
            logger.info("ChromaDB client connected at %s", self._chroma_path)
        return self._client

    def _get_collection(self, name: str):
        """
        Returns a ChromaDB collection by name, creating it if needed.

        get_or_create_collection is idempotent — safe to call on every access.
        We attach our embedding function so ChromaDB calls it automatically
        when documents or queries are added.
        """
        return self._get_client().get_or_create_collection(
            name=name,
            embedding_function=self._get_embedding_fn(),
            # cosine distance is standard for sentence-transformer embeddings.
            # It measures the angle between vectors, not their magnitude —
            # appropriate here since we care about semantic direction, not length.
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def _papers(self):
        if self._papers_col is None:
            self._papers_col = self._get_collection(_COLLECTION_PAPERS)
        return self._papers_col

    @property
    def _articles(self):
        if self._articles_col is None:
            self._articles_col = self._get_collection(_COLLECTION_ARTICLES)
        return self._articles_col

    # ── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _sanitise_metadata(raw: dict, allowed_keys: set[str]) -> dict:
        """
        Filters and type-coerces a document dict into a ChromaDB-safe metadata
        dict.

        ChromaDB metadata values must be str, int, or float.
        Rules applied:
          - Only keys in allowed_keys are kept
          - None values → empty string (ChromaDB rejects None)
          - Lists → comma-separated string (e.g. authors list)
          - Dicts → JSON string (for nested objects like lineage)
          - Booleans → int (True→1, False→0)
          - All other non-str/int/float types → str()
        """
        import json as _json
        result = {}
        for key in allowed_keys:
            val = raw.get(key)
            if val is None:
                result[key] = ""
            elif isinstance(val, (str, float)):
                result[key] = val
            elif isinstance(val, int):
                result[key] = val
            elif isinstance(val, bool):
                result[key] = int(val)
            elif isinstance(val, list):
                result[key] = ", ".join(str(v) for v in val)
            elif isinstance(val, dict):
                result[key] = _json.dumps(val)
            else:
                result[key] = str(val)
        return result

    @staticmethod
    def _derive_paper_id(doc: dict) -> str | None:
        """
        Derives a stable, unique ID for a paper document.

        Priority: arxiv_id → paper_id (Semantic Scholar) → url.
        Returns None if no stable ID can be found — the document is skipped.
        """
        return (
            doc.get("arxiv_id")
            or doc.get("paper_id")
            or doc.get("url")
            or None
        )

    @staticmethod
    def _derive_article_id(doc: dict) -> str | None:
        """Derives a stable ID for an article — URL is the natural key."""
        return doc.get("url") or None

    @staticmethod
    def _make_paper_text(doc: dict) -> str:
        """
        Concatenates title and abstract into the text to embed.

        Embedding both fields together gives richer semantic coverage:
        the title provides the concept label; the abstract provides the
        methodological and contextual vocabulary. This means a query like
        "attention mechanism" can match a paper titled "Transformer" whose
        abstract explains how attention works.
        """
        title    = doc.get("title", "").strip()
        abstract = doc.get("abstract", "").strip()
        if title and abstract:
            return f"{title}. {abstract}"
        return title or abstract or ""

    @staticmethod
    def _make_article_text(doc: dict) -> str:
        """
        Concatenates title and summary/description for article embedding.

        News articles and TechCrunch posts store their excerpt in different
        fields — we try both.
        """
        title   = doc.get("title", "").strip()
        summary = (doc.get("summary") or doc.get("description", "")).strip()
        if title and summary:
            return f"{title}. {summary}"
        return title or summary or ""

    # ── Write operations ─────────────────────────────────────────────────────

    def add_papers(self, papers: list[dict]) -> dict:
        """
        Adds a list of paper dicts to the 'papers' ChromaDB collection.

        Takes output directly from arxiv_tool.search_papers() or
        semantic_scholar_tool.search_papers(). Embeddings are generated
        from title + abstract using all-MiniLM-L6-v2 and stored alongside
        the metadata fields most useful for the analyst agent's citations.

        Deduplication: ChromaDB upserts on ID. Re-adding the same arxiv_id
        updates the document in place — safe to call after every pipeline run.

        Args:
            papers : List of paper dicts from arxiv_tool or semantic_scholar_tool.

        Returns:
            dict: {"added": int, "skipped": int, "total": int}

        Raises:
            RuntimeError: On ChromaDB write errors.
        """
        if not papers:
            return {"added": 0, "skipped": 0, "total": 0}

        ids, documents, metadatas = [], [], []
        skipped = 0

        for doc in papers:
            doc_id = self._derive_paper_id(doc)
            if not doc_id:
                logger.warning("Skipping paper with no stable ID: title=%r", doc.get("title"))
                skipped += 1
                continue

            text = self._make_paper_text(doc)
            if not text.strip():
                logger.warning("Skipping paper with empty text: id=%r", doc_id)
                skipped += 1
                continue

            metadata = self._sanitise_metadata(doc, _PAPER_METADATA_KEYS)
            # Ensure citation_count is int for numeric filtering later
            if "citation_count" in metadata and isinstance(metadata["citation_count"], str):
                try:
                    metadata["citation_count"] = int(metadata["citation_count"])
                except ValueError:
                    metadata["citation_count"] = 0

            ids.append(str(doc_id))
            documents.append(text)
            metadatas.append(metadata)

        if not ids:
            return {"added": 0, "skipped": skipped, "total": len(papers)}

        try:
            # upsert: adds new documents and updates existing ones by ID.
            # ChromaDB calls our embedding_fn on `documents` automatically.
            self._papers.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )
        except Exception as e:
            logger.error("ChromaDB upsert failed for papers: %s", e)
            raise RuntimeError(f"ChromaDB write error in add_papers: {e}") from e

        result = {"added": len(ids), "skipped": skipped, "total": len(papers)}
        logger.info(
            "add_papers | added=%d | skipped=%d | total=%d",
            result["added"], result["skipped"], result["total"],
        )
        return result

    def add_articles(self, articles: list[dict]) -> dict:
        """
        Adds a list of article dicts to the 'articles' ChromaDB collection.

        Takes output from news_tool.search_funding_news(),
        news_tool.get_news_volume(), or techcrunch_scraper.search_funding_articles().
        Embeddings are generated from title + summary.

        Args:
            articles : List of article dicts from news or TechCrunch tools.

        Returns:
            dict: {"added": int, "skipped": int, "total": int}

        Raises:
            RuntimeError: On ChromaDB write errors.
        """
        if not articles:
            return {"added": 0, "skipped": 0, "total": 0}

        ids, documents, metadatas = [], [], []
        skipped = 0

        for doc in articles:
            doc_id = self._derive_article_id(doc)
            if not doc_id:
                logger.warning("Skipping article with no URL: title=%r", doc.get("title"))
                skipped += 1
                continue

            text = self._make_article_text(doc)
            if not text.strip():
                logger.warning("Skipping article with empty text: url=%r", doc_id)
                skipped += 1
                continue

            metadata = self._sanitise_metadata(doc, _ARTICLE_METADATA_KEYS)

            ids.append(str(doc_id))
            documents.append(text)
            metadatas.append(metadata)

        if not ids:
            return {"added": 0, "skipped": skipped, "total": len(articles)}

        try:
            self._articles.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )
        except Exception as e:
            logger.error("ChromaDB upsert failed for articles: %s", e)
            raise RuntimeError(f"ChromaDB write error in add_articles: {e}") from e

        result = {"added": len(ids), "skipped": skipped, "total": len(articles)}
        logger.info(
            "add_articles | added=%d | skipped=%d | total=%d",
            result["added"], result["skipped"], result["total"],
        )
        return result

    # ── Search operations ────────────────────────────────────────────────────

    def _format_results(self, chroma_result: dict, collection_label: str) -> list[dict]:
        """
        Converts a raw ChromaDB query result into a clean list of dicts.

        ChromaDB returns parallel lists: ids[0], documents[0], metadatas[0],
        distances[0]. We zip them into one dict per result and compute a
        similarity score from the cosine distance.

        Cosine distance → similarity:
            ChromaDB returns distance in [0, 2] for cosine space.
            0 = identical vectors, 2 = maximally opposite.
            similarity = 1 - (distance / 2) → maps to [0, 1].
            We multiply by 100 for a 0–100 percentage that's easier to read.
        """
        results = []
        if not chroma_result or not chroma_result.get("ids"):
            return results

        ids       = chroma_result["ids"][0]
        documents = chroma_result["documents"][0]
        metadatas = chroma_result["metadatas"][0]
        distances = chroma_result["distances"][0]

        for doc_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            similarity_score = round((1 - distance / 2) * 100, 1)
            results.append({
                "id":               doc_id,
                "text":             document,
                "similarity_score": similarity_score,
                "collection":       collection_label,
                **metadata,
            })

        return results

    def search_papers(self, query: str, n_results: int = 5) -> list[dict]:
        """
        Semantic search across the 'papers' collection.

        Converts the query to an embedding using the same model used at
        ingest time, then finds the n_results nearest neighbours in the
        vector space. Returns results ranked by semantic similarity
        (highest first).

        Why this beats keyword search for papers:
          Query: "predicting protein shape"
          Keyword: misses "AlphaFold", "structure prediction", "ab initio folding"
          Semantic: finds all of them because they're close in embedding space

        Args:
            query     : Natural-language search string.
            n_results : Number of results to return.

        Returns:
            List of result dicts with: id, text, similarity_score (0–100),
            collection, and all metadata fields (source, year, citation_count,
            url, arxiv_id/paper_id).
            Empty list if the collection has no documents.

        Raises:
            RuntimeError: On ChromaDB query errors.
        """
        n_results = max(1, min(n_results, 20))
        logger.info("search_papers | query=%r | n_results=%d", query, n_results)

        # Guard: ChromaDB raises if n_results > collection size
        count = self._papers.count()
        if count == 0:
            logger.info("search_papers: papers collection is empty")
            return []
        n_results = min(n_results, count)

        try:
            raw = self._papers.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error("ChromaDB query failed for papers: %s", e)
            raise RuntimeError(f"ChromaDB query error in search_papers: {e}") from e

        results = self._format_results(raw, _COLLECTION_PAPERS)
        logger.info("search_papers returned %d results", len(results))
        return results

    def search_articles(self, query: str, n_results: int = 5) -> list[dict]:
        """
        Semantic search across the 'articles' collection.

        Same approach as search_papers() but over news and TechCrunch articles.
        Useful for the analyst agent to find existing investment-phase evidence
        before deciding whether to re-fetch from the live APIs.

        Args:
            query     : Natural-language search string.
            n_results : Number of results to return.

        Returns:
            List of result dicts with: id, text, similarity_score (0–100),
            collection, and all metadata fields (source, date, url).

        Raises:
            RuntimeError: On ChromaDB query errors.
        """
        n_results = max(1, min(n_results, 20))
        logger.info("search_articles | query=%r | n_results=%d", query, n_results)

        count = self._articles.count()
        if count == 0:
            logger.info("search_articles: articles collection is empty")
            return []
        n_results = min(n_results, count)

        try:
            raw = self._articles.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error("ChromaDB query failed for articles: %s", e)
            raise RuntimeError(f"ChromaDB query error in search_articles: {e}") from e

        results = self._format_results(raw, _COLLECTION_ARTICLES)
        logger.info("search_articles returned %d results", len(results))
        return results

    def search_all(self, query: str, n_results: int = 5) -> list[dict]:
        """
        Searches both 'papers' and 'articles' collections and returns a
        merged list ranked by similarity score (highest first).

        Used when the analyst agent wants the most relevant stored context
        regardless of stage — e.g. "give me everything we know about
        multimodal AI" pulls both the foundational papers (Stage 1) and
        the funding news (Stage 3) in one ranked list.

        Merge strategy: fetch n_results from each collection independently,
        then sort the combined list by similarity_score descending and
        return the top n_results. This avoids one collection dominating
        purely because it has more documents.

        Args:
            query     : Natural-language search string.
            n_results : Number of final merged results to return.

        Returns:
            List of up to n_results dicts, merged from both collections,
            sorted by similarity_score descending. Each result has a
            "collection" field indicating which collection it came from.

        Raises:
            RuntimeError: On ChromaDB query errors.
        """
        n_results = max(1, min(n_results, 20))
        logger.info("search_all | query=%r | n_results=%d", query, n_results)

        # Fetch from each collection — errors in one don't block the other
        paper_results   = []
        article_results = []

        try:
            paper_results = self.search_papers(query, n_results=n_results)
        except RuntimeError as e:
            logger.warning("search_all: papers search failed (%s) — continuing", e)

        try:
            article_results = self.search_articles(query, n_results=n_results)
        except RuntimeError as e:
            logger.warning("search_all: articles search failed (%s) — continuing", e)

        merged = sorted(
            paper_results + article_results,
            key=lambda r: r.get("similarity_score", 0),
            reverse=True,
        )[:n_results]

        logger.info(
            "search_all | query=%r | papers=%d | articles=%d | merged=%d",
            query, len(paper_results), len(article_results), len(merged),
        )
        return merged

    def get_collection_counts(self) -> dict:
        """
        Returns the number of documents currently stored in each collection.

        Useful for checking whether the vector store has been populated before
        running a search — an empty collection will return no results.

        Returns:
            dict: {"papers": int, "articles": int}
        """
        return {
            "papers":   self._papers.count(),
            "articles": self._articles.count(),
        }


# ---------------------------------------------------------------------------
# Module-level convenience instance
# ---------------------------------------------------------------------------

# Import pattern:
#     from database.vector_store import vector_store
#     vector_store.add_papers(arxiv_results)
#     results = vector_store.search_all("attention mechanism")
vector_store = VectorStore()


# ---------------------------------------------------------------------------
# Quick smoke-test — run directly to verify ChromaDB + model setup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile, shutil

    tmp_dir = tempfile.mkdtemp(prefix="chroma_test_")
    print(f"\nSmoke-test ChromaDB path: {tmp_dir}")

    vs = VectorStore(chroma_path=tmp_dir)

    # ── Sample papers (mirrors arxiv_tool output shape) ──────────────────
    SAMPLE_PAPERS = [
        {
            "arxiv_id":      "1706.03762",
            "title":         "Attention Is All You Need",
            "abstract":      (
                "We propose a new simple network architecture, the Transformer, "
                "based solely on attention mechanisms, dispensing with recurrence "
                "and convolutions entirely."
            ),
            "authors":       ["Vaswani", "Shazeer", "Parmar"],
            "year":          2017,
            "citation_count": 90000,
            "url":           "https://arxiv.org/abs/1706.03762",
            "source":        "arxiv",
        },
        {
            "arxiv_id":      "2005.14165",
            "title":         "Language Models are Few-Shot Learners",
            "abstract":      (
                "We train GPT-3, an autoregressive language model with 175 billion "
                "parameters, and demonstrate few-shot learning across many NLP tasks."
            ),
            "authors":       ["Brown", "Mann"],
            "year":          2020,
            "citation_count": 20000,
            "url":           "https://arxiv.org/abs/2005.14165",
            "source":        "arxiv",
        },
    ]

    # ── Sample articles (mirrors news_tool / techcrunch output shape) ────
    SAMPLE_ARTICLES = [
        {
            "title":       "OpenAI raises $10B in funding round led by Microsoft",
            "summary":     "OpenAI announced a $10 billion investment from Microsoft to accelerate GPT development.",
            "url":         "https://techcrunch.com/openai-raises-10b",
            "date":        "2023-01-23",
            "source":      "techcrunch",
        },
        {
            "title":       "Anthropic secures $300M Series C for Claude AI",
            "summary":     "Anthropic raised $300M to continue developing constitutional AI and the Claude assistant.",
            "url":         "https://techcrunch.com/anthropic-series-c",
            "date":        "2023-05-01",
            "source":      "techcrunch",
        },
    ]

    print("\n=== add_papers ===")
    r = vs.add_papers(SAMPLE_PAPERS)
    print(f"  added={r['added']} skipped={r['skipped']}")

    print("\n=== add_articles ===")
    r = vs.add_articles(SAMPLE_ARTICLES)
    print(f"  added={r['added']} skipped={r['skipped']}")

    print("\n=== get_collection_counts ===")
    counts = vs.get_collection_counts()
    print(f"  papers={counts['papers']} articles={counts['articles']}")

    print("\n=== search_papers: 'self-attention neural network' ===")
    for res in vs.search_papers("self-attention neural network", n_results=2):
        print(f"  [{res['similarity_score']:5.1f}] {res['text'][:80]}...")

    print("\n=== search_articles: 'AI startup fundraising' ===")
    for res in vs.search_articles("AI startup fundraising", n_results=2):
        print(f"  [{res['similarity_score']:5.1f}] {res['text'][:80]}...")

    print("\n=== search_all: 'large language models investment' ===")
    for res in vs.search_all("large language models investment", n_results=3):
        print(f"  [{res['similarity_score']:5.1f}] [{res['collection']:8s}] {res['text'][:70]}...")

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("\nSmoke-test complete. Temp dir cleaned up.")
