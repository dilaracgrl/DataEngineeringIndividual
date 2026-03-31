"""
Researcher Agent

The Researcher is the data-collection half of the two-agent system.
It exposes an MCP-style tool catalogue and, when invoked, runs the full
pipeline (fetch → clean → embed → score) for a given query.

Responsibilities
────────────────
1. Accept a query from the Analyst agent (or directly from the API).
2. Call pipeline/embedder.py::run_full_pipeline(query).
3. Supplement the structured scores with semantic RAG results from
   ChromaDB and relationship context from Neo4j.
4. Record W3C PROV lineage for every data access.
5. Return a rich ResearchResult dict that the Analyst can reason over.

Agent-to-Agent (A2A) communication
────────────────────────────────────
The Researcher exposes its tools via the FastAPI service in api/agent_service.py.
The Analyst calls those endpoints over HTTP.  Inside this module the Researcher
can also be used directly (no HTTP hop) for tests / CLI use.

Tool catalogue (MCP-style)
───────────────────────────
  run_pipeline        — full fetch + score for a query
  semantic_search     — RAG over ChromaDB papers + articles
  graph_context       — Neo4j neighbourhood for a technology name
  get_trend_history   — historical scores from SQLite
  get_timeline        — stage-detection timeline from SQLite
"""

import json
import logging
import sys
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

load_dotenv(_ROOT / ".env")

# ---------------------------------------------------------------------------
# Internal imports
# ---------------------------------------------------------------------------

from pipeline.embedder import DataEmbedder
from database.vector_store import VectorStore
from database.graph_client import GraphClient
from database.sqlite_client import SQLiteClient
from lineage.tracker import LineageTracker

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("agents.researcher")

# ---------------------------------------------------------------------------
# MCP Tool Definitions
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "run_pipeline",
        "description": (
            "Run the full data-collection and scoring pipeline for a technology "
            "query. Fetches from all 9 sources (arXiv, Semantic Scholar, GitHub, "
            "Product Hunt, YC, NewsAPI, TechCrunch, PatentsView, Wikipedia/Trends), "
            "stores results in MongoDB + ChromaDB + Neo4j + SQLite, and returns "
            "stage scores (1–5) with supporting evidence."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Technology or business concept to research.",
                },
                "force_refresh": {
                    "type": "boolean",
                    "description": "Re-fetch even if recent data exists. Default: false.",
                    "default": False,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "semantic_search",
        "description": (
            "Semantic RAG search across stored academic papers and news/funding "
            "articles using ChromaDB vector embeddings. Returns the most relevant "
            "documents with similarity scores."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "n_results": {
                    "type": "integer",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 20,
                },
                "collection": {
                    "type": "string",
                    "enum": ["papers", "articles", "all"],
                    "default": "all",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "graph_context",
        "description": (
            "Retrieve the Neo4j knowledge graph neighbourhood for a technology: "
            "related companies, papers, investors, and other technologies that "
            "share connections. Supports GraphRAG reasoning."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "technology": {
                    "type": "string",
                    "description": "Technology name as stored in the graph.",
                },
                "include_related": {
                    "type": "boolean",
                    "description": "Also return related technologies. Default: true.",
                    "default": True,
                },
            },
            "required": ["technology"],
        },
    },
    {
        "name": "get_trend_history",
        "description": (
            "Return historical trend scores from SQLite for a query — useful for "
            "detecting score changes over repeated pipeline runs."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 10, "minimum": 1, "maximum": 50},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_timeline",
        "description": (
            "Return the stage-detection timeline for a technology from SQLite: "
            "when each pipeline stage was first detected and from which source."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "technology": {"type": "string"},
            },
            "required": ["technology"],
        },
    },
]

# ---------------------------------------------------------------------------
# ResearcherAgent
# ---------------------------------------------------------------------------


class ResearcherAgent:
    """
    Data-collection and retrieval agent.

    Wraps the pipeline and all four storage layers behind a clean tool
    interface.  Every data access is recorded via LineageTracker.
    """

    def __init__(self) -> None:
        self._embedder = DataEmbedder()
        self._vector_store = VectorStore()
        self._graph = GraphClient()
        self._sqlite = SQLiteClient()
        self._lineage = LineageTracker()
        logger.info("ResearcherAgent initialised")

    # ------------------------------------------------------------------
    # Public tool dispatcher
    # ------------------------------------------------------------------

    def call_tool(self, tool_name: str, params: dict) -> dict:
        """
        Dispatch a tool call by name.

        Returns:
            {"tool": tool_name, "result": ..., "prov": ..., "error": None}
            or
            {"tool": tool_name, "result": None, "prov": None, "error": "message"}
        """
        _dispatch = {
            "run_pipeline": self.run_pipeline,
            "semantic_search": self.semantic_search,
            "graph_context": self.graph_context,
            "get_trend_history": self.get_trend_history,
            "get_timeline": self.get_timeline,
        }
        fn = _dispatch.get(tool_name)
        if fn is None:
            return {
                "tool": tool_name,
                "result": None,
                "prov": None,
                "error": f"Unknown tool: {tool_name}",
            }
        try:
            result = fn(**params)
            return {"tool": tool_name, "result": result, "prov": result.get("prov"), "error": None}
        except Exception as exc:
            logger.error("Tool %s failed: %s", tool_name, exc, exc_info=True)
            return {"tool": tool_name, "result": None, "prov": None, "error": str(exc)}

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def run_pipeline(self, query: str, force_refresh: bool = False) -> dict:
        """
        Full fetch → clean → embed → score pipeline.

        Returns the complete pipeline result dict plus a prov record.
        """
        logger.info("run_pipeline: query=%r", query)
        started = datetime.now(timezone.utc).isoformat()

        result = self._embedder.run_full_pipeline(query)

        ended = datetime.now(timezone.utc).isoformat()
        prov = self._lineage.record_query(
            agent="researcher",
            query=query,
            response_id=f"pipeline:{query}:{ended}",
            attributes={
                "tool": "run_pipeline",
                "overall_stage": result.get("scores", {}).get("overall_stage"),
                "overall_score": result.get("scores", {}).get("overall_score"),
                "started_at": started,
                "ended_at": ended,
            },
        )
        result["prov"] = prov
        return result

    def semantic_search(
        self,
        query: str,
        n_results: int = 5,
        collection: str = "all",
    ) -> dict:
        """
        RAG search over ChromaDB.

        Returns:
            {
                "query": str,
                "collection": str,
                "results": list[dict],  # sorted by similarity_score desc
                "prov": dict
            }
        """
        logger.info("semantic_search: query=%r collection=%s n=%d", query, collection, n_results)

        if collection == "papers":
            results = self._vector_store.search_papers(query, n_results=n_results)
        elif collection == "articles":
            results = self._vector_store.search_articles(query, n_results=n_results)
        else:
            results = self._vector_store.search_all(query, n_results=n_results)

        prov = self._lineage.record_query(
            agent="researcher",
            query=query,
            response_id=f"rag:{collection}:{datetime.now(timezone.utc).isoformat()}",
            attributes={"tool": "semantic_search", "collection": collection, "n_results": len(results)},
        )

        return {"query": query, "collection": collection, "results": results, "prov": prov}

    def graph_context(
        self,
        technology: str,
        include_related: bool = True,
    ) -> dict:
        """
        Neo4j neighbourhood for a technology.

        Returns:
            {
                "technology": str,
                "graph": dict,            # from GraphClient.get_technology_graph()
                "related": list[dict],    # from GraphClient.get_related_technologies()
                "prov": dict
            }
        """
        logger.info("graph_context: technology=%r", technology)

        graph = self._graph.get_technology_graph(technology)
        related = self._graph.get_related_technologies(technology) if include_related else []

        prov = self._lineage.record_query(
            agent="researcher",
            query=technology,
            response_id=f"graph:{technology}:{datetime.now(timezone.utc).isoformat()}",
            attributes={
                "tool": "graph_context",
                "companies": len(graph.get("companies", [])),
                "papers": len(graph.get("papers", [])),
                "related": len(related),
            },
        )

        return {"technology": technology, "graph": graph, "related": related, "prov": prov}

    def get_trend_history(self, query: str, limit: int = 10) -> dict:
        """
        Historical trend scores from SQLite.

        Returns:
            {"query": str, "history": list[dict], "prov": dict}
        """
        logger.info("get_trend_history: query=%r limit=%d", query, limit)
        history = self._sqlite.get_trend_history(query, limit=limit)
        prov = self._lineage.record_query(
            agent="researcher",
            query=query,
            response_id=f"history:{query}:{datetime.now(timezone.utc).isoformat()}",
            attributes={"tool": "get_trend_history", "records": len(history)},
        )
        return {"query": query, "history": history, "prov": prov}

    def get_timeline(self, technology: str) -> dict:
        """
        Stage-detection timeline from SQLite.

        Returns:
            {"technology": str, "timeline": list[dict], "prov": dict}
        """
        logger.info("get_timeline: technology=%r", technology)
        timeline = self._sqlite.get_technology_timeline(technology)
        prov = self._lineage.record_query(
            agent="researcher",
            query=technology,
            response_id=f"timeline:{technology}:{datetime.now(timezone.utc).isoformat()}",
            attributes={"tool": "get_timeline", "events": len(timeline)},
        )
        return {"technology": technology, "timeline": timeline, "prov": prov}

    # ------------------------------------------------------------------
    # Compound research method (used by the Analyst)
    # ------------------------------------------------------------------

    def full_research(self, query: str) -> dict:
        """
        Run pipeline + RAG + graph in one call.

        This is the primary method the Analyst agent calls.  It returns
        a ResearchResult dict containing all evidence layers needed for
        stage assessment and narrative generation.

        Returns:
            {
                "query": str,
                "scores": dict,           # 10 per-tool scores + overall
                "rag_papers": list[dict], # top semantic matches
                "rag_articles": list[dict],
                "graph": dict,            # tech graph from Neo4j
                "related_technologies": list[dict],
                "timeline": list[dict],   # stage-detection history
                "trend_history": list[dict],
                "raw_summary": dict,      # fetch_all summary counts
                "prov": list[dict],       # all lineage records
            }
        """
        logger.info("full_research: query=%r", query)

        pipeline_result = self.run_pipeline(query)
        scores = pipeline_result.get("scores", {})
        raw_summary = pipeline_result.get("raw", {}).get("summary", {})

        rag_papers = self.semantic_search(query, n_results=5, collection="papers")
        rag_articles = self.semantic_search(query, n_results=5, collection="articles")
        graph_data = self.graph_context(query)
        timeline = self.get_timeline(query)
        history = self.get_trend_history(query, limit=5)

        return {
            "query": query,
            "scores": scores,
            "rag_papers": rag_papers["results"],
            "rag_articles": rag_articles["results"],
            "graph": graph_data["graph"],
            "related_technologies": graph_data["related"],
            "timeline": timeline["timeline"],
            "trend_history": history["history"],
            "raw_summary": raw_summary,
            "prov": [
                pipeline_result.get("prov"),
                rag_papers.get("prov"),
                rag_articles.get("prov"),
                graph_data.get("prov"),
                timeline.get("prov"),
                history.get("prov"),
            ],
        }


# ---------------------------------------------------------------------------
# Module-level convenience instance
# ---------------------------------------------------------------------------

researcher = ResearcherAgent()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "transformer neural network"
    print(f"\nRunning semantic_search for: {query!r}\n")

    agent = ResearcherAgent()
    result = agent.semantic_search(query, n_results=3)
    print(f"Results ({len(result['results'])}):")
    for r in result["results"]:
        score = r.get("similarity_score", 0)
        title = r.get("title", r.get("name", "?"))
        print(f"  [{score:.1f}%] {title}")

    print("\nTool catalogue:")
    for td in TOOL_DEFINITIONS:
        print(f"  {td['name']}: {td['description'][:60]}...")
