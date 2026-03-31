"""
FastAPI Agent Service — A2A + SSE endpoint layer

Exposes the ResearcherAgent and AnalystAgent over HTTP so that:
  - The Analyst can call the Researcher via HTTP (A2A communication)
  - External clients (UI, notebooks, curl) can query the pipeline

Endpoints
──────────
  POST /research/pipeline         — run_pipeline (blocking, returns JSON)
  POST /research/semantic_search  — RAG search (blocking)
  POST /research/graph_context    — Neo4j neighbourhood (blocking)
  GET  /research/trend_history    — SQLite history (blocking)
  GET  /research/timeline         — SQLite timeline (blocking)

  POST /analyse                   — full analysis, blocking JSON response
  POST /analyse/stream            — full analysis, SSE streaming response

  GET  /tools                     — list all MCP tool definitions
  GET  /lineage                   — query the PROV event log
  GET  /health                    — liveness check

A2A flow
─────────
  Client → POST /analyse/stream → AnalystAgent.analyse_stream()
                                        → ResearcherAgent.full_research()
                                              → DataEmbedder.run_full_pipeline()
                                        → Claude API (streaming)
                                  ← SSE token events
                                  ← SSE complete event (full assessment JSON)
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
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

from agents.researcher import ResearcherAgent, TOOL_DEFINITIONS as RESEARCHER_TOOLS
from agents.analyst import AnalystAgent, _sse
from database.mongo_client import MongoDBClient
from lineage.tracker import LineageTracker

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("api.agent_service")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Technology Pipeline Tracker — Agent Service",
    description=(
        "Multi-agent API for assessing where a technology sits in its journey "
        "from academic research to mainstream product. "
        "Uses Researcher + Analyst agents with RAG, GraphRAG, and Claude."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Agent singletons (lazy — initialised on first request)
# ---------------------------------------------------------------------------

_researcher: Optional[ResearcherAgent] = None
_analyst:    Optional[AnalystAgent]    = None
_lineage:    Optional[LineageTracker]  = None
_mongo:      Optional[MongoDBClient]   = None

_CACHE_TTL_HOURS = 24


def _get_researcher() -> ResearcherAgent:
    global _researcher
    if _researcher is None:
        _researcher = ResearcherAgent()
    return _researcher


def _get_analyst() -> AnalystAgent:
    global _analyst
    if _analyst is None:
        _analyst = AnalystAgent()
    return _analyst


def _get_lineage() -> LineageTracker:
    global _lineage
    if _lineage is None:
        _lineage = LineageTracker()
    return _lineage


def _get_mongo() -> MongoDBClient:
    global _mongo
    if _mongo is None:
        _mongo = MongoDBClient()
    return _mongo


# ---------------------------------------------------------------------------
# Analysis result cache helpers (24-hour TTL, stored in MongoDB)
# ---------------------------------------------------------------------------

def _cache_key(query: str) -> str:
    """Normalise query to a stable cache key."""
    return query.strip().lower()


def _cache_get(query: str) -> Optional[dict]:
    """Return a cached analysis result if it exists and is < 24 h old."""
    try:
        coll = _get_mongo().get_analysis_cache()
        doc  = coll.find_one({"query": _cache_key(query)})
        if not doc:
            return None
        cached_at = datetime.fromisoformat(doc["cached_at"])
        if datetime.now(timezone.utc) - cached_at > timedelta(hours=_CACHE_TTL_HOURS):
            return None
        return doc
    except Exception as exc:
        logger.warning("Cache read failed: %s", exc)
        return None


def _cache_put(query: str, scores: dict, assessment: dict,
               velocity: dict, raw_critique: str, logged_at: str) -> None:
    """Upsert an analysis result into the cache."""
    try:
        coll = _get_mongo().get_analysis_cache()
        coll.replace_one(
            {"query": _cache_key(query)},
            {
                "query":       _cache_key(query),
                "cached_at":   datetime.now(timezone.utc).isoformat(),
                "scores":      scores,
                "assessment":  assessment,
                "velocity":    velocity,
                "raw_critique": raw_critique,
                "logged_at":   logged_at,
            },
            upsert=True,
        )
        logger.info("Cached analysis result for query=%r", query)
    except Exception as exc:
        logger.warning("Cache write failed: %s", exc)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    query: str = Field(..., description="Technology or concept to research.")


class PipelineRequest(QueryRequest):
    force_refresh: bool = Field(False, description="Re-fetch even if recent data exists.")


class SemanticSearchRequest(QueryRequest):
    n_results: int = Field(5, ge=1, le=20)
    collection: str = Field("all", pattern="^(papers|articles|all)$")


class GraphContextRequest(BaseModel):
    technology: str = Field(..., description="Technology name in the graph.")
    include_related: bool = Field(True)


class ToolCallRequest(BaseModel):
    tool_name: str
    params: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dashboard UI
# ---------------------------------------------------------------------------

_UI_FILE = _ROOT / "ui" / "index.html"


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    """Serve the single-page intelligence dashboard."""
    if not _UI_FILE.exists():
        raise HTTPException(status_code=404, detail="ui/index.html not found")
    return FileResponse(str(_UI_FILE), media_type="text/html")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health", tags=["System"])
def health() -> dict:
    return {
        "status": "ok",
        "service": "agent_service",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Tool catalogue
# ---------------------------------------------------------------------------


@app.get("/tools", tags=["System"])
def list_tools() -> dict:
    """Return all MCP-style tool definitions registered in the Researcher."""
    return {"tools": RESEARCHER_TOOLS, "count": len(RESEARCHER_TOOLS)}


# ---------------------------------------------------------------------------
# Researcher endpoints
# ---------------------------------------------------------------------------


@app.post("/research/pipeline", tags=["Researcher"])
def run_pipeline(req: PipelineRequest) -> dict:
    """
    Run the full fetch → clean → embed → score pipeline for a query.
    Blocking — waits for all 9 data sources before responding.
    """
    researcher = _get_researcher()
    result = researcher.call_tool(
        "run_pipeline",
        {"query": req.query, "force_refresh": req.force_refresh},
    )
    if result["error"]:
        raise HTTPException(status_code=500, detail=result["error"])
    # Strip heavy raw data from response; keep scores + summary
    pipeline = result["result"]
    return {
        "query": req.query,
        "scores": pipeline.get("scores"),
        "raw_summary": pipeline.get("raw", {}).get("summary"),
        "storage": pipeline.get("storage"),
        "fetched_at": pipeline.get("fetched_at"),
        "finished_at": pipeline.get("finished_at"),
        "prov": pipeline.get("prov"),
    }


@app.post("/research/semantic_search", tags=["Researcher"])
def semantic_search(req: SemanticSearchRequest) -> dict:
    """RAG search over ChromaDB papers and/or articles."""
    researcher = _get_researcher()
    result = researcher.call_tool(
        "semantic_search",
        {"query": req.query, "n_results": req.n_results, "collection": req.collection},
    )
    if result["error"]:
        raise HTTPException(status_code=500, detail=result["error"])
    return result["result"]


@app.post("/research/graph_context", tags=["Researcher"])
def graph_context(req: GraphContextRequest) -> dict:
    """Neo4j technology neighbourhood."""
    researcher = _get_researcher()
    result = researcher.call_tool(
        "graph_context",
        {"technology": req.technology, "include_related": req.include_related},
    )
    if result["error"]:
        raise HTTPException(status_code=500, detail=result["error"])
    return result["result"]


@app.get("/research/trend_history", tags=["Researcher"])
def trend_history(
    query: str = Query(..., description="Query to look up"),
    limit: int = Query(10, ge=1, le=50),
) -> dict:
    """Historical trend scores from SQLite."""
    researcher = _get_researcher()
    result = researcher.call_tool("get_trend_history", {"query": query, "limit": limit})
    if result["error"]:
        raise HTTPException(status_code=500, detail=result["error"])
    return result["result"]


@app.get("/research/timeline", tags=["Researcher"])
def timeline(
    technology: str = Query(..., description="Technology name"),
) -> dict:
    """Stage-detection timeline from SQLite."""
    researcher = _get_researcher()
    result = researcher.call_tool("get_timeline", {"technology": technology})
    if result["error"]:
        raise HTTPException(status_code=500, detail=result["error"])
    return result["result"]


@app.post("/research/tool", tags=["Researcher"])
def call_researcher_tool(req: ToolCallRequest) -> dict:
    """
    Generic tool dispatcher — call any Researcher tool by name.
    Useful for A2A: the Analyst can POST here with any tool_name + params.
    """
    researcher = _get_researcher()
    result = researcher.call_tool(req.tool_name, req.params)
    if result["error"]:
        raise HTTPException(status_code=500, detail=result["error"])
    return result


# ---------------------------------------------------------------------------
# Analyst endpoints
# ---------------------------------------------------------------------------


@app.post("/analyse", tags=["Analyst"])
def analyse(req: QueryRequest) -> dict:
    """
    Full pipeline + Claude reasoning — blocking JSON response.

    Returns the complete assessment including:
    - confirmed_stage (1-5)
    - narrative with inline citations
    - evidence_by_stage
    - conflicting_signals
    - next_stage_prediction
    - sources_cited
    """
    analyst = _get_analyst()
    try:
        result = analyst.analyse(req.query)
    except EnvironmentError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error("Analyst error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "query": result["query"],
        "assessment": result["assessment"],
        "prov": result["prov"],
        "logged_at": result["logged_at"],
    }


@app.post("/analyse/stream", tags=["Analyst"])
def analyse_stream(req: QueryRequest) -> StreamingResponse:
    """
    Full pipeline + Claude reasoning — SSE streaming response.

    Checks the 24-hour MongoDB cache before running the pipeline.
    On a cache hit the full result is returned instantly with
    "cached": true in the complete event.

    Event stream format:
        data: {"event": "status",   "message": "...", "elapsed": 1.2}
        data: {"event": "scores",   "scores": {...}}
        data: {"event": "token",    "token": "..."}
        data: {"event": "complete", "assessment": {...}, "cached": false, ...}
        data: {"event": "error",    "message": "..."}
    """
    analyst  = _get_analyst()
    query    = req.query
    t0       = datetime.now(timezone.utc)

    def _elapsed() -> float:
        return round((datetime.now(timezone.utc) - t0).total_seconds(), 1)

    def generate() -> Iterator[str]:
        # ── Cache check ──────────────────────────────────────────────────────
        cached = _cache_get(query)
        if cached:
            logger.info("Cache hit for query=%r", query)
            yield _sse({"event": "status", "message": "Returning cached result (< 24 h old)...", "elapsed": _elapsed()})
            scores = cached.get("scores", {})
            yield _sse({
                "event":   "scores",
                "scores":  scores,
                "message": f"Stage {scores.get('overall_stage')} (cached)",
            })
            yield _sse({
                "event":        "complete",
                "query":        query,
                "assessment":   cached.get("assessment", {}),
                "velocity":     cached.get("velocity", {}),
                "raw_critique": cached.get("raw_critique", ""),
                "prov":         None,
                "logged_at":    cached.get("logged_at", ""),
                "cached":       True,
                "cached_at":    cached.get("cached_at", ""),
                "elapsed":      _elapsed(),
            })
            return

        # ── Fresh run ────────────────────────────────────────────────────────
        try:
            scores_snapshot: dict  = {}
            complete_payload: dict = {}

            for chunk in analyst.analyse_stream(query):
                # Intercept scores and complete events to cache after streaming
                if chunk.startswith("data: "):
                    try:
                        payload = json.loads(chunk[6:])
                        ev = payload.get("event", "")
                        if ev == "scores":
                            scores_snapshot = payload.get("scores", {})
                        elif ev == "complete":
                            complete_payload = payload
                    except Exception:
                        pass
                yield chunk

            # Persist to cache once the stream is complete
            if complete_payload:
                _cache_put(
                    query        = query,
                    scores       = scores_snapshot,
                    assessment   = complete_payload.get("assessment", {}),
                    velocity     = complete_payload.get("velocity", {}),
                    raw_critique = complete_payload.get("raw_critique", ""),
                    logged_at    = complete_payload.get("logged_at", ""),
                )

        except EnvironmentError as exc:
            yield _sse({"event": "error", "message": str(exc)})
        except Exception as exc:
            logger.error("Stream error: %s", exc, exc_info=True)
            yield _sse({"event": "error", "message": str(exc)})

    return StreamingResponse(generate(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Monitoring alerts endpoints
# ---------------------------------------------------------------------------


@app.get("/alerts", tags=["Monitoring"])
def get_alerts(limit: int = Query(20, ge=1, le=200)) -> dict:
    """
    Returns the most recent monitoring alerts across all watchlist technologies.

    Alerts are written by scripts/monitor.py on each scheduled run.
    Two types:
      STAGE_TRANSITION   — overall pipeline stage changed
      SIGNIFICANT_CHANGE — overall score moved > 15 points
    """
    sqlite = _get_lineage()._sqlite if hasattr(_get_lineage(), "_sqlite") \
        else __import__("database.sqlite_client", fromlist=["SQLiteClient"]).SQLiteClient()
    try:
        from database.sqlite_client import SQLiteClient as _SC
        alerts = _SC().get_alerts(limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"alerts": alerts, "count": len(alerts)}


@app.get("/alerts/{technology}", tags=["Monitoring"])
def get_alerts_for_technology(technology: str) -> dict:
    """
    Returns all monitoring alerts for a specific technology name.

    URL-encode spaces: /alerts/quantum%20computing
    """
    try:
        from database.sqlite_client import SQLiteClient as _SC
        alerts = _SC().get_alerts_for_technology(technology)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"technology": technology, "alerts": alerts, "count": len(alerts)}


# ---------------------------------------------------------------------------
# Lineage endpoint
# ---------------------------------------------------------------------------


@app.get("/lineage", tags=["Observability"])
def get_lineage(
    query: Optional[str] = Query(None, description="Filter events by query label"),
    activity_type: Optional[str] = Query(None, description="Filter by activity type"),
    limit: int = Query(50, ge=1, le=500),
) -> dict:
    """
    Query the W3C PROV event log.

    Returns provenance events for auditing data lineage and checking
    for hallucinations (every cited source should appear in the log).
    """
    tracker = _get_lineage()
    if query:
        events = tracker.get_lineage_for_query(query, limit=limit)
    else:
        events = tracker.get_activity_log(activity_type=activity_type, limit=limit)

    return {"events": events, "count": len(events)}


@app.get("/lineage/summary", tags=["Observability"])
def lineage_summary() -> dict:
    """Return a count of provenance events per agent/activity type."""
    tracker = _get_lineage()
    return {"summary": tracker.get_agent_activity_summary()}


# ---------------------------------------------------------------------------
# Entry point for direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("API_PORT", "8000"))
    logger.info("Starting agent service on port %d", port)
    uvicorn.run(
        "api.agent_service:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info",
    )
