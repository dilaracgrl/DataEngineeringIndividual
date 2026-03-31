"""
SQLite client for the Technology Pipeline Tracker.

Why SQLite here and MongoDB for raw documents?
──────────────────────────────────────────────
MongoDB stores raw, variable-shape documents from tools — a patent record
looks nothing like a GitHub repo record. MongoDB handles this heterogeneity
well. SQLite stores the *output* of scoring those documents: fixed-schema,
structured numbers and timestamps that need fast aggregation and comparison.

Concretely:
  MongoDB  → "give me all arXiv papers about diffusion models"
             (variable shape, large text fields, provenance blobs)
  SQLite   → "show me the trend score for diffusion models over the last
             30 days and whether the overall stage has changed"
             (fixed columns, numeric aggregations, GROUP BY queries)

SQLite is the right choice for the scoring layer because:
  1. No server process — the DB is a single file next to the code,
     trivially portable and version-control friendly (for schema migrations).
  2. The scoring schema is fixed — every query produces the same 10
     per-tool scores plus an overall stage and score. Relational tables
     fit this perfectly.
  3. Trend history comparisons (has stage changed? is score rising?) are
     trivial SQL queries — much simpler than MongoDB aggregation pipelines.
  4. The data volume is small — one row per (query, run), not millions of
     raw documents. SQLite handles this scale with zero infrastructure.

Tables
──────
  trend_scores        — one row per pipeline scoring run per query
  query_history       — audit log of every question asked to the system
  technology_timeline — milestone log of when a technology crossed each stage

The DB file path defaults to database/pipeline_tracker.db and can be
overridden via SQLITE_DB_PATH in .env.

Usage:
    from database.sqlite_client import SQLiteClient
    db = SQLiteClient()
    db.save_trend_score("diffusion models", scores)
    history = db.get_trend_history("diffusion models")
"""

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment & logging setup
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sqlite_client")

# Default DB file sits inside the database/ directory alongside this module.
# Override with SQLITE_DB_PATH in .env for a custom location.
_DEFAULT_DB_PATH = Path(__file__).parent / "pipeline_tracker.db"

# ── Per-tool score column names ──────────────────────────────────────────────
# These map exactly to the tools in the pipeline. Each score is 0–100.
# Keeping them as individual columns (not a JSON blob) means SQL can
# easily answer "which queries have a high arxiv_score but low patents_score?"
# without deserialising JSON on every row.
SCORE_COLUMNS = [
    "arxiv_score",            # Stage 1 — arXiv paper volume/recency
    "semantic_scholar_score", # Stage 1 — citation velocity/influence
    "github_score",           # Stage 2 — developer repo activity
    "producthunt_score",      # Stage 2 — Product Hunt launch presence
    "yc_score",               # Stage 2 — YC portfolio company count
    "news_score",             # Stage 3 — NewsAPI investment coverage
    "techcrunch_score",       # Stage 3 — TechCrunch funding timeline
    "patents_score",          # Stage 4 — PatentsView institutional filings
    "wikipedia_score",        # Stage 5 — Wikipedia article existence/views
    "trends_score",           # Stage 5 — Google Trends mainstream presence
]

# Valid stage values — the overall_stage column is constrained to these
VALID_STAGES = {1, 2, 3, 4, 5}


# ---------------------------------------------------------------------------
# DDL — table definitions
# ---------------------------------------------------------------------------

_DDL = """
-- ── trend_scores ─────────────────────────────────────────────────────────
-- One row per pipeline scoring run per query.
-- Each run captures all 10 per-tool scores at a point in time, plus the
-- derived overall_stage (1-5) and overall_score (0-100).
-- Tracking multiple rows over time for the same query reveals whether
-- a technology is advancing through the pipeline — the core output of
-- the system.
CREATE TABLE IF NOT EXISTS trend_scores (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    query                    TEXT    NOT NULL,
    timestamp                TEXT    NOT NULL,           -- ISO 8601 UTC

    -- Stage 1: Academic signals (0–100 each)
    arxiv_score              REAL    NOT NULL DEFAULT 0,
    semantic_scholar_score   REAL    NOT NULL DEFAULT 0,

    -- Stage 2: Developer/startup signals (0–100 each)
    github_score             REAL    NOT NULL DEFAULT 0,
    producthunt_score        REAL    NOT NULL DEFAULT 0,
    yc_score                 REAL    NOT NULL DEFAULT 0,

    -- Stage 3: Investment signals (0–100 each)
    news_score               REAL    NOT NULL DEFAULT 0,
    techcrunch_score         REAL    NOT NULL DEFAULT 0,

    -- Stage 4: Institutional/big-tech signals (0–100 each)
    patents_score            REAL    NOT NULL DEFAULT 0,

    -- Stage 5: Mainstream signals (0–100 each)
    wikipedia_score          REAL    NOT NULL DEFAULT 0,
    trends_score             REAL    NOT NULL DEFAULT 0,

    -- Derived summary — set by the scoring agent, not computed here
    overall_stage            INTEGER NOT NULL DEFAULT 1
                             CHECK (overall_stage BETWEEN 1 AND 5),
    overall_score            REAL    NOT NULL DEFAULT 0
                             CHECK (overall_score BETWEEN 0 AND 100)
);

-- Index on (query, timestamp) for the common access pattern:
-- "get recent score history for this query"
CREATE INDEX IF NOT EXISTS idx_trend_scores_query_ts
    ON trend_scores (query, timestamp DESC);


-- ── query_history ─────────────────────────────────────────────────────────
-- Audit log of every question asked to the system.
-- Stores the agent's full response text and the list of source tools
-- that contributed to the answer. Enables hallucination checking
-- (compare response claims against sources_used) and usage analytics.
-- sources_used is stored as a JSON array of tool names, e.g.:
--   ["arxiv_search", "github_repo_activity", "patents_search"]
CREATE TABLE IF NOT EXISTS query_history (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    query          TEXT    NOT NULL,
    timestamp      TEXT    NOT NULL,      -- ISO 8601 UTC
    agent_response TEXT    NOT NULL,      -- full text of the agent's answer
    sources_used   TEXT    NOT NULL       -- JSON array of tool/source names
                   DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_query_history_query
    ON query_history (query, timestamp DESC);


-- ── technology_timeline ────────────────────────────────────────────────────
-- Milestone log: records the *first time* a technology was detected at
-- each pipeline stage. One row per (technology, stage) pair — a given
-- technology should only appear at stage 3 once, even if stage 3 is
-- detected on multiple runs.
--
-- This table is the system's long-term memory of technology progression.
-- It answers: "When did diffusion models first get VC funding coverage?"
-- and "Has any technology skipped straight from stage 1 to stage 4?"
--
-- stage values: 1=Academic, 2=Startup, 3=Investment, 4=BigTech, 5=Mainstream
-- signal_source: which tool produced the detection, e.g. "techcrunch_scraper"
-- notes: free-text context, e.g. "First Series A article: OpenAI raises $10M"
CREATE TABLE IF NOT EXISTS technology_timeline (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    technology     TEXT    NOT NULL,
    stage          INTEGER NOT NULL
                   CHECK (stage BETWEEN 1 AND 5),
    first_detected TEXT    NOT NULL,      -- ISO 8601 UTC
    signal_source  TEXT    NOT NULL,      -- tool that produced the detection
    notes          TEXT    NOT NULL DEFAULT '',

    -- A technology can only be "first detected" at a stage once.
    -- If we see it again at the same stage, we UPDATE the notes rather
    -- than inserting a duplicate row.
    UNIQUE (technology, stage)
);

CREATE INDEX IF NOT EXISTS idx_timeline_technology
    ON technology_timeline (technology, stage);


-- ── monitoring_alerts ─────────────────────────────────────────────────────
-- One row per alert fired by the scheduled monitoring system.
-- Two alert types are currently defined:
--   STAGE_TRANSITION   — overall_stage changed between two consecutive runs
--   SIGNIFICANT_CHANGE — overall_score moved by more than 15 points
--
-- previous_stage / new_stage are NULL for SIGNIFICANT_CHANGE-only alerts.
-- previous_score / new_score are always populated.
--
-- This table is the permanent audit trail for the watchlist monitor.
-- It answers: "When was the first time quantum computing crossed Stage 4?"
-- and "How many score spikes has neuromorphic computing had?"
CREATE TABLE IF NOT EXISTS monitoring_alerts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    technology      TEXT    NOT NULL,
    alert_type      TEXT    NOT NULL,     -- 'STAGE_TRANSITION' | 'SIGNIFICANT_CHANGE'
    previous_stage  INTEGER,              -- NULL for score-only alerts
    new_stage       INTEGER,              -- NULL for score-only alerts
    previous_score  REAL,
    new_score       REAL,
    timestamp       TEXT    NOT NULL,     -- ISO 8601 UTC
    message         TEXT    NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_monitoring_alerts_technology
    ON monitoring_alerts (technology, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_monitoring_alerts_type
    ON monitoring_alerts (alert_type, timestamp DESC);
"""


# ---------------------------------------------------------------------------
# SQLiteClient
# ---------------------------------------------------------------------------

class SQLiteClient:
    """
    SQLite client for structured trend scores and pipeline timeline data.

    All reads and writes go through this class. The schema is created
    automatically on first connection — no migration step required.

    Thread safety: SQLite in WAL mode (enabled on connect) supports
    one writer and multiple readers concurrently. For single-process
    use this is sufficient. Do not share a SQLiteClient instance across
    threads — create one per thread instead.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        """
        Initialises the SQLiteClient.

        Args:
            db_path : Path to the SQLite database file.
                      Falls back to SQLITE_DB_PATH in .env, then to
                      database/pipeline_tracker.db next to this module.
        """
        self._db_path = Path(
            db_path
            or os.getenv("SQLITE_DB_PATH")
            or _DEFAULT_DB_PATH
        )
        self._initialised = False
        logger.debug("SQLiteClient targeting %s", self._db_path)

    # ── Connection management ────────────────────────────────────────────────

    @contextmanager
    def _connect(self):
        """
        Context manager that yields a sqlite3 Connection.

        Opens a fresh connection per operation — appropriate for a
        research/analysis tool where operations are infrequent. The
        connection is committed and closed on clean exit, rolled back
        on any exception.

        WAL mode: Write-Ahead Logging allows reads to proceed while a
        write is in progress. Essential when the pipeline and a notebook
        inspection are running simultaneously.

        detect_types=PARSE_DECLTYPES enables automatic type conversion
        for DATE and TIMESTAMP columns if added later.
        """
        conn = sqlite3.connect(
            self._db_path,
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        conn.row_factory = sqlite3.Row   # Rows as dict-like objects
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        """
        Creates all tables and indexes if they do not already exist.

        Called lazily on first use — no separate setup step needed.
        executescript() runs the DDL as a single transaction.
        """
        if self._initialised:
            return
        with self._connect() as conn:
            conn.executescript(_DDL)
        self._initialised = True
        logger.info("SQLite schema ready at %s", self._db_path)

    # ── Write operations ─────────────────────────────────────────────────────

    def save_trend_score(self, query: str, scores_dict: dict) -> int:
        """
        Saves a complete pipeline scoring run for `query` to trend_scores.

        scores_dict should contain all per-tool scores (0–100 each) plus
        overall_stage (1–5) and overall_score (0–100). Missing keys default
        to 0 so partial runs (e.g. when a tool is unavailable) still persist.

        Multiple rows per query are intended — this is how we track a
        technology's progress over time. Use get_trend_history() to read
        the full history and detect stage transitions.

        Args:
            query       : Technology or concept that was scored.
            scores_dict : Dict containing any subset of SCORE_COLUMNS plus
                          optional overall_stage and overall_score.

        Returns:
            The rowid (integer primary key) of the inserted row.

        Raises:
            ValueError  : If overall_stage is not 1–5.
            RuntimeError: On database write errors.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        overall_stage = int(scores_dict.get("overall_stage", 1))
        if overall_stage not in VALID_STAGES:
            raise ValueError(
                f"overall_stage must be 1–5, got {overall_stage}"
            )

        overall_score = float(scores_dict.get("overall_score", 0.0))
        if not (0 <= overall_score <= 100):
            raise ValueError(
                f"overall_score must be 0–100, got {overall_score}"
            )

        self._ensure_schema()
        timestamp = datetime.now(timezone.utc).isoformat()

        # Build the column→value mapping for all score columns.
        # We explicitly default missing scores to 0.0 so the INSERT
        # always provides values for every NOT NULL column.
        score_values = {
            col: float(scores_dict.get(col, 0.0))
            for col in SCORE_COLUMNS
        }

        columns = ["query", "timestamp"] + SCORE_COLUMNS + ["overall_stage", "overall_score"]
        placeholders = ", ".join(["?"] * len(columns))
        values = (
            [query.strip(), timestamp]
            + [score_values[col] for col in SCORE_COLUMNS]
            + [overall_stage, overall_score]
        )

        sql = f"INSERT INTO trend_scores ({', '.join(columns)}) VALUES ({placeholders})"

        try:
            with self._connect() as conn:
                cursor = conn.execute(sql, values)
                rowid = cursor.lastrowid
        except sqlite3.Error as e:
            logger.error("save_trend_score failed for query=%r: %s", query, e)
            raise RuntimeError(f"SQLite write error in save_trend_score: {e}") from e

        logger.info(
            "save_trend_score | query=%r | stage=%d | score=%.1f | rowid=%d",
            query, overall_stage, overall_score, rowid,
        )
        return rowid

    def log_query(
        self,
        query: str,
        response: str,
        sources: list[str],
    ) -> int:
        """
        Appends a row to query_history recording a system query and its response.

        Every time a user asks the system "where is X in the pipeline?", this
        method logs the full agent response and the list of tools that contributed
        data. This enables:
          - Usage analytics: what technologies are being tracked?
          - Hallucination auditing: does the response cite tools that weren't
            actually called?
          - Reproducibility: re-running with the same sources should give a
            similar answer.

        Args:
            query    : The user's technology question.
            response : The agent's full text response.
            sources  : List of tool/source names used, e.g.
                       ["arxiv_search", "github_repo_activity"].

        Returns:
            The rowid of the inserted row.

        Raises:
            RuntimeError: On database write errors.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        self._ensure_schema()
        timestamp = datetime.now(timezone.utc).isoformat()
        sources_json = json.dumps(sources if isinstance(sources, list) else [])

        sql = """
            INSERT INTO query_history (query, timestamp, agent_response, sources_used)
            VALUES (?, ?, ?, ?)
        """
        try:
            with self._connect() as conn:
                cursor = conn.execute(sql, [query.strip(), timestamp, response, sources_json])
                rowid = cursor.lastrowid
        except sqlite3.Error as e:
            logger.error("log_query failed for query=%r: %s", query, e)
            raise RuntimeError(f"SQLite write error in log_query: {e}") from e

        logger.info("log_query | query=%r | sources=%s | rowid=%d", query, sources, rowid)
        return rowid

    def save_timeline_event(
        self,
        technology: str,
        stage: int,
        source: str,
        notes: str = "",
    ) -> dict:
        """
        Records that `technology` was first detected at `stage` by `source`.

        Uses INSERT OR IGNORE so the first detection of a (technology, stage)
        pair is preserved and subsequent detections do not overwrite it —
        the milestone date should reflect when we *first* saw this transition,
        not the most recent confirmation.

        If notes is provided and the row already exists, the notes field is
        updated (notes can accumulate new evidence without changing the date).

        Args:
            technology : Technology name, e.g. "diffusion models".
            stage      : Pipeline stage 1–5.
            source     : Tool that produced the detection signal,
                         e.g. "techcrunch_scraper", "yc_scraper".
            notes      : Optional free-text context about the detection,
                         e.g. "YC W23: 12 companies in LLM infra space".

        Returns:
            dict: {"technology": ..., "stage": ..., "action": "inserted"|"noted"}

        Raises:
            ValueError  : If stage is not 1–5.
            RuntimeError: On database write errors.
        """
        if stage not in VALID_STAGES:
            raise ValueError(f"stage must be 1–5, got {stage}")
        if not technology or not technology.strip():
            raise ValueError("technology must be a non-empty string")

        self._ensure_schema()
        first_detected = datetime.now(timezone.utc).isoformat()

        try:
            with self._connect() as conn:
                # Try to insert — silently skip if (technology, stage) already exists
                conn.execute(
                    """
                    INSERT OR IGNORE INTO technology_timeline
                        (technology, stage, first_detected, signal_source, notes)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [technology.strip(), stage, first_detected, source, notes],
                )

                # If notes were provided, always append them to the existing row
                # so new evidence is captured even when the date stays fixed
                if notes:
                    conn.execute(
                        """
                        UPDATE technology_timeline
                        SET notes = CASE
                            WHEN notes = '' THEN ?
                            ELSE notes || ' | ' || ?
                        END
                        WHERE technology = ? AND stage = ? AND notes != ?
                        """,
                        [notes, notes, technology.strip(), stage, notes],
                    )
                    action = "noted"
                else:
                    action = "inserted"

        except sqlite3.Error as e:
            logger.error(
                "save_timeline_event failed for technology=%r stage=%d: %s",
                technology, stage, e,
            )
            raise RuntimeError(
                f"SQLite write error in save_timeline_event: {e}"
            ) from e

        logger.info(
            "save_timeline_event | technology=%r | stage=%d | source=%s | action=%s",
            technology, stage, source, action,
        )
        return {"technology": technology, "stage": stage, "action": action}

    # ── Read operations ──────────────────────────────────────────────────────

    def get_trend_history(self, query: str, limit: int = 10) -> list[dict]:
        """
        Returns the most recent scoring runs for `query`, newest first.

        Comparing rows over time answers: has the overall_stage increased?
        Is the overall_score trending upward? Which individual tool scores
        changed the most between runs?

        Each row includes all 10 per-tool scores plus overall_stage and
        overall_score, so the caller can compute deltas between runs.

        Args:
            query : Technology to retrieve history for.
            limit : Max rows to return (default 10 covers ~10 days of daily runs).

        Returns:
            List of dicts, newest first. Each dict has all trend_scores columns.

        Raises:
            RuntimeError: On database read errors.
        """
        limit = max(1, min(limit, 100))
        self._ensure_schema()

        try:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    SELECT *
                    FROM   trend_scores
                    WHERE  query = ?
                    ORDER  BY timestamp DESC
                    LIMIT  ?
                    """,
                    [query.strip(), limit],
                )
                rows = [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error("get_trend_history failed for query=%r: %s", query, e)
            raise RuntimeError(
                f"SQLite read error in get_trend_history: {e}"
            ) from e

        logger.info(
            "get_trend_history | query=%r | rows_returned=%d", query, len(rows)
        )
        return rows

    def get_technology_timeline(self, technology: str) -> list[dict]:
        """
        Returns all pipeline stage milestones for `technology`, ordered by stage.

        Each row represents when the technology was first detected at that stage.
        Reading all rows together tells the complete adoption story:
          Stage 1 at 2019-03 → academic research begins
          Stage 2 at 2021-06 → first YC companies appear
          Stage 3 at 2022-01 → VC funding press coverage begins
          Stage 4 at 2023-09 → first big-tech patent filings
          Stage 5 at 2024-02 → Wikipedia views trending upward

        Args:
            technology : Technology name to retrieve timeline for.

        Returns:
            List of dicts ordered by stage (1 → 5), each containing:
            technology, stage, first_detected, signal_source, notes.
            Returns an empty list if the technology has no timeline entries yet.

        Raises:
            RuntimeError: On database read errors.
        """
        self._ensure_schema()

        try:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    SELECT technology, stage, first_detected, signal_source, notes
                    FROM   technology_timeline
                    WHERE  technology = ?
                    ORDER  BY stage ASC
                    """,
                    [technology.strip()],
                )
                rows = [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error(
                "get_technology_timeline failed for technology=%r: %s", technology, e
            )
            raise RuntimeError(
                f"SQLite read error in get_technology_timeline: {e}"
            ) from e

        logger.info(
            "get_technology_timeline | technology=%r | stages_found=%d",
            technology, len(rows),
        )
        return rows

    # ── Monitoring alerts ─────────────────────────────────────────────────────

    def save_alert(
        self,
        technology: str,
        alert_type: str,
        prev_stage: int | None,
        new_stage: int | None,
        prev_score: float | None,
        new_score: float | None,
        message: str = "",
    ) -> int:
        """
        Persists a monitoring alert to the monitoring_alerts table.

        Called by scripts/monitor.py whenever a stage transition or
        significant score change is detected on the watchlist.

        Args:
            technology : Technology name, e.g. "quantum computing".
            alert_type : "STAGE_TRANSITION" | "SIGNIFICANT_CHANGE".
            prev_stage : Stage integer from the previous run (None for score-only).
            new_stage  : Stage integer from the current run  (None for score-only).
            prev_score : overall_score from the previous run.
            new_score  : overall_score from the current run.
            message    : Human-readable description of the alert.

        Returns:
            The rowid of the inserted alert row.
        """
        if not technology or not technology.strip():
            raise ValueError("technology must be a non-empty string")
        if alert_type not in ("STAGE_TRANSITION", "SIGNIFICANT_CHANGE"):
            raise ValueError(
                f"alert_type must be STAGE_TRANSITION or SIGNIFICANT_CHANGE, "
                f"got {alert_type!r}"
            )

        self._ensure_schema()
        timestamp = datetime.now(timezone.utc).isoformat()

        try:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO monitoring_alerts
                        (technology, alert_type, previous_stage, new_stage,
                         previous_score, new_score, timestamp, message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        technology.strip(), alert_type,
                        prev_stage, new_stage,
                        prev_score, new_score,
                        timestamp, message,
                    ],
                )
                rowid = cursor.lastrowid
        except sqlite3.Error as e:
            logger.error("save_alert failed for technology=%r: %s", technology, e)
            raise RuntimeError(f"SQLite write error in save_alert: {e}") from e

        logger.info(
            "save_alert | technology=%r | type=%s | stage %s→%s | "
            "score %.1f→%.1f | rowid=%d",
            technology, alert_type,
            prev_stage, new_stage,
            prev_score or 0.0, new_score or 0.0,
            rowid,
        )
        return rowid

    def get_alerts(self, limit: int = 20) -> list[dict]:
        """
        Returns the most recent monitoring alerts across all technologies,
        newest first.

        Args:
            limit : Maximum rows to return (capped at 200).

        Returns:
            List of dicts with all monitoring_alerts columns.
        """
        limit = max(1, min(limit, 200))
        self._ensure_schema()

        try:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    SELECT id, technology, alert_type,
                           previous_stage, new_stage,
                           previous_score, new_score,
                           timestamp, message
                    FROM   monitoring_alerts
                    ORDER  BY timestamp DESC
                    LIMIT  ?
                    """,
                    [limit],
                )
                rows = [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error("get_alerts failed: %s", e)
            raise RuntimeError(f"SQLite read error in get_alerts: {e}") from e

        logger.info("get_alerts | rows_returned=%d", len(rows))
        return rows

    def get_alerts_for_technology(self, technology: str) -> list[dict]:
        """
        Returns all monitoring alerts for a specific technology, newest first.

        Args:
            technology : Technology name to filter on (exact match).

        Returns:
            List of dicts with all monitoring_alerts columns, or [] if none.
        """
        if not technology or not technology.strip():
            raise ValueError("technology must be a non-empty string")

        self._ensure_schema()

        try:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    SELECT id, technology, alert_type,
                           previous_stage, new_stage,
                           previous_score, new_score,
                           timestamp, message
                    FROM   monitoring_alerts
                    WHERE  technology = ?
                    ORDER  BY timestamp DESC
                    """,
                    [technology.strip()],
                )
                rows = [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error(
                "get_alerts_for_technology failed for technology=%r: %s",
                technology, e,
            )
            raise RuntimeError(
                f"SQLite read error in get_alerts_for_technology: {e}"
            ) from e

        logger.info(
            "get_alerts_for_technology | technology=%r | rows_returned=%d",
            technology, len(rows),
        )
        return rows

    def get_all_tracked_technologies(self) -> list[str]:
        """
        Returns a sorted list of all technology names that have at least
        one entry in technology_timeline.

        Useful for the analyst agent to know which technologies the system
        has already profiled, avoiding redundant re-scoring.

        Returns:
            Sorted list of unique technology name strings.
        """
        self._ensure_schema()

        try:
            with self._connect() as conn:
                cursor = conn.execute(
                    "SELECT DISTINCT technology FROM technology_timeline ORDER BY technology"
                )
                return [row["technology"] for row in cursor.fetchall()]
        except sqlite3.Error as e:
            raise RuntimeError(
                f"SQLite read error in get_all_tracked_technologies: {e}"
            ) from e

    # ── Context manager support ──────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *_):
        # Nothing to close — connections are per-operation.
        # This context manager exists for symmetry with MongoDBClient.
        pass


# ---------------------------------------------------------------------------
# Module-level convenience instance
# ---------------------------------------------------------------------------

# A single shared instance for use across the pipeline.
# Import pattern:
#     from database.sqlite_client import sqlite_db
#     sqlite_db.save_trend_score("diffusion models", scores)
sqlite_db = SQLiteClient()


# ---------------------------------------------------------------------------
# Quick smoke-test — run directly to verify schema creation and all methods
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    # Use a temp DB so the smoke-test doesn't pollute the real DB
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        tmp_path = f.name

    print(f"\nSmoke-test DB: {tmp_path}")

    with SQLiteClient(db_path=tmp_path) as db:

        # ── save_trend_score ─────────────────────────────────────────────
        print("\n=== save_trend_score ===")
        rowid = db.save_trend_score(
            query="diffusion models",
            scores_dict={
                "arxiv_score":            85.0,
                "semantic_scholar_score": 78.0,
                "github_score":           90.0,
                "producthunt_score":      60.0,
                "yc_score":               55.0,
                "news_score":             45.0,
                "techcrunch_score":       40.0,
                "patents_score":          30.0,
                "wikipedia_score":        70.0,
                "trends_score":           65.0,
                "overall_stage":          3,
                "overall_score":          61.8,
            },
        )
        print(f"Inserted row id: {rowid}")

        # ── get_trend_history ────────────────────────────────────────────
        print("\n=== get_trend_history ===")
        history = db.get_trend_history("diffusion models", limit=5)
        for row in history:
            print(
                f"  [{row['timestamp'][:10]}] "
                f"stage={row['overall_stage']} "
                f"score={row['overall_score']} "
                f"arxiv={row['arxiv_score']} "
                f"github={row['github_score']}"
            )

        # ── log_query ────────────────────────────────────────────────────
        print("\n=== log_query ===")
        qid = db.log_query(
            query="diffusion models",
            response=(
                "Diffusion models are currently in Stage 3 (Investment Phase). "
                "arXiv shows 400+ papers; GitHub repos have 50k+ stars; "
                "TechCrunch coverage shows 12 funding articles in the last 90 days."
            ),
            sources=["arxiv_search", "github_repo_activity", "techcrunch_search_funding"],
        )
        print(f"Logged query id: {qid}")

        # ── save_timeline_event ──────────────────────────────────────────
        print("\n=== save_timeline_event ===")
        for stage, source, note in [
            (1, "arxiv_search",           "First arXiv papers on score-based generative models"),
            (2, "github_repo_activity",   "denoising-diffusion-pytorch: 8k stars"),
            (3, "techcrunch_search_funding", "Stability AI raises $101M Series A"),
        ]:
            result = db.save_timeline_event(
                technology="diffusion models",
                stage=stage,
                source=source,
                notes=note,
            )
            print(f"  Stage {stage}: {result['action']}")

        # ── get_technology_timeline ──────────────────────────────────────
        print("\n=== get_technology_timeline ===")
        timeline = db.get_technology_timeline("diffusion models")
        stage_labels = {1: "Academic", 2: "Developer", 3: "Investment", 4: "BigTech", 5: "Mainstream"}
        for event in timeline:
            label = stage_labels.get(event["stage"], "Unknown")
            print(
                f"  Stage {event['stage']} ({label:12s}) | "
                f"{event['first_detected'][:10]} | "
                f"{event['signal_source']:35s} | "
                f"{event['notes']}"
            )

        # ── get_all_tracked_technologies ─────────────────────────────────
        print("\n=== get_all_tracked_technologies ===")
        techs = db.get_all_tracked_technologies()
        print(f"  Tracked: {techs}")

    # Clean up temp file
    Path(tmp_path).unlink(missing_ok=True)
    print("\nSmoke-test complete.")
