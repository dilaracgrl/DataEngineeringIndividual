"""
W3C PROV Data Lineage Tracker

Every record that flows through the pipeline carries a provenance record
conforming to the W3C PROV Data Model (https://www.w3.org/TR/prov-dm/).

Core concepts used here:
  prov:Entity    — a piece of data (a fetched document, a cleaned record,
                   a score, a query result)
  prov:Activity  — a process that generated or transformed the entity
                   (fetch, clean, embed, score, query)
  prov:Agent     — the software component that carried out the activity
                   (tool name, pipeline stage, agent name)

Every provenance event is:
  1. Written to SQLite (durable, queryable audit log)
  2. Returned as a dict so callers can attach it to their records in-memory

Lineage chain for a typical record:
    fetch_activity   → prov:wasGeneratedBy →  raw_entity
    clean_activity   → prov:used           →  raw_entity
    clean_activity   → prov:wasGeneratedBy →  cleaned_entity
    embed_activity   → prov:used           →  cleaned_entity
    embed_activity   → prov:wasGeneratedBy →  stored_entity
    query_activity   → prov:used           →  stored_entity
    query_activity   → prov:wasGeneratedBy →  response_entity
"""

import json
import logging
import sqlite3
import sys
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Path setup — allow imports from project root
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

load_dotenv(_ROOT / ".env")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("lineage.tracker")

# ---------------------------------------------------------------------------
# DDL — lineage table lives in the same SQLite file as pipeline data
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS prov_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at     TEXT    NOT NULL,

    -- prov:Entity
    entity_id       TEXT    NOT NULL,   -- stable URI-style ID for this datum
    entity_type     TEXT    NOT NULL,   -- e.g. "raw_document", "cleaned_record", "score"
    entity_label    TEXT,               -- human-readable label (query or title)

    -- prov:Activity
    activity_id     TEXT    NOT NULL,   -- e.g. "fetch:arxiv:2024-01-15T10:00:00Z"
    activity_type   TEXT    NOT NULL,   -- "fetch" | "clean" | "embed" | "score" | "query"
    started_at      TEXT,
    ended_at        TEXT,

    -- prov:Agent
    agent_id        TEXT    NOT NULL,   -- e.g. "tool:arxiv_tool"
    agent_type      TEXT    NOT NULL,   -- "tool" | "scraper" | "pipeline" | "agent"
    agent_label     TEXT,

    -- Provenance relations (stored as JSON arrays of entity_ids)
    was_generated_by    TEXT,           -- JSON list of activity_ids
    was_derived_from    TEXT,           -- JSON list of source entity_ids
    used                TEXT,           -- JSON list of entity_ids this activity consumed

    -- Payload snapshot (abridged — not full document)
    attributes      TEXT                -- JSON dict of key metadata
);

CREATE INDEX IF NOT EXISTS idx_prov_entity_id  ON prov_events(entity_id);
CREATE INDEX IF NOT EXISTS idx_prov_activity   ON prov_events(activity_type);
CREATE INDEX IF NOT EXISTS idx_prov_agent      ON prov_events(agent_id);
CREATE INDEX IF NOT EXISTS idx_prov_recorded   ON prov_events(recorded_at);
"""

# ---------------------------------------------------------------------------
# LineageTracker
# ---------------------------------------------------------------------------


class LineageTracker:
    """
    Records W3C PROV events to SQLite and returns provenance dicts that
    callers can embed directly into their data records.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        raw = db_path or os.getenv("SQLITE_DB_PATH", "database/pipeline_tracker.db")
        self._db_path = str(_ROOT / raw) if not os.path.isabs(raw) else raw
        self._init_db()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_DDL)
        logger.debug("Lineage DB ready at %s", self._db_path)

    def _connect(self):
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # ------------------------------------------------------------------
    # Core record method
    # ------------------------------------------------------------------

    def record(
        self,
        *,
        entity_id: str,
        entity_type: str,
        entity_label: str,
        activity_type: str,          # "fetch" | "clean" | "embed" | "score" | "query"
        agent_id: str,               # e.g. "tool:arxiv_tool"
        agent_type: str,             # "tool" | "scraper" | "pipeline" | "agent"
        agent_label: str = "",
        started_at: Optional[str] = None,
        ended_at: Optional[str] = None,
        was_derived_from: Optional[list[str]] = None,
        used: Optional[list[str]] = None,
        attributes: Optional[dict] = None,
    ) -> dict:
        """
        Record one provenance event and return the prov dict.

        Returns a dict with all prov fields so the caller can attach it to
        their data record as record["prov"] = tracker.record(...).
        """
        now = datetime.now(timezone.utc).isoformat()
        activity_id = f"{activity_type}:{agent_id}:{now}"

        prov = {
            "prov:entity": entity_id,
            "prov:entity_type": entity_type,
            "prov:activity": activity_id,
            "prov:activity_type": activity_type,
            "prov:agent": agent_id,
            "prov:agent_type": agent_type,
            "prov:wasGeneratedBy": activity_id,
            "prov:wasDerivedFrom": was_derived_from or [],
            "prov:used": used or [],
            "prov:generatedAtTime": ended_at or now,
            "prov:startedAtTime": started_at or now,
        }

        row = {
            "recorded_at": now,
            "entity_id": entity_id,
            "entity_type": entity_type,
            "entity_label": entity_label,
            "activity_id": activity_id,
            "activity_type": activity_type,
            "started_at": started_at or now,
            "ended_at": ended_at or now,
            "agent_id": agent_id,
            "agent_type": agent_type,
            "agent_label": agent_label or agent_id,
            "was_generated_by": json.dumps([activity_id]),
            "was_derived_from": json.dumps(was_derived_from or []),
            "used": json.dumps(used or []),
            "attributes": json.dumps(attributes or {}),
        }

        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO prov_events (
                        recorded_at, entity_id, entity_type, entity_label,
                        activity_id, activity_type, started_at, ended_at,
                        agent_id, agent_type, agent_label,
                        was_generated_by, was_derived_from, used, attributes
                    ) VALUES (
                        :recorded_at, :entity_id, :entity_type, :entity_label,
                        :activity_id, :activity_type, :started_at, :ended_at,
                        :agent_id, :agent_type, :agent_label,
                        :was_generated_by, :was_derived_from, :used, :attributes
                    )
                    """,
                    row,
                )
        except Exception as exc:
            logger.warning("Lineage write failed for %s: %s", entity_id, exc)

        return prov

    # ------------------------------------------------------------------
    # Convenience helpers for each pipeline stage
    # ------------------------------------------------------------------

    def record_fetch(
        self, *, tool: str, query: str, doc_id: str, doc_type: str,
        doc_label: str = "", attributes: Optional[dict] = None,
    ) -> dict:
        """Record a raw-document fetch event."""
        return self.record(
            entity_id=f"raw:{tool}:{doc_id}",
            entity_type="raw_document",
            entity_label=doc_label or query,
            activity_type="fetch",
            agent_id=f"tool:{tool}",
            agent_type="tool",
            agent_label=tool,
            attributes={"query": query, "doc_type": doc_type, **(attributes or {})},
        )

    def record_clean(
        self, *, pipeline_stage: str, source_entity_id: str,
        clean_id: str, label: str = "", attributes: Optional[dict] = None,
    ) -> dict:
        """Record a cleaning/normalisation event."""
        return self.record(
            entity_id=f"clean:{clean_id}",
            entity_type="cleaned_record",
            entity_label=label,
            activity_type="clean",
            agent_id=f"pipeline:{pipeline_stage}",
            agent_type="pipeline",
            agent_label=pipeline_stage,
            was_derived_from=[source_entity_id],
            used=[source_entity_id],
            attributes=attributes or {},
        )

    def record_embed(
        self, *, store: str, clean_entity_id: str,
        embed_id: str, label: str = "", attributes: Optional[dict] = None,
    ) -> dict:
        """Record an embedding/storage event."""
        return self.record(
            entity_id=f"embed:{store}:{embed_id}",
            entity_type="stored_entity",
            entity_label=label,
            activity_type="embed",
            agent_id=f"pipeline:embedder",
            agent_type="pipeline",
            agent_label="embedder",
            was_derived_from=[clean_entity_id],
            used=[clean_entity_id],
            attributes={"store": store, **(attributes or {})},
        )

    def record_query(
        self, *, agent: str, query: str, response_id: str,
        used_entities: Optional[list[str]] = None,
        attributes: Optional[dict] = None,
    ) -> dict:
        """Record an agent query event."""
        return self.record(
            entity_id=f"response:{agent}:{response_id}",
            entity_type="agent_response",
            entity_label=query,
            activity_type="query",
            agent_id=f"agent:{agent}",
            agent_type="agent",
            agent_label=agent,
            used=used_entities or [],
            attributes={"query": query, **(attributes or {})},
        )

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_lineage_for_entity(self, entity_id: str) -> list[dict]:
        """Return all provenance events for a given entity_id."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM prov_events WHERE entity_id = ? ORDER BY recorded_at",
                (entity_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_lineage_for_query(self, query: str, limit: int = 50) -> list[dict]:
        """Return provenance events where the label matches the query."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM prov_events
                WHERE entity_label LIKE ?
                ORDER BY recorded_at DESC
                LIMIT ?
                """,
                (f"%{query}%", limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_activity_log(
        self, activity_type: Optional[str] = None, limit: int = 100
    ) -> list[dict]:
        """Return the most recent provenance events, optionally filtered by type."""
        with self._connect() as conn:
            if activity_type:
                rows = conn.execute(
                    """
                    SELECT * FROM prov_events
                    WHERE activity_type = ?
                    ORDER BY recorded_at DESC LIMIT ?
                    """,
                    (activity_type, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM prov_events ORDER BY recorded_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(r) for r in rows]

    def get_agent_activity_summary(self) -> list[dict]:
        """Return a count of events per agent."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT agent_id, agent_type, activity_type,
                       COUNT(*) AS event_count,
                       MIN(recorded_at) AS first_seen,
                       MAX(recorded_at) AS last_seen
                FROM prov_events
                GROUP BY agent_id, agent_type, activity_type
                ORDER BY event_count DESC
                """,
            ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Module-level convenience instance
# ---------------------------------------------------------------------------

lineage_tracker = LineageTracker()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t = LineageTracker()

    p1 = t.record_fetch(
        tool="arxiv_tool",
        query="quantum computing",
        doc_id="2401.12345",
        doc_label="Advances in Quantum Error Correction",
        doc_type="paper",
        attributes={"stage": 1},
    )
    print("Fetch prov:", json.dumps(p1, indent=2))

    p2 = t.record_clean(
        pipeline_stage="cleaner",
        source_entity_id=p1["prov:entity"],
        clean_id="2401.12345",
        label="Advances in Quantum Error Correction",
    )
    print("Clean prov:", json.dumps(p2, indent=2))

    p3 = t.record_embed(
        store="chromadb",
        clean_entity_id=p2["prov:entity"],
        embed_id="2401.12345",
        label="Advances in Quantum Error Correction",
    )
    print("Embed prov:", json.dumps(p3, indent=2))

    log = t.get_lineage_for_query("quantum computing")
    print(f"\nLineage events for 'quantum computing': {len(log)}")
    for ev in log:
        print(f"  [{ev['activity_type']}] {ev['agent_id']} → {ev['entity_id']}")

    summary = t.get_agent_activity_summary()
    print("\nAgent activity summary:")
    for row in summary:
        print(f"  {row['agent_id']} ({row['activity_type']}): {row['event_count']} events")
