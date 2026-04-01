"""
Neo4j graph client for mapping relationships between technologies,
companies, researchers, and investors.

Why a graph database — and why it reveals what MongoDB and SQLite cannot
────────────────────────────────────────────────────────────────────────
MongoDB and SQLite answer "what do we know about X?". Neo4j answers
"how is X connected to everything else?" — a fundamentally different
question, and the one that exposes the hidden structure of how ideas
travel from academia into products.

Consider the path a technology like "diffusion models" takes:
  - A paper authored by researcher A at university U gets cited by
    researcher B who then joins startup S.
  - Startup S gets funded by investor V in a YC batch.
  - Big tech company C acquires startup S two years later.
  - Researcher A goes on to found startup S2, also backed by V.

MongoDB stores paper records and company records in separate collections.
SQLite stores score numbers. Neither can answer: "which companies are
connected through shared researchers?" or "which investor keeps appearing
just before a technology breaks into mainstream?" or "what is the
shortest path from this academic paper to this big-tech product?"

Those are graph traversal questions. Neo4j's Cypher query language is
built for them. The query `MATCH (t:Technology)<-[:WORKS_ON]-(c:Company)
<-[:FUNDED_BY]-(i:Investor)` would instantly reveal every investor
funding companies working on a technology — a join that would require
multiple MongoDB queries and manual stitching.

Practical example for the pipeline tracker:
  - If two competing technologies share 80% of the same companies, one
    will likely absorb the other (graph distance collapses over time).
  - If a technology has papers from 5 universities but all the companies
    trace back to researchers from one lab, it reveals the origin cluster.
  - "Related technologies" (technologies sharing companies or papers) can
    reveal convergence patterns years before market analysts notice them.

Node types
──────────
  Technology  — the concept being tracked (e.g. "diffusion models")
  Company     — startup or corporate entity (YC company, acquired, etc.)
  Paper       — academic paper (arXiv or Semantic Scholar)
  Investor    — VC fund or angel investor

Relationship types
──────────────────
  WORKS_ON    — Company → Technology (a company is building in this space)
  RESEARCHES  — Paper → Technology (a paper is about this technology)
  FUNDED_BY   — Company → Investor (an investor funded this company)
  ACQUIRED_BY — Company → Company (M&A event)

All nodes use MERGE (create-or-update) so re-running the pipeline never
creates duplicates — the graph reflects current known state.

Required .env keys:
    NEO4J_URI      — bolt://localhost:7687 (local) or neo4j+s://... (Aura)
    NEO4J_USER     — neo4j (default)
    NEO4J_PASSWORD — your password

Usage:
    from database.graph_client import GraphClient
    g = GraphClient()
    g.add_technology("diffusion models", "Generative model using denoising", 1)
    g.add_company("Stability AI", stage=3, source="techcrunch", batch=None)
    g.link_company_to_technology("Stability AI", "diffusion models")
    graph = g.get_technology_graph("diffusion models")
"""

import logging
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import (
    AuthError,
    ServiceUnavailable,
    Neo4jError,
)

# ---------------------------------------------------------------------------
# Environment & logging setup
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("graph_client")

# Valid relationship types — enforced to prevent Cypher injection via
# relationship type strings (Neo4j does not support parameterised rel types)
_VALID_COMPANY_RELS = {"WORKS_ON", "FUNDED_BY", "ACQUIRED_BY"}
_VALID_PAPER_RELS   = {"RESEARCHES"}

# Pipeline stage labels for human-readable node properties
_STAGE_LABELS = {
    1: "Academic",
    2: "Developer/Startup",
    3: "Investment",
    4: "Big Tech",
    5: "Mainstream",
}


# ---------------------------------------------------------------------------
# GraphClient
# ---------------------------------------------------------------------------

class GraphClient:
    """
    Neo4j graph client for the Technology Pipeline Tracker.

    All reads and writes use parameterised Cypher queries — no string
    interpolation into query bodies, which would create Cypher injection
    risk. The one exception is relationship type names, which Neo4j does
    not support as parameters; these are validated against an allowlist
    before use.

    Lazy connection: the Neo4j driver is initialised on first use.
    """

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ) -> None:
        """
        Args:
            uri      : Neo4j connection URI. Falls back to NEO4J_URI in .env,
                       then to bolt://localhost:7687.
            user     : Neo4j username. Falls back to NEO4J_USER, then "neo4j".
            password : Neo4j password. Falls back to NEO4J_PASSWORD.
        """
        self._uri      = uri      or os.getenv("NEO4J_URI",      "bolt://localhost:7687")
        self._user     = user     or os.getenv("NEO4J_USER",     "neo4j")
        self._password = password or os.getenv("NEO4J_PASSWORD", "")
        self._driver   = None
        logger.debug("GraphClient targeting %s as %s", self._uri, self._user)

    # ── Connection management ────────────────────────────────────────────────

    def _get_driver(self):
        """
        Returns the Neo4j driver, creating it on first call.

        Neo4j drivers maintain a connection pool internally — one driver
        per process is the recommended pattern. We create it lazily so
        importing this module does not immediately require Neo4j to be up.

        Raises:
            RuntimeError: If credentials are missing or connection fails.
        """
        if self._driver is None:
            if not self._password:
                raise RuntimeError(
                    "NEO4J_PASSWORD is not set. Add it to .env.\n"
                    "For local Neo4j: set the password you chose during setup.\n"
                    "For Neo4j Aura: copy the password from the Aura console."
                )
            try:
                self._driver = GraphDatabase.driver(
                    self._uri,
                    auth=(self._user, self._password),
                    # connection_timeout: fail fast rather than hanging
                    connection_timeout=5,
                )
                # Verify connectivity immediately — raises ServiceUnavailable
                # if the database is unreachable rather than failing silently
                self._driver.verify_connectivity()
                logger.info("Neo4j connected at %s", self._uri)
            except AuthError as e:
                self._driver = None
                raise RuntimeError(
                    f"Neo4j authentication failed at '{self._uri}'. "
                    f"Check NEO4J_USER and NEO4J_PASSWORD in .env. ({e})"
                ) from e
            except ServiceUnavailable as e:
                self._driver = None
                raise RuntimeError(
                    f"Neo4j is not reachable at '{self._uri}'. "
                    f"Check that the database is running. ({e})"
                ) from e
        return self._driver

    def _run(self, query: str, parameters: dict | None = None) -> list[dict]:
        """
        Executes a Cypher query in an auto-commit transaction and returns
        results as a list of plain dicts.

        Auto-commit (session.run) is appropriate for single-statement
        operations. For multi-statement transactions that must succeed or
        fail together, use session.execute_write() — see _run_write().

        Raises:
            RuntimeError: On Neo4j query errors.
        """
        try:
            with self._get_driver().session() as session:
                result = session.run(query, parameters or {})
                return [dict(record) for record in result]
        except Neo4jError as e:
            logger.error("Cypher query failed: %s\nQuery: %s", e, query[:200])
            raise RuntimeError(f"Neo4j query error: {e}") from e

    def _run_write(self, query: str, parameters: dict | None = None) -> list[dict]:
        """
        Executes a write Cypher query inside an explicit write transaction.

        Using execute_write() gives Neo4j's automatic retry on transient
        errors (e.g. deadlocks in a multi-writer scenario), which is
        better practice than raw session.run() for mutations.
        """
        def _tx(tx):
            result = tx.run(query, parameters or {})
            return [dict(record) for record in result]

        try:
            with self._get_driver().session() as session:
                return session.execute_write(_tx)
        except Neo4jError as e:
            logger.error("Cypher write failed: %s\nQuery: %s", e, query[:200])
            raise RuntimeError(f"Neo4j write error: {e}") from e

    def close(self) -> None:
        """Closes the Neo4j driver and releases the connection pool."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            logger.debug("Neo4j connection closed")

    # ── Schema constraints ───────────────────────────────────────────────────

    def ensure_constraints(self) -> None:
        """
        Creates uniqueness constraints on node identity properties.

        Constraints serve two purposes:
          1. Correctness: MERGE on a constrained property is atomic —
             no race condition between "does this node exist?" and "create it".
          2. Performance: Neo4j creates an index automatically for every
             constraint, making node lookups O(log n) instead of O(n).

        Call this once after connecting, e.g. in a setup script or on
        first run. Subsequent calls are no-ops (CREATE CONSTRAINT IF NOT EXISTS).
        """
        constraints = [
            "CREATE CONSTRAINT tech_name IF NOT EXISTS FOR (t:Technology) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT company_name IF NOT EXISTS FOR (c:Company) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.arxiv_id IS UNIQUE",
            "CREATE CONSTRAINT investor_name IF NOT EXISTS FOR (i:Investor) REQUIRE i.name IS UNIQUE",
        ]
        for cypher in constraints:
            try:
                self._run(cypher)
            except RuntimeError as e:
                # Constraint may already exist under a different name — log and continue
                logger.warning("Constraint creation skipped: %s", e)
        logger.info("Neo4j schema constraints ensured")

    # ── Node creation ────────────────────────────────────────────────────────

    def add_technology(
        self,
        name: str,
        description: str = "",
        first_seen_stage: int = 1,
    ) -> dict:
        """
        Creates or updates a Technology node.

        Technology nodes are the central hubs in the graph — all other
        nodes connect to them. Every MERGE on technology name is idempotent:
        re-adding "diffusion models" updates its properties without creating
        a duplicate node.

        first_seen_stage records the pipeline stage at which we first
        detected this technology (usually 1 — academic). As the pipeline
        tracks the technology over time, this field stays fixed at its
        first value (ON CREATE SET) so we preserve the discovery date.

        Args:
            name             : Technology name (unique node identity key).
            description      : Short description of the technology.
            first_seen_stage : Pipeline stage 1–5 when first detected.

        Returns:
            dict with node properties.
        """
        if not name or not name.strip():
            raise ValueError("Technology name must be a non-empty string")

        stage_label = _STAGE_LABELS.get(first_seen_stage, "Unknown")
        now = datetime.now(timezone.utc).isoformat()

        results = self._run_write(
            """
            MERGE (t:Technology {name: $name})
            ON CREATE SET
                t.description      = $description,
                t.first_seen_stage = $first_seen_stage,
                t.stage_label      = $stage_label,
                t.created_at       = $now
            ON MATCH SET
                t.description      = CASE WHEN $description <> '' THEN $description ELSE t.description END,
                t.updated_at       = $now
            RETURN t.name AS name, t.first_seen_stage AS first_seen_stage,
                   t.created_at AS created_at
            """,
            {
                "name": name.strip(),
                "description": description,
                "first_seen_stage": first_seen_stage,
                "stage_label": stage_label,
                "now": now,
            },
        )
        logger.info("add_technology | name=%r | stage=%d", name, first_seen_stage)
        return results[0] if results else {}

    def add_company(
        self,
        name: str,
        stage: int,
        source: str,
        batch: str | None = None,
    ) -> dict:
        """
        Creates or updates a Company node.

        Company nodes represent startups, scale-ups, and corporates.
        The stage property records the pipeline stage at which we first
        found this company (2 for YC/ProductHunt, 3 for TechCrunch funding,
        4 for big-tech acquirers). This lets graph queries like "show me all
        companies that appeared only at Stage 4" reveal late-stage consolidators.

        The batch property (e.g. "W23") is YC-specific and None for all
        other sources — it is only set ON CREATE to preserve the original
        batch assignment.

        Args:
            name   : Company name (unique node identity key).
            stage  : Pipeline stage 2–4 where this company was first found.
            source : Tool that produced this company, e.g. "ycombinator",
                     "techcrunch", "producthunt".
            batch  : YC batch code (e.g. "W24") or None.

        Returns:
            dict with node properties.
        """
        if not name or not name.strip():
            raise ValueError("Company name must be a non-empty string")

        now = datetime.now(timezone.utc).isoformat()

        results = self._run_write(
            """
            MERGE (c:Company {name: $name})
            ON CREATE SET
                c.stage      = $stage,
                c.source     = $source,
                c.batch      = $batch,
                c.created_at = $now
            ON MATCH SET
                c.updated_at = $now,
                c.source     = $source
            RETURN c.name AS name, c.stage AS stage,
                   c.batch AS batch, c.source AS source
            """,
            {
                "name":  name.strip(),
                "stage": stage,
                "source": source,
                "batch": batch or "",
                "now":   now,
            },
        )
        logger.info("add_company | name=%r | stage=%d | source=%s", name, stage, source)
        return results[0] if results else {}

    def add_paper(
        self,
        arxiv_id: str,
        title: str,
        year: int | None = None,
    ) -> dict:
        """
        Creates or updates a Paper node.

        Paper nodes are Stage 1 entities. Connecting papers to technologies
        (via RESEARCHES) and eventually to companies (when paper authors
        found startups) creates the academic-to-commercial lineage that is
        the most valuable insight in the graph.

        arxiv_id is the unique identity key. For Semantic Scholar papers
        without an arXiv ID, pass the paper_id prefixed with "s2:" to
        keep the namespace distinct.

        Args:
            arxiv_id : arXiv ID (e.g. "1706.03762") or "s2:{paper_id}".
            title    : Paper title.
            year     : Publication year (int).

        Returns:
            dict with node properties.
        """
        if not arxiv_id or not arxiv_id.strip():
            raise ValueError("arxiv_id must be a non-empty string")

        now = datetime.now(timezone.utc).isoformat()

        results = self._run_write(
            """
            MERGE (p:Paper {arxiv_id: $arxiv_id})
            ON CREATE SET
                p.title      = $title,
                p.year       = $year,
                p.url        = $url,
                p.created_at = $now
            ON MATCH SET
                p.title      = CASE WHEN $title <> '' THEN $title ELSE p.title END,
                p.updated_at = $now
            RETURN p.arxiv_id AS arxiv_id, p.title AS title, p.year AS year
            """,
            {
                "arxiv_id": arxiv_id.strip(),
                "title":    title,
                "year":     year,
                "url":      f"https://arxiv.org/abs/{arxiv_id.strip()}",
                "now":      now,
            },
        )
        logger.info("add_paper | arxiv_id=%r | year=%s", arxiv_id, year)
        return results[0] if results else {}

    def add_investor(self, name: str, investor_type: str = "") -> dict:
        """
        Creates or updates an Investor node.

        Investor nodes are Stage 3 entities. Connecting investors to
        companies (FUNDED_BY) reveals the VC ecosystem around a technology.
        When the same investor appears across multiple competing technologies,
        it signals they have a thesis about a technology category — a strong
        indicator of sustained investment-phase activity.

        Args:
            name          : Investor or fund name (unique identity key).
            investor_type : Optional classification, e.g. "VC", "corporate",
                            "angel", "accelerator".

        Returns:
            dict with node properties.
        """
        if not name or not name.strip():
            raise ValueError("Investor name must be a non-empty string")

        now = datetime.now(timezone.utc).isoformat()

        results = self._run_write(
            """
            MERGE (i:Investor {name: $name})
            ON CREATE SET
                i.investor_type = $investor_type,
                i.created_at    = $now
            ON MATCH SET
                i.updated_at    = $now
            RETURN i.name AS name, i.investor_type AS investor_type
            """,
            {"name": name.strip(), "investor_type": investor_type, "now": now},
        )
        logger.info("add_investor | name=%r", name)
        return results[0] if results else {}

    # ── Relationship creation ────────────────────────────────────────────────

    def link_company_to_technology(
        self,
        company_name: str,
        technology_name: str,
        relationship: str = "WORKS_ON",
    ) -> dict:
        """
        Creates a relationship from a Company node to a Technology node.

        This is the core edge in the pipeline graph. "Stability AI WORKS_ON
        diffusion models" encodes the claim that a real company is building
        commercial products in this technology space — the Stage 2→3 signal.

        Relationship type is validated against _VALID_COMPANY_RELS before
        use. Neo4j does not support parameterised relationship type names
        (unlike node labels and property values), so we whitelist them
        explicitly to prevent injection.

        MERGE on the relationship means re-linking the same pair is a no-op
        after the first call — safe to run after every pipeline iteration.

        Args:
            company_name      : Name of an existing Company node.
            technology_name   : Name of an existing Technology node.
            relationship      : Relationship type. One of:
                                "WORKS_ON"   — company builds in this space
                                "FUNDED_BY"  — (use link_company_to_investor)
                                "ACQUIRED_BY" — company was acquired by another

        Returns:
            dict confirming the relationship was created/merged.

        Raises:
            ValueError  : If relationship type is not in the allowlist.
            RuntimeError: If either node does not exist, or on Neo4j error.
        """
        if relationship not in _VALID_COMPANY_RELS:
            raise ValueError(
                f"Invalid relationship '{relationship}'. "
                f"Must be one of: {_VALID_COMPANY_RELS}"
            )

        now = datetime.now(timezone.utc).isoformat()

        # Relationship type is from a validated allowlist — safe to interpolate
        query = f"""
            MATCH (c:Company     {{name: $company_name}})
            MATCH (t:Technology  {{name: $technology_name}})
            MERGE (c)-[r:{relationship}]->(t)
            ON CREATE SET r.created_at = $now
            RETURN c.name AS company, type(r) AS relationship, t.name AS technology
        """
        results = self._run_write(
            query,
            {
                "company_name":    company_name.strip(),
                "technology_name": technology_name.strip(),
                "now":             now,
            },
        )
        if not results:
            raise RuntimeError(
                f"Could not create {relationship} link: one or both nodes not found. "
                f"Call add_company('{company_name}') and add_technology('{technology_name}') first."
            )
        logger.info(
            "link_company_to_technology | %s -[%s]-> %s",
            company_name, relationship, technology_name,
        )
        return results[0]

    def link_paper_to_technology(
        self,
        arxiv_id: str,
        technology_name: str,
        relationship: str = "RESEARCHES",
    ) -> dict:
        """
        Creates a relationship from a Paper node to a Technology node.

        "Paper 1706.03762 RESEARCHES transformers" encodes that the
        Attention paper is about the transformer technology. Over time,
        as companies are added and linked to transformers, the graph
        captures the full lineage: academic paper → developer activity →
        startup → VC funding → big-tech product.

        Args:
            arxiv_id         : arXiv ID of an existing Paper node.
            technology_name  : Name of an existing Technology node.
            relationship     : Relationship type. Currently only "RESEARCHES".

        Returns:
            dict confirming the relationship.

        Raises:
            ValueError  : If relationship type is not in the allowlist.
            RuntimeError: If either node does not exist, or on Neo4j error.
        """
        if relationship not in _VALID_PAPER_RELS:
            raise ValueError(
                f"Invalid relationship '{relationship}'. "
                f"Must be one of: {_VALID_PAPER_RELS}"
            )

        now = datetime.now(timezone.utc).isoformat()

        query = f"""
            MATCH (p:Paper      {{arxiv_id: $arxiv_id}})
            MATCH (t:Technology {{name: $technology_name}})
            MERGE (p)-[r:{relationship}]->(t)
            ON CREATE SET r.created_at = $now
            RETURN p.arxiv_id AS paper, type(r) AS relationship, t.name AS technology
        """
        results = self._run_write(
            query,
            {
                "arxiv_id":        arxiv_id.strip(),
                "technology_name": technology_name.strip(),
                "now":             now,
            },
        )
        if not results:
            raise RuntimeError(
                f"Could not create {relationship} link: one or both nodes not found. "
                f"Call add_paper('{arxiv_id}') and add_technology('{technology_name}') first."
            )
        logger.info(
            "link_paper_to_technology | %s -[%s]-> %s",
            arxiv_id, relationship, technology_name,
        )
        return results[0]

    def link_company_to_investor(
        self,
        company_name: str,
        investor_name: str,
    ) -> dict:
        """
        Creates a FUNDED_BY relationship from a Company node to an Investor node.

        This edge is the Stage 3 signal in graph form. When multiple companies
        working on the same technology are all funded by the same investor, it
        reveals an investor with a thesis — a strong predictor of continued
        investment-phase activity and eventual big-tech acquisition interest.

        Args:
            company_name  : Name of an existing Company node.
            investor_name : Name of an existing Investor node.

        Returns:
            dict confirming the relationship.
        """
        now = datetime.now(timezone.utc).isoformat()

        results = self._run_write(
            """
            MATCH (c:Company  {name: $company_name})
            MATCH (i:Investor {name: $investor_name})
            MERGE (c)-[r:FUNDED_BY]->(i)
            ON CREATE SET r.created_at = $now
            RETURN c.name AS company, type(r) AS relationship, i.name AS investor
            """,
            {
                "company_name":  company_name.strip(),
                "investor_name": investor_name.strip(),
                "now":           now,
            },
        )
        if not results:
            raise RuntimeError(
                f"Could not create FUNDED_BY link: one or both nodes not found. "
                f"Ensure add_company('{company_name}') and add_investor('{investor_name}') "
                "were called first."
            )
        logger.info(
            "link_company_to_investor | %s -[FUNDED_BY]-> %s",
            company_name, investor_name,
        )
        return results[0]

    def link_acquisition(
        self,
        acquired_company: str,
        acquiring_company: str,
    ) -> dict:
        """
        Creates an ACQUIRED_BY relationship between two Company nodes.

        Acquisitions are the Stage 4 signal in the graph. When a big-tech
        company acquires a startup working on a technology, it marks the
        transition from the investment phase to the institutional phase.
        Tracking these in the graph lets queries like "which technologies
        have been acquired by the same acquiring company?" reveal platform
        consolidation patterns.

        Args:
            acquired_company  : Name of the company that was acquired.
            acquiring_company : Name of the acquirer (often a big-tech company).

        Returns:
            dict confirming the relationship.
        """
        now = datetime.now(timezone.utc).isoformat()

        results = self._run_write(
            """
            MATCH (target:Company   {name: $acquired})
            MATCH (acquirer:Company {name: $acquirer})
            MERGE (target)-[r:ACQUIRED_BY]->(acquirer)
            ON CREATE SET r.created_at = $now
            RETURN target.name AS acquired, type(r) AS relationship,
                   acquirer.name AS acquirer
            """,
            {
                "acquired":  acquired_company.strip(),
                "acquirer":  acquiring_company.strip(),
                "now":       now,
            },
        )
        if not results:
            raise RuntimeError(
                f"Could not create ACQUIRED_BY link: ensure both company nodes exist. "
                f"add_company('{acquired_company}') and add_company('{acquiring_company}')"
            )
        logger.info(
            "link_acquisition | %s -[ACQUIRED_BY]-> %s",
            acquired_company, acquiring_company,
        )
        return results[0]

    # ── Read operations ──────────────────────────────────────────────────────

    def get_technology_graph(self, technology_name: str) -> dict:
        """
        Returns all nodes and relationships within 2 hops of a Technology node.

        This is the full neighbourhood query — it fetches every entity
        connected to the technology and their connections to each other.
        The analyst agent uses this to build a complete picture before
        scoring a technology's pipeline stage.

        The 2-hop radius captures:
          Hop 1: companies and papers directly linked to the technology
          Hop 2: investors who funded those companies; other technologies
                 that share companies with this one

        Graph structure returned:
          "technology"  — the central Technology node properties
          "companies"   — list of connected Company nodes
          "papers"      — list of connected Paper nodes
          "investors"   — list of Investors connected via FUNDED_BY
          "relationships" — list of {from, type, to} edge summaries

        Args:
            technology_name : Name of the Technology node to traverse from.

        Returns:
            dict with "technology", "companies", "papers", "investors",
            "relationships". All lists are empty if the technology has no
            connections yet.

        Raises:
            RuntimeError: If the technology node does not exist, or on Neo4j error.
        """
        # Step 1: verify the technology node exists
        tech_result = self._run(
            "MATCH (t:Technology {name: $name}) RETURN properties(t) AS props",
            {"name": technology_name.strip()},
        )
        if not tech_result:
            raise RuntimeError(
                f"Technology node '{technology_name}' not found. "
                f"Call add_technology('{technology_name}') first."
            )

        technology_props = tech_result[0]["props"]

        # Step 2: fetch all companies connected to this technology
        companies = self._run(
            """
            MATCH (c:Company)-[r]->(t:Technology {name: $name})
            RETURN properties(c) AS company, type(r) AS rel_type
            """,
            {"name": technology_name.strip()},
        )

        # Step 3: fetch all papers connected to this technology
        papers = self._run(
            """
            MATCH (p:Paper)-[r]->(t:Technology {name: $name})
            RETURN properties(p) AS paper, type(r) AS rel_type
            """,
            {"name": technology_name.strip()},
        )

        # Step 4: fetch all investors reachable via company→technology path
        # This is the 2-hop query: Technology ← Company ← Investor
        investors = self._run(
            """
            MATCH (i:Investor)<-[:FUNDED_BY]-(c:Company)-[]->(t:Technology {name: $name})
            RETURN DISTINCT properties(i) AS investor, c.name AS via_company
            """,
            {"name": technology_name.strip()},
        )

        # Step 5: collect all relationship edges as simple triples for the
        # analyst agent to describe the graph in its response
        relationships = self._run(
            """
            MATCH (a)-[r]->(b)
            WHERE (a)-[]->(:Technology {name: $name})
               OR (:Technology {name: $name})<-[]-(a)
               OR (a)-[]->(:Company)-[]->(:Technology {name: $name})
            RETURN labels(a)[0] + ': ' + COALESCE(a.name, a.arxiv_id, '') AS from_node,
                   type(r)  AS rel_type,
                   labels(b)[0] + ': ' + COALESCE(b.name, b.arxiv_id, '') AS to_node
            LIMIT 100
            """,
            {"name": technology_name.strip()},
        )

        graph = {
            "technology":    technology_props,
            "companies":     [r["company"]  for r in companies],
            "papers":        [r["paper"]    for r in papers],
            "investors":     [r["investor"] for r in investors],
            "relationships": [
                {
                    "from": r["from_node"],
                    "type": r["rel_type"],
                    "to":   r["to_node"],
                }
                for r in relationships
            ],
        }

        logger.info(
            "get_technology_graph | tech=%r | companies=%d | papers=%d | investors=%d",
            technology_name,
            len(graph["companies"]),
            len(graph["papers"]),
            len(graph["investors"]),
        )
        return graph

    def get_related_technologies(self, technology_name: str) -> list[dict]:
        """
        Finds technologies that share companies or papers with `technology_name`.

        This is the hidden-connection query — the one that justifies the
        graph database. Two technologies sharing companies means the same
        people are building in both spaces simultaneously, which is a
        convergence signal. Two technologies sharing foundational papers
        means they have overlapping academic origins, often preceding a
        merger of the two fields.

        Real example: "diffusion models" and "text-to-image generation"
        would share 90% of companies and most foundational papers. A
        naive pipeline tracker treats them as separate technologies; the
        graph reveals they are the same wave. This guides the analyst
        agent to note "this technology is converging with X."

        Cypher explanation:
            Start at the target Technology.
            Walk backwards through any relationship to connected Companies/Papers.
            Walk forwards to *other* Technologies those nodes connect to.
            Exclude the original technology (t1.name <> t2.name).
            Count shared connectors as connection_strength.

        Args:
            technology_name : Technology to find related technologies for.

        Returns:
            List of dicts, each with:
                related_technology (str), connection_strength (int),
                shared_via (list of "Company: X" or "Paper: Y" strings)
            Sorted by connection_strength descending.

        Raises:
            RuntimeError: On Neo4j error.
        """
        results = self._run(
            """
            MATCH (t1:Technology {name: $name})<-[r1]-(connector)-[r2]->(t2:Technology)
            WHERE t2.name <> $name
            WITH t2.name  AS related_technology,
                 COUNT(connector) AS connection_strength,
                 COLLECT(
                     DISTINCT labels(connector)[0] + ': ' +
                     COALESCE(connector.name, connector.arxiv_id, 'unknown')
                 ) AS shared_via
            ORDER BY connection_strength DESC
            RETURN related_technology, connection_strength, shared_via
            """,
            {"name": technology_name.strip()},
        )

        logger.info(
            "get_related_technologies | tech=%r | found=%d related",
            technology_name, len(results),
        )
        return results

    # ── Context manager support ──────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------------
# Module-level convenience instance
# ---------------------------------------------------------------------------

# Import pattern:
#     from database.graph_client import graph_db
#     graph_db.add_technology("diffusion models", "...", first_seen_stage=1)
graph_db = GraphClient()


# ---------------------------------------------------------------------------
# Quick smoke-test — run directly to verify Neo4j connection and all methods
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    with GraphClient() as g:
        print("\n=== ensure_constraints ===")
        g.ensure_constraints()
        print("  Constraints created")

        print("\n=== add_technology ===")
        t = g.add_technology(
            "diffusion models",
            "Generative models that learn to reverse a noising process",
            first_seen_stage=1,
        )
        print(f"  {t}")

        print("\n=== add_company ===")
        g.add_company("Stability AI",   stage=3, source="techcrunch")
        g.add_company("Midjourney",     stage=3, source="techcrunch")
        g.add_company("DALL-E / OpenAI", stage=4, source="techcrunch")
        print("  3 companies added")

        print("\n=== add_paper ===")
        g.add_paper("2006.11239", "Denoising Diffusion Probabilistic Models", year=2020)
        g.add_paper("2112.10752", "High-Resolution Image Synthesis with Latent Diffusion Models", year=2021)
        print("  2 papers added")

        print("\n=== add_investor ===")
        g.add_investor("Coatue Management", investor_type="VC")
        g.add_investor("Y Combinator",      investor_type="accelerator")
        print("  2 investors added")

        print("\n=== link_company_to_technology ===")
        for co in ["Stability AI", "Midjourney", "DALL-E / OpenAI"]:
            r = g.link_company_to_technology(co, "diffusion models")
            print(f"  {r}")

        print("\n=== link_paper_to_technology ===")
        for pid in ["2006.11239", "2112.10752"]:
            r = g.link_paper_to_technology(pid, "diffusion models")
            print(f"  {r}")

        print("\n=== link_company_to_investor ===")
        r = g.link_company_to_investor("Stability AI", "Coatue Management")
        print(f"  {r}")

        print("\n=== get_technology_graph ===")
        graph = g.get_technology_graph("diffusion models")
        print(f"  Technology : {graph['technology']['name']}")
        print(f"  Companies  : {[c['name'] for c in graph['companies']]}")
        print(f"  Papers     : {[p['arxiv_id'] for p in graph['papers']]}")
        print(f"  Investors  : {[i['name'] for i in graph['investors']]}")
        print(f"  Edges      : {len(graph['relationships'])} relationships")

        print("\n=== get_related_technologies ===")
        # Add a second technology with some shared companies to demonstrate
        g.add_technology("text-to-image generation", "AI that creates images from text", 2)
        g.link_company_to_technology("Stability AI", "text-to-image generation")
        g.link_company_to_technology("Midjourney",   "text-to-image generation")

        related = g.get_related_technologies("diffusion models")
        for r in related:
            print(
                f"  {r['related_technology']:35s} "
                f"strength={r['connection_strength']} "
                f"via={r['shared_via']}"
            )
