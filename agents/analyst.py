"""
Analyst Agent

The Analyst is the reasoning half of the two-agent system.  It receives a
ResearchResult from the Researcher, interprets the evidence, and produces a
structured stage assessment with a human-readable narrative.

Responsibilities
────────────────
1. Call ResearcherAgent.full_research(query) to gather evidence.
2. Send the evidence to Claude claude-sonnet-4-6 (via Anthropic API) with a
   structured prompt that asks it to:
   - Confirm or refine the computed stage (1-5)
   - Explain *why* using evidence from the data
   - Cite specific sources (paper titles, company names, news headlines)
   - Flag any conflicting signals
   - Suggest the most likely next stage transition
3. Stream the Claude response token-by-token via SSE (used by the API layer).
4. Log every tool call, input, and output for observability and hallucination
   checking.
5. Attach W3C PROV lineage to the final response.

Stage definitions (reminder)
──────────────────────────────
  1 — Academic     : Research papers only; no commercial activity
  2 — Developer    : OSS repos, hackathons, startup building
  3 — Investment   : VC funding, TechCrunch coverage
  4 — Big Tech     : Patent filings, enterprise products
  5 — Mainstream   : Wikipedia article, Google Trends presence
"""

import json
import logging
import sys
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

import anthropic
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

from agents.researcher import ResearcherAgent
from lineage.tracker import LineageTracker
from database.sqlite_client import SQLiteClient

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("agents.analyst")


def _anthropic_failure_message(exc: anthropic.APIStatusError) -> str:
    """User-facing message for Anthropic HTTP errors (billing, auth, model)."""
    body = getattr(exc, "body", None)
    detail = ""
    if isinstance(body, dict):
        err = body.get("error") or {}
        detail = (err.get("message") or "").strip()
    if not detail:
        detail = str(exc)
    low = detail.lower()
    if "credit balance" in low or "too low to access" in low:
        return (
            "Anthropic API: insufficient credits. Add credits under Plans & Billing: "
            "https://console.anthropic.com/settings/plans"
        )
    if exc.status_code == 401:
        return (
            "Anthropic API: unauthorized — check ANTHROPIC_KEY or ANTHROPIC_API_KEY in .env."
        )
    return f"Anthropic API error (HTTP {exc.status_code}): {detail}"


# ---------------------------------------------------------------------------
# Claude model
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "claude-sonnet-4-6"
_MAX_TOKENS = 2048


def _anthropic_model() -> str:
    """Override with ANTHROPIC_MODEL in .env (e.g. claude-3-5-sonnet-20241022)."""
    m = (os.getenv("ANTHROPIC_MODEL") or _DEFAULT_MODEL).strip()
    return m or _DEFAULT_MODEL

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a Technology Pipeline Analyst — an expert at assessing where a \
technology or business idea sits in its journey from academic research to \
mainstream adoption.

You have access to scored evidence across five pipeline stages:

  Stage 1 — Academic    : Peer-reviewed papers, citation counts
  Stage 2 — Developer   : GitHub repos, Product Hunt launches, YC companies
  Stage 3 — Investment  : VC funding news, TechCrunch articles
  Stage 4 — Big Tech    : Patent filings by large organisations
  Stage 5 — Mainstream  : Wikipedia page, Google Trends global presence

Your job is to:
  1. Confirm or refine the computed overall_stage based on the evidence.
  2. Write a concise (3–5 sentence) narrative explaining the stage assessment.
  3. Cite at least one specific piece of evidence per active stage \
     (e.g. paper title, company name, news headline).
  4. Flag any conflicting signals (e.g. mainstream trend but no patents).
  5. Predict the most likely next stage transition and what would trigger it.

IMPORTANT RULES:
  - Never hallucinate evidence. Only reference documents explicitly present \
    in the evidence JSON.
  - Every factual claim must be followed by a citation in brackets: \
    [Source: <name>, <date>]
  - If a stage score is 0, state that no signal was found for that stage.
  - Output valid JSON matching the schema below. No markdown fences.

Output schema:
{
  "confirmed_stage": <int 1-5>,
  "stage_label": "<Academic|Developer|Investment|BigTech|Mainstream>",
  "confidence": "<High|Medium|Low>",
  "narrative": "<3-5 sentence plain-English explanation with inline citations>",
  "evidence_by_stage": {
    "stage_1": ["<cited evidence string>", ...],
    "stage_2": [...],
    "stage_3": [...],
    "stage_4": [...],
    "stage_5": [...]
  },
  "conflicting_signals": ["<description>", ...],
  "next_stage_prediction": {
    "stage": <int>,
    "trigger": "<what would cause this transition>"
  },
  "sources_cited": [
    {"type": "<paper|company|article|patent|trend>", "name": "<title>", "date": "<ISO date or year>"}
  ]
}
"""

_STAGE_LABELS = {
    1: "Academic",
    2: "Developer",
    3: "Investment",
    4: "BigTech",
    5: "Mainstream",
}


def _build_evidence_prompt(research: dict) -> str:
    """
    Serialise the ResearchResult into a compact evidence block for the
    Claude prompt.  Keeps only the fields Claude needs to cite.
    """
    scores = research.get("scores", {})
    query = research.get("query", "")

    lines = [
        f'Query: "{query}"',
        "",
        "=== COMPUTED SCORES (0-100 each) ===",
        json.dumps(
            {k: v for k, v in scores.items() if k != "score_details"},
            indent=2,
        ),
        "",
        "=== RAG: TOP ACADEMIC PAPERS ===",
    ]

    for p in research.get("rag_papers", [])[:5]:
        title = p.get("title", "?")
        year = p.get("year", "?")
        score = p.get("similarity_score", 0)
        authors = p.get("authors", "")
        lines.append(f'  [{score:.0f}%] "{title}" ({year}) — {authors}')

    lines += ["", "=== RAG: TOP FUNDING/NEWS ARTICLES ==="]
    for a in research.get("rag_articles", [])[:5]:
        title = a.get("title", "?")
        date = a.get("date", "?")
        source = a.get("source", "?")
        score = a.get("similarity_score", 0)
        lines.append(f'  [{score:.0f}%] "{title}" — {source} ({date})')

    graph = research.get("graph", {})
    companies = graph.get("companies", [])
    papers = graph.get("papers", [])
    investors = graph.get("investors", [])

    if companies:
        lines += ["", "=== GRAPH: COMPANIES ==="]
        for c in companies[:10]:
            name = c.get("name", "?")
            stage = c.get("stage", "?")
            batch = c.get("batch", "")
            lines.append(f'  {name} (stage {stage}{"  batch: " + batch if batch else ""})')

    if investors:
        lines += ["", "=== GRAPH: INVESTORS ==="]
        for inv in investors[:5]:
            lines.append(f'  {inv.get("name", "?")} ({inv.get("type", "?")})')

    if papers:
        lines += ["", "=== GRAPH: LINKED PAPERS ==="]
        for pa in papers[:5]:
            lines.append(f'  "{pa.get("title", "?")}" ({pa.get("year", "?")})')

    related = research.get("related_technologies", [])
    if related:
        lines += ["", "=== GRAPH: RELATED TECHNOLOGIES ==="]
        for r in related[:5]:
            tech = r.get("related_technology", "?")
            strength = r.get("connection_strength", 0)
            lines.append(f'  {tech} (shared connections: {strength})')

    timeline = research.get("timeline", [])
    if timeline:
        lines += ["", "=== STAGE DETECTION TIMELINE ==="]
        for ev in timeline:
            stage_n = ev.get("stage", "?")
            label = _STAGE_LABELS.get(stage_n, str(stage_n))
            ts = ev.get("detected_at", "?")
            src = ev.get("source", "?")
            lines.append(f'  Stage {stage_n} ({label}): first detected {ts} via {src}')

    raw_summary = research.get("raw_summary", {})
    if raw_summary:
        lines += ["", "=== RAW FETCH SUMMARY ===", json.dumps(raw_summary, indent=2)]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# AnalystAgent
# ---------------------------------------------------------------------------


class AnalystAgent:
    """
    Reasoning agent — calls the Researcher, interprets evidence with Claude,
    and returns a structured stage assessment.
    """

    def __init__(self) -> None:
        api_key = os.getenv("ANTHROPIC_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_KEY not set in .env — required for the Analyst agent."
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = _anthropic_model()
        self._researcher = ResearcherAgent()
        self._lineage = LineageTracker()
        self._sqlite = SQLiteClient()
        logger.info("AnalystAgent initialised (model=%s)", self._model)

    # ------------------------------------------------------------------
    # Primary entry points
    # ------------------------------------------------------------------

    def analyse(self, query: str) -> dict:
        """
        Full analysis: research + Claude reasoning.

        Returns:
            {
                "query": str,
                "research": dict,          # ResearchResult
                "assessment": dict,        # Claude JSON output
                "raw_response": str,       # full Claude text
                "prov": dict,
                "logged_at": str,
            }
        """
        logger.info("analyse: query=%r", query)
        started = datetime.now(timezone.utc).isoformat()

        # --- Step 1: gather evidence ----------------------------------------
        research = self._researcher.full_research(query)

        # --- Step 2: build prompt -------------------------------------------
        evidence_text = _build_evidence_prompt(research)
        user_message = (
            f"Please analyse the following evidence and produce your structured "
            f"JSON assessment.\n\n{evidence_text}"
        )

        # --- Step 3: call Claude --------------------------------------------
        logger.info("Calling Claude %s", self._model)
        self._log_tool_call("claude_api", {"model": self._model, "query": query})

        try:
            message = self._client.messages.create(
                model=self._model,
                max_tokens=_MAX_TOKENS,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
        except anthropic.APIStatusError as e:
            logger.error("Anthropic API rejected request: %s", e)
            raise RuntimeError(_anthropic_failure_message(e)) from e

        raw_text = message.content[0].text if message.content else ""
        self._log_tool_response("claude_api", raw_text[:500])

        # --- Step 4: parse JSON --------------------------------------------
        assessment = self._parse_assessment(raw_text, research)

        # --- Step 5: lineage -----------------------------------------------
        ended = datetime.now(timezone.utc).isoformat()
        prov = self._lineage.record_query(
            agent="analyst",
            query=query,
            response_id=f"analysis:{query}:{ended}",
            used_entities=[p["prov:entity"] for p in research.get("prov", []) if p],
            attributes={
                "confirmed_stage": assessment.get("confirmed_stage"),
                "confidence": assessment.get("confidence"),
                "model": self._model,
                "started_at": started,
                "ended_at": ended,
            },
        )

        # --- Step 6: log query to SQLite -----------------------------------
        self._sqlite.log_query(
            query=query,
            response=raw_text,
            sources=[s.get("name", "") for s in assessment.get("sources_cited", [])],
        )

        return {
            "query": query,
            "research": research,
            "assessment": assessment,
            "raw_response": raw_text,
            "prov": prov,
            "logged_at": ended,
        }

    def analyse_stream(self, query: str) -> Iterator[str]:
        """
        Streaming version of analyse().  Yields SSE-formatted strings:
            "data: <json_chunk>\n\n"

        The final event includes the complete assessment JSON.

        Usage (FastAPI):
            return StreamingResponse(analyst.analyse_stream(query),
                                     media_type="text/event-stream")
        """
        logger.info("analyse_stream: query=%r", query)
        started = datetime.now(timezone.utc).isoformat()

        # --- Step 1: research (blocking — pipeline must complete first) -----
        yield _sse({"event": "status", "message": "Fetching data sources..."})
        research = self._researcher.full_research(query)
        scores = research.get("scores", {})
        yield _sse({
            "event": "scores",
            "scores": scores,
            "message": f"Pipeline complete. Computed stage: {scores.get('overall_stage')}",
        })

        # --- Step 2: stream Claude response ---------------------------------
        evidence_text = _build_evidence_prompt(research)
        user_message = (
            f"Please analyse the following evidence and produce your structured "
            f"JSON assessment.\n\n{evidence_text}"
        )

        yield _sse({"event": "status", "message": "Claude reasoning..."})
        self._log_tool_call("claude_api_stream", {"model": self._model, "query": query})

        full_text = []
        try:
            with self._client.messages.stream(
                model=self._model,
                max_tokens=_MAX_TOKENS,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            ) as stream:
                for text in stream.text_stream:
                    full_text.append(text)
                    yield _sse({"event": "token", "token": text})
        except anthropic.APIStatusError as e:
            logger.error("Anthropic API rejected streaming request: %s", e)
            yield _sse({"event": "error", "message": _anthropic_failure_message(e)})
            return

        raw_text = "".join(full_text)
        self._log_tool_response("claude_api_stream", raw_text[:500])

        # --- Step 3: parse and emit final assessment ------------------------
        assessment = self._parse_assessment(raw_text, research)

        ended = datetime.now(timezone.utc).isoformat()
        prov = self._lineage.record_query(
            agent="analyst",
            query=query,
            response_id=f"analysis:{query}:{ended}",
            used_entities=[p["prov:entity"] for p in research.get("prov", []) if p],
            attributes={
                "confirmed_stage": assessment.get("confirmed_stage"),
                "confidence": assessment.get("confidence"),
                "model": self._model,
                "started_at": started,
                "ended_at": ended,
                "streamed": True,
            },
        )

        self._sqlite.log_query(
            query=query,
            response=raw_text,
            sources=[s.get("name", "") for s in assessment.get("sources_cited", [])],
        )

        yield _sse({
            "event": "complete",
            "query": query,
            "assessment": assessment,
            "prov": prov,
            "logged_at": ended,
        })

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_assessment(raw_text: str, research: dict) -> dict:
        """
        Attempt to parse Claude's JSON response.
        Falls back to a structured error dict with the computed scores if
        Claude's output is not valid JSON.
        """
        text = raw_text.strip()
        # Strip markdown code fences if Claude wrapped the JSON
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                ln for ln in lines if not ln.startswith("```")
            ).strip()

        try:
            assessment = json.loads(text)
            return assessment
        except json.JSONDecodeError:
            logger.warning("Claude response was not valid JSON — using fallback")
            scores = research.get("scores", {})
            stage = scores.get("overall_stage", 1)
            return {
                "confirmed_stage": stage,
                "stage_label": _STAGE_LABELS.get(stage, "Unknown"),
                "confidence": "Low",
                "narrative": raw_text[:500],
                "evidence_by_stage": {},
                "conflicting_signals": ["Claude response was not valid JSON."],
                "next_stage_prediction": {"stage": min(stage + 1, 5), "trigger": "Unknown"},
                "sources_cited": [],
                "_parse_error": True,
            }

    def _log_tool_call(self, tool: str, params: dict) -> None:
        """Structured log for observability / hallucination auditing."""
        logger.info(
            "TOOL_CALL tool=%s params=%s",
            tool,
            json.dumps(params, default=str)[:200],
        )

    def _log_tool_response(self, tool: str, response_preview: str) -> None:
        logger.info("TOOL_RESP tool=%s preview=%r", tool, response_preview)


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------


def _sse(payload: dict) -> str:
    """Format a dict as an SSE data event."""
    return f"data: {json.dumps(payload, default=str)}\n\n"


# ---------------------------------------------------------------------------
# Module-level convenience instance
# ---------------------------------------------------------------------------

analyst = AnalystAgent()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "large language models"
    print(f"\nAnalysing: {query!r}\n{'='*60}")

    agent = AnalystAgent()
    result = agent.analyse(query)

    assessment = result.get("assessment", {})
    print(f"Stage:      {assessment.get('confirmed_stage')} — {assessment.get('stage_label')}")
    print(f"Confidence: {assessment.get('confidence')}")
    print(f"\nNarrative:\n{assessment.get('narrative')}")

    sources = assessment.get("sources_cited", [])
    if sources:
        print(f"\nSources cited ({len(sources)}):")
        for s in sources:
            print(f"  [{s.get('type')}] {s.get('name')} ({s.get('date')})")
