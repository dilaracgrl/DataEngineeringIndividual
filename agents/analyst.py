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
import queue
import sys
import os
import threading
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
_MAX_TOKENS = 4096


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
  2. Write a concise (3–5 sentence) narrative explaining the stage assessment. \
     Incorporate the velocity data — e.g. "this technology is Stage 3 and \
     accelerating rapidly — estimated Stage 4 within 12–18 months."
  3. Cite at least one specific piece of evidence per active stage \
     (e.g. paper title, company name, news headline).
  4. Flag any conflicting signals (e.g. mainstream trend but no patents).
  5. Predict the most likely next stage transition and what would trigger it, \
     using the estimated_next_stage_months from VELOCITY ANALYSIS when available.

IMPORTANT RULES:
  - Never hallucinate evidence. Only reference documents explicitly present \
    in the evidence JSON.
  - Every factual claim must be followed by a citation in brackets: \
    [Source: <name>, <date>]
  - If a stage score is 0, state that no signal was found for that stage.
  - Use velocity data to add momentum language: "accelerating", "stalling", \
    "cooling off", "gaining rapidly", etc.
  - Output valid JSON matching the schema below. No markdown fences.

Output schema:
{
  "confirmed_stage": <int 1-5>,
  "stage_label": "<Academic|Developer|Investment|BigTech|Mainstream>",
  "confidence": "<High|Medium|Low>",
  "narrative": "<3-5 sentence plain-English explanation with inline citations and velocity language>",
  "velocity_assessment": {
    "direction": "<accelerating|stable|decelerating>",
    "summary": "<one sentence on momentum>",
    "estimated_next_stage_months": <int or null>
  },
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
    "trigger": "<what would cause this transition>",
    "estimated_months": <int or null>
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

# ---------------------------------------------------------------------------
# Self-critique prompt and schema
# ---------------------------------------------------------------------------

_CRITIQUE_SYSTEM_PROMPT = """\
You are a Critical Quality Reviewer for a Technology Pipeline Analyst system. \
Your sole job is to evaluate the accuracy and evidence coverage of a stage \
assessment that another model produced.

You will receive:
  EVIDENCE BLOCK — the raw data that was available when the assessment was made.
  INITIAL ASSESSMENT — the JSON conclusion the analyst reached.

Your task is to scrutinise the assessment against the evidence and produce a \
structured critique.  Be strict: only count a claim as supported if the \
supporting item is explicitly named in the EVIDENCE BLOCK.

Definitions
───────────
  Supporting evidence   : Items from the EVIDENCE BLOCK that directly confirm \
the stage conclusion or a specific claim in the narrative.
  Contradicting evidence: Items from the EVIDENCE BLOCK that weaken, \
contradict, or are inconsistent with the stage conclusion.
  Unsupported claims    : Specific statements in the narrative that cannot be \
traced to any item in the EVIDENCE BLOCK — potential hallucinations.

Confidence score (0–100)
────────────────────────
  90–100  All key claims backed by named evidence; stage conclusion strongly supported.
  70–89   Good coverage; minor details inferred but core conclusion is solid.
  50–69   Partial evidence; plausible but meaningful gaps remain.
  30–49   Thin evidence; conclusion is largely inferential.
   0–29   Very little evidence; conclusion is mostly speculative.

Reliability
───────────
  "High"   — confidence_score ≥ 70
  "Medium" — confidence_score 40–69
  "Low"    — confidence_score < 40

RULES:
  - Reference only items explicitly present in the EVIDENCE BLOCK.
  - Quote or paraphrase the specific claim when listing unsupported_claims.
  - Output valid JSON only.  No markdown fences, no commentary outside the JSON.

Output schema:
{
  "confidence_score": <int 0-100>,
  "reliability": "<High|Medium|Low>",
  "supporting_evidence": ["<item from evidence block>", ...],
  "contradicting_evidence": ["<item that weakens the conclusion>", ...],
  "unsupported_claims": ["<verbatim or paraphrased claim with no evidence>", ...]
}
"""

_CRITIQUE_MAX_TOKENS = 1024


def _merge_assessment_and_critique(assessment: dict, critique: dict) -> dict:
    """
    Combines the initial assessment (first Claude call) with the critique
    (second Claude call) into the final output dict.

    The first call's qualitative confidence label ("High"/"Medium"/"Low") is
    replaced by the critique's numeric score (0–100) and reliability rating
    so callers get a single, auditable confidence value.

    All original assessment fields are preserved — the critique adds to them
    rather than replacing them.
    """
    return {
        # Core identification — from first call
        "stage":                  assessment.get("confirmed_stage"),
        "stage_label":            assessment.get("stage_label", ""),
        # Evidence quality — from second (critique) call
        "confidence":             critique.get("confidence_score", 50),
        "reliability":            critique.get("reliability", "Medium"),
        # Narrative — from first call (with citations)
        "narrative":              assessment.get("narrative", ""),
        # Critique outputs
        "supporting_evidence":    critique.get("supporting_evidence", []),
        "contradicting_evidence": critique.get("contradicting_evidence", []),
        "unsupported_claims":     critique.get("unsupported_claims", []),
        # Velocity — from first call
        "velocity_assessment":    assessment.get("velocity_assessment", {}),
        # Full evidence breakdown — from first call
        "evidence_by_stage":      assessment.get("evidence_by_stage", {}),
        "conflicting_signals":    assessment.get("conflicting_signals", []),
        "next_stage_prediction":  assessment.get("next_stage_prediction", {}),
        "sources_cited":          assessment.get("sources_cited", []),
        # Raw outputs preserved for auditing
        "_raw_assessment":        assessment,
        "_raw_critique":          critique,
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

    velocity = research.get("velocity", {})
    if velocity:
        lines += ["", "=== VELOCITY ANALYSIS ==="]
        lines.append(f'  Summary       : {velocity.get("velocity_summary", "N/A")}')
        lines.append(f'  Overall       : {velocity.get("overall_velocity", "N/A")} '
                     f'({velocity.get("score_growth_per_month", 0):+.1f} pts/month)')
        lines.append(f'  Academic      : {velocity.get("academic_velocity", "N/A")} '
                     f'(YoY growth: {velocity.get("academic_growth_rate", 0):+.1f}%)')
        lines.append(f'  Startup       : {velocity.get("startup_velocity", "N/A")}')
        lines.append(f'  News/Press    : {velocity.get("news_velocity", "N/A")}')
        next_months = velocity.get("estimated_next_stage_months")
        if next_months is not None:
            lines.append(f'  Next stage in : ~{next_months} months (data-driven estimate)')
        else:
            lines.append(  '  Next stage in : insufficient history to estimate')

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
        Full analysis: research → stage assessment → self-critique → merge.

        Two Claude calls are made:
          Call 1 — stage assessment (full evidence prompt → JSON assessment)
          Call 2 — self-critique   (assessment + evidence → confidence score,
                                    supporting/contradicting evidence,
                                    unsupported claims)

        Both calls are logged to SQLite query_history for auditing.

        Returns:
            {
                "query": str,
                "research": dict,       # ResearchResult
                "assessment": dict,     # merged final output (see _merge_assessment_and_critique)
                "velocity": dict,
                "raw_response": str,    # Call 1 raw text
                "raw_critique": str,    # Call 2 raw text
                "prov": dict,
                "logged_at": str,
            }
        """
        logger.info("analyse: query=%r", query)
        started = datetime.now(timezone.utc).isoformat()

        # --- Step 1: gather evidence ----------------------------------------
        research = self._researcher.full_research(query)
        evidence_text = _build_evidence_prompt(research)

        # --- Step 2: Call 1 — stage assessment ------------------------------
        user_message = (
            f"Please analyse the following evidence and produce your structured "
            f"JSON assessment.\n\n{evidence_text}"
        )
        logger.info("Call 1 — stage assessment (model=%s)", self._model)
        self._log_tool_call("claude_api:assessment", {"model": self._model, "query": query})

        try:
            message = self._client.messages.create(
                model=self._model,
                max_tokens=_MAX_TOKENS,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
        except anthropic.APIStatusError as e:
            logger.error("Anthropic API rejected assessment request: %s", e)
            raise RuntimeError(_anthropic_failure_message(e)) from e

        raw_assessment = message.content[0].text if message.content else ""
        self._log_tool_response("claude_api:assessment", raw_assessment[:500])
        assessment = self._parse_assessment(raw_assessment, research)

        # Log Call 1 to SQLite immediately for audit trail
        self._sqlite.log_query(
            query=query,
            response=raw_assessment,
            sources=[s.get("name", "") for s in assessment.get("sources_cited", [])],
        )

        # --- Step 3: Call 2 — self-critique ---------------------------------
        logger.info("Call 2 — self-critique (model=%s)", self._model)
        raw_critique, critique = self._run_critique(
            assessment=assessment,
            evidence_text=evidence_text,
            query=query,
        )

        # Log Call 2 to SQLite — tagged with "[critique]" for easy filtering
        self._sqlite.log_query(
            query=f"{query} [critique]",
            response=raw_critique,
            sources=[],
        )

        # --- Step 4: merge ---------------------------------------------------
        final = _merge_assessment_and_critique(assessment, critique)

        # --- Step 5: lineage -------------------------------------------------
        ended = datetime.now(timezone.utc).isoformat()
        prov = self._lineage.record_query(
            agent="analyst",
            query=query,
            response_id=f"analysis:{query}:{ended}",
            used_entities=[p["prov:entity"] for p in research.get("prov", []) if p],
            attributes={
                "confirmed_stage":  final.get("stage"),
                "confidence_score": final.get("confidence"),
                "reliability":      final.get("reliability"),
                "model":            self._model,
                "started_at":       started,
                "ended_at":         ended,
            },
        )

        return {
            "query":        query,
            "research":     research,
            "assessment":   final,
            "velocity":     research.get("velocity", {}),
            "raw_response": raw_assessment,
            "raw_critique": raw_critique,
            "prov":         prov,
            "logged_at":    ended,
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

        # --- Step 1: research with live progress events ----------------------
        # Run full_research() in a background thread so the generator can yield
        # progress SSE events as each pipeline stage completes.
        progress_q: queue.SimpleQueue = queue.SimpleQueue()
        result_box: dict = {}
        error_box:  dict = {}

        def _research_worker() -> None:
            try:
                result_box["data"] = self._researcher.full_research(
                    query,
                    progress_callback=lambda msg: progress_q.put(msg),
                )
            except Exception as exc:
                error_box["err"] = exc
            finally:
                progress_q.put(None)  # sentinel — drain complete

        t = threading.Thread(target=_research_worker, daemon=True)
        t.start()

        # Yield progress events until the sentinel arrives
        while True:
            msg = progress_q.get()
            if msg is None:
                break
            yield _sse({"event": "status", "message": msg})

        t.join()

        if "err" in error_box:
            raise error_box["err"]

        research = result_box["data"]
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

        yield _sse({"event": "status", "message": "Running analyst agent..."})
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

        # --- Step 3: parse Call 1 and log to SQLite -------------------------
        assessment = self._parse_assessment(raw_text, research)
        self._sqlite.log_query(
            query=query,
            response=raw_text,
            sources=[s.get("name", "") for s in assessment.get("sources_cited", [])],
        )

        # --- Step 4: self-critique (blocking) --------------------------------
        yield _sse({"event": "status", "message": "Running self-critique..."})
        raw_critique, critique = self._run_critique(
            assessment=assessment,
            evidence_text=evidence_text,
            query=query,
        )
        self._sqlite.log_query(
            query=f"{query} [critique]",
            response=raw_critique,
            sources=[],
        )

        # --- Step 5: merge and emit final event ------------------------------
        final = _merge_assessment_and_critique(assessment, critique)

        ended = datetime.now(timezone.utc).isoformat()
        prov = self._lineage.record_query(
            agent="analyst",
            query=query,
            response_id=f"analysis:{query}:{ended}",
            used_entities=[p["prov:entity"] for p in research.get("prov", []) if p],
            attributes={
                "confirmed_stage":  final.get("stage"),
                "confidence_score": final.get("confidence"),
                "reliability":      final.get("reliability"),
                "model":            self._model,
                "started_at":       started,
                "ended_at":         ended,
                "streamed":         True,
            },
        )

        yield _sse({
            "event": "complete",
            "query": query,
            "assessment": final,
            "velocity": research.get("velocity", {}),
            "raw_critique": raw_critique,
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

        # First attempt: direct parse
        try:
            assessment = json.loads(text)
            return assessment
        except json.JSONDecodeError:
            pass

        # Second attempt: extract the outermost {...} block in case Claude
        # prefixed or suffixed the JSON with prose despite the prompt instruction
        start = text.find("{")
        end   = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                assessment = json.loads(text[start:end + 1])
                logger.info("_parse_assessment: extracted JSON from mixed-text response")
                return assessment
            except json.JSONDecodeError:
                pass

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

    def _run_critique(
        self,
        assessment: dict,
        evidence_text: str,
        query: str,
    ) -> tuple[str, dict]:
        """
        Call 2 — self-critique.

        Sends the initial assessment JSON alongside the original evidence block
        to Claude using _CRITIQUE_SYSTEM_PROMPT.  Returns (raw_text, parsed_dict).
        Falls back to a safe default critique dict on any error so the overall
        analyse() call is never blocked by the critique step.
        """
        self._log_tool_call("claude_api:critique", {"model": self._model, "query": query})

        user_message = (
            "EVIDENCE BLOCK:\n"
            f"{evidence_text}\n\n"
            "INITIAL ASSESSMENT:\n"
            f"{json.dumps(assessment, indent=2, default=str)}\n\n"
            "Please produce your structured critique JSON."
        )

        try:
            message = self._client.messages.create(
                model=self._model,
                max_tokens=_CRITIQUE_MAX_TOKENS,
                system=_CRITIQUE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
        except anthropic.APIStatusError as e:
            logger.warning("Self-critique Call 2 failed (API error): %s", e)
            raw = ""
            return raw, self._fallback_critique()
        except Exception as e:
            logger.warning("Self-critique Call 2 failed (unexpected): %s", e)
            return "", self._fallback_critique()

        raw = message.content[0].text if message.content else ""
        self._log_tool_response("claude_api:critique", raw[:500])
        return raw, self._parse_critique(raw)

    @staticmethod
    def _parse_critique(raw_text: str) -> dict:
        """
        Parse the critique JSON response.
        Falls back to a safe default dict if the response is not valid JSON.
        """
        text = raw_text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(ln for ln in lines if not ln.startswith("```")).strip()

        def _apply_defaults(c: dict) -> dict:
            c.setdefault("confidence_score", 50)
            c.setdefault("reliability", "Medium")
            c.setdefault("supporting_evidence", [])
            c.setdefault("contradicting_evidence", [])
            c.setdefault("unsupported_claims", [])
            return c

        try:
            return _apply_defaults(json.loads(text))
        except json.JSONDecodeError:
            pass

        # Fallback: extract outermost {...} block
        start = text.find("{")
        end   = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                logger.info("_parse_critique: extracted JSON from mixed-text response")
                return _apply_defaults(json.loads(text[start:end + 1]))
            except json.JSONDecodeError:
                pass

        logger.warning("Critique response was not valid JSON — using fallback")
        return {
            "confidence_score": 50,
            "reliability": "Medium",
            "supporting_evidence": [],
            "contradicting_evidence": [],
            "unsupported_claims": ["Critique parse failed — raw response not valid JSON."],
            "_parse_error": True,
        }

    @staticmethod
    def _fallback_critique() -> dict:
        """Safe default critique returned when Call 2 itself fails."""
        return {
            "confidence_score": 50,
            "reliability": "Medium",
            "supporting_evidence": [],
            "contradicting_evidence": [],
            "unsupported_claims": [],
            "_critique_skipped": True,
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
