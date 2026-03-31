"""
Technology Pipeline Tracker — main entry point

Modes
──────
  serve      Start the FastAPI agent service (default)
  analyse    Run a one-shot analysis for a query (CLI)
  pipeline   Run only the data pipeline (no Claude) for a query (CLI)
  lineage    Print the W3C PROV event log for a query (CLI)
  monitor    Run the scheduled watchlist monitor (pipeline + alerts)

Usage
──────
  # Start the API server (default port 8000)
  python main.py serve

  # One-shot analysis
  python main.py analyse "large language models"

  # Run pipeline only (faster, no Claude cost)
  python main.py pipeline "quantum computing"

  # Show lineage for a query
  python main.py lineage "transformer"

  # Watchlist monitor (same as scripts/monitor.py)
  python main.py monitor
  python main.py monitor --watchlist "quantum computing" --threshold 20

  # Show help
  python main.py --help
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path and env setup
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.validate_env import validate_environment

if not validate_environment():
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")

# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the FastAPI agent service."""
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn not installed. Run: pip install uvicorn")
        sys.exit(1)

    port = args.port
    print(f"Dashboard running at http://localhost:{port}")
    print(f"API docs at         http://localhost:{port}/docs")
    logger.info("Starting agent service on http://0.0.0.0:%d", port)

    uvicorn.run(
        "api.agent_service:app",
        host="0.0.0.0",
        port=port,
        reload=args.reload,
        log_level="info",
    )


def cmd_analyse(args: argparse.Namespace) -> None:
    """Run a full analysis for a query and print the assessment."""
    from agents.analyst import AnalystAgent

    print(f"\nAnalysing: {args.query!r}")
    print("=" * 70)

    agent = AnalystAgent()

    if args.stream:
        for chunk in agent.analyse_stream(args.query):
            # Strip "data: " prefix and parse
            line = chunk.strip()
            if line.startswith("data: "):
                payload = json.loads(line[6:])
                event = payload.get("event", "")
                if event == "status":
                    print(f"[{event}] {payload.get('message')}")
                elif event == "scores":
                    scores = payload.get("scores", {})
                    print(f"[scores] Stage {scores.get('overall_stage')} — "
                          f"Score {scores.get('overall_score'):.1f}/100")
                elif event == "token":
                    print(payload.get("token", ""), end="", flush=True)
                elif event == "complete":
                    print("\n")
                    _print_assessment(payload.get("assessment", {}))
                elif event == "error":
                    print(f"[ERROR] {payload.get('message')}")
    else:
        try:
            result = agent.analyse(args.query)
        except RuntimeError as e:
            print(f"\n{e}", file=sys.stderr)
            sys.exit(1)
        _print_assessment(result.get("assessment", {}))


def cmd_pipeline(args: argparse.Namespace) -> None:
    """Run the data pipeline only (no Claude)."""
    from pipeline.embedder import DataEmbedder

    print(f"\nRunning pipeline for: {args.query!r}")
    print("=" * 70)

    embedder = DataEmbedder()
    result = embedder.run_full_pipeline(args.query)

    scores = result.get("scores", {})
    print(f"\nOverall Stage : {scores.get('overall_stage')} — "
          f"{_stage_label(scores.get('overall_stage'))}")
    print(f"Overall Score : {scores.get('overall_score', 0):.1f}/100")
    print("\nPer-tool scores:")
    _score_keys = [
        "arxiv_score", "semantic_scholar_score", "github_score",
        "producthunt_score", "yc_score", "news_score", "techcrunch_score",
        "patents_score", "wikipedia_score", "trends_score",
    ]
    for k in _score_keys:
        val = scores.get(k, 0)
        bar = "█" * int(val // 5) + "░" * (20 - int(val // 5))
        print(f"  {k:<30} {bar} {val:.0f}")

    storage = result.get("storage", {})
    print(f"\nStorage summary:")
    for layer, info in storage.items():
        print(f"  {layer}: {info}")


def cmd_lineage(args: argparse.Namespace) -> None:
    """Print the PROV event log for a query."""
    from lineage.tracker import LineageTracker

    tracker = LineageTracker()
    events = tracker.get_lineage_for_query(args.query, limit=args.limit)

    print(f"\nLineage events for {args.query!r} ({len(events)} found):")
    print("=" * 70)
    for ev in events:
        print(
            f"  [{ev.get('recorded_at', '?')[:19]}] "
            f"{ev.get('activity_type', '?'):10} "
            f"{ev.get('agent_id', '?'):30} "
            f"→ {ev.get('entity_id', '?')}"
        )


def cmd_monitor(args: argparse.Namespace) -> None:
    """Run the technology watchlist monitor (see scripts/monitor.py)."""
    from scripts.monitor import WATCHLIST, run_monitor

    watchlist = args.watchlist if args.watchlist else WATCHLIST
    run_monitor(
        watchlist,
        score_delta_threshold=args.threshold,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stage_label(stage: int) -> str:
    return {1: "Academic", 2: "Developer", 3: "Investment",
            4: "BigTech", 5: "Mainstream"}.get(stage, "Unknown")


def _print_assessment(assessment: dict) -> None:
    stage = assessment.get("confirmed_stage", "?")
    label = assessment.get("stage_label", _stage_label(stage))
    confidence = assessment.get("confidence", "?")
    narrative = assessment.get("narrative", "")

    print(f"Stage      : {stage} — {label}")
    print(f"Confidence : {confidence}")
    print(f"\nNarrative:\n{narrative}")

    conflicts = assessment.get("conflicting_signals", [])
    if conflicts:
        print(f"\nConflicting signals:")
        for c in conflicts:
            print(f"  • {c}")

    next_s = assessment.get("next_stage_prediction", {})
    if next_s:
        print(f"\nNext stage prediction: Stage {next_s.get('stage')}")
        print(f"  Trigger: {next_s.get('trigger')}")

    sources = assessment.get("sources_cited", [])
    if sources:
        print(f"\nSources cited ({len(sources)}):")
        for s in sources:
            print(f"  [{s.get('type', '?')}] {s.get('name', '?')} ({s.get('date', '?')})")


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Technology Pipeline Tracker — CLI",
    )
    sub = parser.add_subparsers(dest="command")

    # serve
    p_serve = sub.add_parser("serve", help="Start the FastAPI agent service")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true", help="Enable hot reload (dev mode)")

    # analyse
    p_analyse = sub.add_parser("analyse", help="Run full analysis for a query")
    p_analyse.add_argument("query", help="Technology or concept to analyse")
    p_analyse.add_argument("--stream", action="store_true", help="Stream Claude tokens")

    # pipeline
    p_pipeline = sub.add_parser("pipeline", help="Run data pipeline only (no Claude)")
    p_pipeline.add_argument("query", help="Technology or concept to pipeline")

    # lineage
    p_lineage = sub.add_parser("lineage", help="Print PROV event log for a query")
    p_lineage.add_argument("query", help="Query to look up in lineage log")
    p_lineage.add_argument("--limit", type=int, default=50)

    # monitor
    p_monitor = sub.add_parser(
        "monitor",
        help="Run watchlist pipeline monitor (alerts + JSON report under reports/)",
    )
    p_monitor.add_argument(
        "--watchlist",
        nargs="+",
        metavar="TECH",
        default=None,
        help=(
            "Override default watchlist. Quote multi-word names: "
            '--watchlist "quantum computing" "diffusion models"'
        ),
    )
    p_monitor.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Score-change threshold for SIGNIFICANT_CHANGE alerts (default: 15.0)",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "serve" or args.command is None:
        # Default to serve if no command given
        if args.command is None:
            args.port = 8000
            args.reload = False
        cmd_serve(args)
    elif args.command == "analyse":
        cmd_analyse(args)
    elif args.command == "pipeline":
        cmd_pipeline(args)
    elif args.command == "lineage":
        cmd_lineage(args)
    elif args.command == "monitor":
        cmd_monitor(args)
    else:
        parser.print_help()
