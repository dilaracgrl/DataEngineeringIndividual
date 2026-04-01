"""
Scheduled Monitoring System — Technology Pipeline Watchlist
============================================================

Runs the full data pipeline for every technology in WATCHLIST, compares
the result to the previous run stored in SQLite, and fires alerts when
something meaningful changes.

Two alert types
───────────────
  STAGE_TRANSITION   — overall_stage changed between runs
                       (e.g. Stage 3 → 4: Big Tech entering)
  SIGNIFICANT_CHANGE — overall_score moved by more than SCORE_DELTA_THRESHOLD
                       (default: 15 points) without a full stage change

Outputs
───────
  1. Console summary — colour-coded, one line per technology
  2. reports/monitor_{timestamp}.json — full machine-readable report
  3. database/pipeline_tracker.db    — alerts persisted to monitoring_alerts

Scheduling
──────────
Run manually:
    python scripts/monitor.py

Run on a cron schedule (daily at 07:00):
    0 7 * * * cd /path/to/project && python scripts/monitor.py >> logs/monitor.log 2>&1

Run from the project CLI (after wiring into main.py):
    python main.py monitor
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── Path setup — allow running as `python scripts/monitor.py` from anywhere ──
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

from pipeline.embedder    import DataEmbedder
from database.sqlite_client import SQLiteClient

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.WARNING,          # keep pipeline chatter quiet in monitor runs
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("monitor")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WATCHLIST: list[str] = [
    "large language models",
    "quantum computing",
    "brain computer interface",
    "spatial computing",
    "neuromorphic computing",
]

# An alert fires when |new_score − previous_score| exceeds this threshold.
SCORE_DELTA_THRESHOLD: float = 15.0

# Where to write JSON reports.
REPORTS_DIR = _ROOT / "reports"

# ---------------------------------------------------------------------------
# Stage labels
# ---------------------------------------------------------------------------

_STAGE_LABELS: dict[int, str] = {
    1: "Academic",
    2: "Developer",
    3: "Investment",
    4: "BigTech",
    5: "Mainstream",
}


def _stage_label(stage: int | None) -> str:
    if stage is None:
        return "?"
    return _STAGE_LABELS.get(stage, str(stage))


# ---------------------------------------------------------------------------
# Console colours (ANSI, stripped on non-TTY)
# ---------------------------------------------------------------------------

def _supports_colour() -> bool:
    return sys.stdout.isatty()


_RESET  = "\033[0m"  if _supports_colour() else ""
_BOLD   = "\033[1m"  if _supports_colour() else ""
_DIM    = "\033[2m"  if _supports_colour() else ""
_RED    = "\033[91m" if _supports_colour() else ""
_YELLOW = "\033[93m" if _supports_colour() else ""
_GREEN  = "\033[92m" if _supports_colour() else ""
_CYAN   = "\033[96m" if _supports_colour() else ""
_WHITE  = "\033[97m" if _supports_colour() else ""


# ---------------------------------------------------------------------------
# Core monitoring logic
# ---------------------------------------------------------------------------

class TechResult:
    """Holds the outcome of monitoring a single technology."""

    __slots__ = (
        "technology", "current_stage", "current_score",
        "previous_stage", "previous_score",
        "stage_changed", "score_delta",
        "alerts", "error",
    )

    def __init__(self, technology: str) -> None:
        self.technology:     str            = technology
        self.current_stage:  Optional[int]  = None
        self.current_score:  Optional[float]= None
        self.previous_stage: Optional[int]  = None
        self.previous_score: Optional[float]= None
        self.stage_changed:  bool           = False
        self.score_delta:    float          = 0.0
        self.alerts:         list[str]      = []   # alert_type strings
        self.error:          Optional[str]  = None


def _run_one(
    technology: str,
    embedder: DataEmbedder,
    sqlite: SQLiteClient,
    index: int,
    total: int,
) -> TechResult:
    """
    Runs the full pipeline for a single technology, compares to the
    previous run, fires alerts, and returns a TechResult.
    """
    result = TechResult(technology)
    prefix = f"[{index}/{total}]"

    print(f"{_DIM}{prefix}{_RESET} {_WHITE}{technology}{_RESET} — running pipeline…",
          flush=True)

    # ── 1. Run the full pipeline ──────────────────────────────────────────────
    try:
        pipeline_result = embedder.run_full_pipeline(technology)
    except Exception as exc:
        result.error = str(exc)
        logger.error("Pipeline failed for %r: %s", technology, exc, exc_info=True)
        print(f"       {_RED}ERROR: {exc}{_RESET}", flush=True)
        return result

    scores = pipeline_result.get("scores", {})
    result.current_stage = int(scores.get("overall_stage", 1))
    result.current_score = float(scores.get("overall_score", 0.0))

    # ── 2. Retrieve the two most recent runs from SQLite ──────────────────────
    # run_full_pipeline already called save_trend_score, so history[0] is the
    # row we just wrote and history[1] (if present) is the previous run.
    try:
        history = sqlite.get_trend_history(technology, limit=2)
    except Exception as exc:
        logger.warning("Could not read trend history for %r: %s", technology, exc)
        history = []

    if len(history) >= 2:
        prev_row            = history[1]           # second-newest = previous run
        result.previous_stage = int(prev_row.get("overall_stage", result.current_stage))
        result.previous_score = float(prev_row.get("overall_score", result.current_score))
        result.score_delta    = result.current_score - result.previous_score
        result.stage_changed  = result.current_stage != result.previous_stage
    else:
        # First ever run — no comparison available
        result.previous_stage = result.current_stage
        result.previous_score = result.current_score

    # ── 3. Fire alerts ────────────────────────────────────────────────────────
    if result.stage_changed:
        msg = (
            f"{technology}: Stage {result.previous_stage} "
            f"({_stage_label(result.previous_stage)}) → "
            f"{result.current_stage} ({_stage_label(result.current_stage)})"
        )
        try:
            sqlite.save_alert(
                technology = technology,
                alert_type = "STAGE_TRANSITION",
                prev_stage = result.previous_stage,
                new_stage  = result.current_stage,
                prev_score = result.previous_score,
                new_score  = result.current_score,
                message    = msg,
            )
        except Exception as exc:
            logger.warning("Could not save STAGE_TRANSITION alert: %s", exc)
        result.alerts.append("STAGE_TRANSITION")

    if abs(result.score_delta) > SCORE_DELTA_THRESHOLD:
        direction = "+" if result.score_delta >= 0 else ""
        msg = (
            f"{technology}: score moved {direction}{result.score_delta:.1f} pts "
            f"({result.previous_score:.1f} → {result.current_score:.1f})"
        )
        try:
            sqlite.save_alert(
                technology = technology,
                alert_type = "SIGNIFICANT_CHANGE",
                prev_stage = result.previous_stage if result.stage_changed else None,
                new_stage  = result.current_stage  if result.stage_changed else None,
                prev_score = result.previous_score,
                new_score  = result.current_score,
                message    = msg,
            )
        except Exception as exc:
            logger.warning("Could not save SIGNIFICANT_CHANGE alert: %s", exc)
        result.alerts.append("SIGNIFICANT_CHANGE")

    return result


# ---------------------------------------------------------------------------
# Console output helpers
# ---------------------------------------------------------------------------

def _format_result_line(r: TechResult) -> str:
    """Returns a single formatted console line for one technology."""
    name = f"{_WHITE}{r.technology:<30}{_RESET}"

    if r.error:
        return f"  {name}  {_RED}ERROR — {r.error}{_RESET}"

    # Stage display
    if r.stage_changed:
        stage_str = (
            f"Stage {r.previous_stage} ({_stage_label(r.previous_stage)}) "
            f"{_YELLOW}→{_RESET} "
            f"{_CYAN}{r.current_stage} ({_stage_label(r.current_stage)}){_RESET}"
        )
    else:
        stage_str = (
            f"Stage {_CYAN}{r.current_stage} "
            f"({_stage_label(r.current_stage)}){_RESET}"
        )

    # Score display
    if r.previous_score is not None and r.previous_score != r.current_score:
        sign  = "+" if r.score_delta >= 0 else ""
        delta_colour = _GREEN if r.score_delta >= 0 else _RED
        score_str = (
            f"Score {r.current_score:.1f}  "
            f"({delta_colour}{sign}{r.score_delta:.1f}{_RESET})"
        )
    else:
        score_str = f"Score {r.current_score:.1f}"

    # Alert badges
    badges = ""
    if "STAGE_TRANSITION" in r.alerts:
        badges += f"  {_YELLOW}{_BOLD}⚠  STAGE TRANSITION{_RESET}"
    if "SIGNIFICANT_CHANGE" in r.alerts:
        badges += f"  {_YELLOW}▲  SIGNIFICANT CHANGE{_RESET}"
    if not r.alerts and r.previous_score is not None:
        badges = f"  {_DIM}(no change){_RESET}"

    return f"  {name}  {stage_str}  |  {score_str}{badges}"


# ---------------------------------------------------------------------------
# Report serialisation
# ---------------------------------------------------------------------------

def _build_report(
    run_ts: str,
    results: list[TechResult],
) -> dict:
    """Serialises all results into the JSON report structure."""
    alerts_fired    = sum(len(r.alerts) for r in results)
    stage_changes   = sum(1 for r in results if r.stage_changed)
    errors          = sum(1 for r in results if r.error)

    tech_records = []
    for r in results:
        tech_records.append({
            "technology":     r.technology,
            "current_stage":  r.current_stage,
            "current_score":  r.current_score,
            "previous_stage": r.previous_stage,
            "previous_score": r.previous_score,
            "stage_changed":  r.stage_changed,
            "score_delta":    round(r.score_delta, 2),
            "alerts":         r.alerts,
            "error":          r.error,
        })

    return {
        "timestamp":            run_ts,
        "technologies_checked": len(results),
        "alerts_fired":         alerts_fired,
        "stage_changes":        stage_changes,
        "errors":               errors,
        "score_delta_threshold": SCORE_DELTA_THRESHOLD,
        "technologies":         tech_records,
        "summary": (
            f"{len(results)} technologies checked — "
            f"{stage_changes} stage transition(s), "
            f"{alerts_fired} total alert(s)"
            + (f", {errors} error(s)" if errors else "")
        ),
    }


def _save_report(report: dict, run_ts: str) -> Path:
    """Writes the report JSON to reports/monitor_{timestamp}.json."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    # Filesystem-safe timestamp: 2024-03-15T14:30:00+00:00 → 20240315_143000
    safe_ts = run_ts[:19].replace("-", "").replace(":", "").replace("T", "_")
    path = REPORTS_DIR / f"monitor_{safe_ts}.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_monitor(
    watchlist: list[str] | None = None,
    *,
    score_delta_threshold: float | None = None,
) -> dict:
    """
    Runs the full monitoring cycle for every technology in watchlist.

    Returns the report dict (also saved to disk and summarised to console).
    Can be imported and called programmatically, or run directly.

    Args:
        watchlist: Defaults to :data:`WATCHLIST` when omitted.
        score_delta_threshold: Overrides :data:`SCORE_DELTA_THRESHOLD` for this run.
    """
    global SCORE_DELTA_THRESHOLD
    if score_delta_threshold is not None:
        SCORE_DELTA_THRESHOLD = score_delta_threshold

    wl = watchlist if watchlist is not None else WATCHLIST
    run_ts = datetime.now(timezone.utc).isoformat()
    total  = len(wl)

    print()
    print(f"{_BOLD}{'=' * 52}{_RESET}")
    print(f"{_BOLD}  MONITORING REPORT{_RESET}  "
          f"{_DIM}{run_ts[:19]} UTC{_RESET}")
    print(f"{_BOLD}{'=' * 52}{_RESET}")
    print(f"  Watchlist: {total} technologies\n")

    embedder = DataEmbedder()
    sqlite   = SQLiteClient()
    results: list[TechResult] = []

    for i, tech in enumerate(wl, start=1):
        r = _run_one(
            technology = tech,
            embedder   = embedder,
            sqlite     = sqlite,
            index      = i,
            total      = total,
        )
        results.append(r)
        print(_format_result_line(r), flush=True)
        print()

    # ── Summary section ───────────────────────────────────────────────────────
    alerts_fired  = sum(len(r.alerts) for r in results)
    stage_changes = [r for r in results if r.stage_changed]
    errors        = [r for r in results if r.error]

    print(f"{_BOLD}{'─' * 52}{_RESET}")
    print(f"  Technologies checked : {_WHITE}{total}{_RESET}")
    print(f"  Stage transitions    : "
          + (f"{_YELLOW}{_BOLD}{len(stage_changes)}{_RESET}" if stage_changes
             else f"{_DIM}0{_RESET}"))
    print(f"  Total alerts fired   : "
          + (f"{_YELLOW}{alerts_fired}{_RESET}" if alerts_fired
             else f"{_DIM}0{_RESET}"))
    if errors:
        print(f"  Errors               : {_RED}{len(errors)}{_RESET}")

    # ── Stage transition detail ───────────────────────────────────────────────
    if stage_changes:
        print()
        print(f"  {_BOLD}Stage transitions:{_RESET}")
        for r in stage_changes:
            print(
                f"    {_CYAN}{r.technology}{_RESET}  "
                f"Stage {r.previous_stage} ({_stage_label(r.previous_stage)}) "
                f"{_YELLOW}→{_RESET} "
                f"{r.current_stage} ({_stage_label(r.current_stage)})"
            )

    # ── Save report ───────────────────────────────────────────────────────────
    report    = _build_report(run_ts, results)
    report_path = _save_report(report, run_ts)

    print()
    print(f"  Report saved to: {_DIM}{report_path.relative_to(_ROOT)}{_RESET}")
    print(f"{_BOLD}{'=' * 52}{_RESET}")
    print()

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Technology Pipeline Watchlist Monitor",
    )
    parser.add_argument(
        "--watchlist",
        nargs="+",
        metavar="TECH",
        default=None,
        help=(
            "Override the default watchlist. "
            "Use quotes for multi-word names: "
            '--watchlist "quantum computing" "diffusion models"'
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=SCORE_DELTA_THRESHOLD,
        help=f"Score-change threshold for SIGNIFICANT_CHANGE alerts "
             f"(default: {SCORE_DELTA_THRESHOLD})",
    )
    args = parser.parse_args()

    watchlist_to_use = args.watchlist if args.watchlist else WATCHLIST
    run_monitor(
        watchlist_to_use,
        score_delta_threshold=args.threshold,
    )
