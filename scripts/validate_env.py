"""
Validate required environment variables before the pipeline or API starts.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent

# Required keys: env name -> where to obtain it
REQUIRED: dict[str, str] = {
    "GITHUB_TOKEN": (
        "GitHub → Settings → Developer settings → Personal access tokens: "
        "https://github.com/settings/tokens"
    ),
    "NEWS_API_KEY": (
        "NewsAPI registration (free tier): https://newsapi.org/register"
    ),
    "NEO4J_URI": (
        "Local Docker: bolt://localhost:7687 — Neo4j Aura: neo4j+s://… in Aura console"
    ),
    "NEO4J_USER": (
        "Neo4j username (often `neo4j` locally; Aura shows the connection user)"
    ),
    "NEO4J_PASSWORD": (
        "Set when creating the DB (Docker default in compose) or Aura password"
    ),
    "MONGO_URI": (
        "Local: mongodb://localhost:27017 — Atlas: mongodb+srv://… in cluster connect"
    ),
}

# Optional: warn if unset
OPTIONAL: dict[str, str] = {
    "S2_API_KEY": (
        "Semantic Scholar API key for higher rate limits: "
        "https://www.semanticscholar.org/product/api"
    ),
}


def _is_set(name: str) -> bool:
    v = os.getenv(name)
    return v is not None and str(v).strip() != ""


def _ph_get(name: str) -> str:
    """Same typo tolerance as tools/producthunt_tool: PR0DUCTHUNT vs PRODUCTHUNT."""
    if _is_set(name):
        return os.getenv(name, "").strip()
    if name.startswith("PRODUCTHUNT"):
        alt = "PR0" + name[3:]
        if _is_set(alt):
            return os.getenv(alt, "").strip()
    return ""


def _producthunt_configured() -> bool:
    """Developer token OR OAuth `client_credentials` pair (see `tools/producthunt_tool.py`)."""
    if _ph_get("PRODUCTHUNT_ACCESS_TOKEN"):
        return True
    return bool(_ph_get("PRODUCTHUNT_CLIENT_ID")) and bool(_ph_get("PRODUCTHUNT_CLIENT_SECRET"))


def _anthropic_configured() -> bool:
    """``agents/analyst.py`` accepts either name."""
    return _is_set("ANTHROPIC_API_KEY") or _is_set("ANTHROPIC_KEY")


def _warn_if_local_mongo_unreachable() -> None:
    """Best-effort ping when MONGO_URI points at this machine."""
    uri = os.getenv("MONGO_URI", "").strip()
    if not uri or ("localhost" not in uri and "127.0.0.1" not in uri):
        return
    try:
        from pymongo import MongoClient

        client = MongoClient(uri, serverSelectionTimeoutMS=2500)
        client.admin.command("ping")
        client.close()
    except Exception:
        print(
            "WARNING: MongoDB did not respond at MONGO_URI. "
            "Raw document storage will fail until it is running. "
            "Example: docker compose up -d mongo",
            file=sys.stderr,
        )


def _warn_if_local_neo4j_unreachable() -> None:
    """Best-effort ping when NEO4J_URI points at this machine."""
    uri = os.getenv("NEO4J_URI", "").strip()
    if not uri or ("localhost" not in uri and "127.0.0.1" not in uri):
        return
    password = os.getenv("NEO4J_PASSWORD", "").strip()
    if not password:
        return
    user = os.getenv("NEO4J_USER", "neo4j").strip() or "neo4j"
    try:
        from neo4j import GraphDatabase

        drv = GraphDatabase.driver(uri, auth=(user, password))
        drv.verify_connectivity()
        drv.close()
    except Exception:
        print(
            "WARNING: Neo4j did not respond at NEO4J_URI. "
            "Graph storage will fail until it is running. "
            "Example: docker compose up -d neo4j",
            file=sys.stderr,
        )


def validate_environment() -> bool:
    """
    Load `.env` from the project root, check required and optional keys.
    Prints messages to stdout. Returns True if all required keys are set.
    """
    load_dotenv(_ROOT / ".env")

    missing: list[str] = []

    if not _producthunt_configured():
        missing.append("PRODUCTHUNT")
        print(
            "ERROR: Product Hunt is not configured.\n"
            "       Set PRODUCTHUNT_ACCESS_TOKEN (developer token) or both "
            "PRODUCTHUNT_CLIENT_ID and PRODUCTHUNT_CLIENT_SECRET in .env. "
            "https://www.producthunt.com/v2/oauth/applications\n",
            file=sys.stderr,
        )

    if not _anthropic_configured():
        missing.append("ANTHROPIC_API_KEY")
        print(
            "ERROR: Anthropic API key is missing or empty.\n"
            "       Set ANTHROPIC_API_KEY or ANTHROPIC_KEY in .env. "
            "Anthropic Console → API Keys: https://console.anthropic.com/\n",
            file=sys.stderr,
        )

    for key, hint in REQUIRED.items():
        if not _is_set(key):
            missing.append(key)
            print(
                f"ERROR: {key} is missing or empty.\n"
                f"       Set it in .env at the project root. {hint}\n",
                file=sys.stderr,
            )

    for key, hint in OPTIONAL.items():
        if not _is_set(key):
            print(
                f"WARNING: {key} is not set (optional). "
                f"Tools work without it but rate limits are lower. {hint}",
                file=sys.stderr,
            )

    n = len(missing)
    if n == 0:
        print("All required keys present - ready to run", flush=True)
        _warn_if_local_mongo_unreachable()
        _warn_if_local_neo4j_unreachable()
        return True

    print(f"{n} keys missing - cannot start", file=sys.stderr)
    return False


def main() -> None:
    ok = validate_environment()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
