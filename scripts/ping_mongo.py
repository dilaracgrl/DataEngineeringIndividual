#!/usr/bin/env python3
"""Ping MongoDB using pymongo (same URI as the pipeline)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient

_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    load_dotenv(_ROOT / ".env")
    uri = os.getenv("MONGO_URI", "mongodb://localhost:27017").strip()
    print(f"Connecting to {uri!r} ...", flush=True)
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
    except Exception as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
    print("OK: admin ping succeeded")
    sys.exit(0)


if __name__ == "__main__":
    main()
