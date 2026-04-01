"""
Diagnostic: run each tool/scraper once with a small query.

Note on names vs this repo:
  - arxiv_tool exposes ``search_arxiv`` (there is no ``search_papers``).
  - producthunt_tool exposes ``search_producthunt`` (not ``search_products``).
  - trends_tool exposes ``search_trends`` (not ``get_trends``).

Loads ``.env`` from the project root so API keys are available.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

load_dotenv(_ROOT / ".env")
logging.basicConfig(level=logging.CRITICAL, force=True)
logging.disable(logging.CRITICAL)

from scrapers.techcrunch_scraper import search_funding_articles
from scrapers.yc_scraper import search_yc_companies
from tools.arxiv_tool import search_arxiv
from tools.github_tool import search_repositories
from tools.news_tool import search_funding_news
from tools.patents_tool import search_patents
from tools.producthunt_tool import search_producthunt
from tools.semantic_scholar_tool import search_papers
from tools.trends_tool import search_trends
from tools.wikipedia_tool import search_wikipedia


def _ok(result: Any) -> bool:
    """Treat empty list/dict as failure for quick diagnostics."""
    if result is None:
        return False
    if isinstance(result, list) and len(result) == 0:
        return False
    if isinstance(result, dict) and len(result) == 0:
        return False
    return True


def _run_case(label: str, fn: Callable[[], Any]) -> bool:
    try:
        result = fn()
        passed = _ok(result)
        status = "PASS" if passed else "FAIL"
        print(f"{status}: {label}")
        if not passed:
            print("       (empty or None result)")
        return passed
    except Exception as e:
        print(f"FAIL: {label}")
        print(f"       {type(e).__name__}: {e}")
        return False


def main() -> None:
    tests: list[tuple[str, Callable[[], Any]]] = [
        (
            'arxiv_tool — search_arxiv("machine learning")',
            lambda: search_arxiv("machine learning", max_results=3),
        ),
        (
            'semantic_scholar_tool — search_papers("neural networks")',
            lambda: search_papers("neural networks", limit=3),
        ),
        (
            'github_tool — search_repositories("artificial intelligence")',
            lambda: search_repositories("artificial intelligence", limit=3),
        ),
        (
            'producthunt_tool — search_producthunt("AI tools")',
            lambda: search_producthunt("AI tools", limit=3),
        ),
        (
            'patents_tool — search_patents("machine learning")',
            lambda: search_patents("machine learning", max_results=3),
        ),
        (
            'news_tool — search_funding_news("artificial intelligence")',
            lambda: search_funding_news("artificial intelligence", page_size=5),
        ),
        (
            'wikipedia_tool — search_wikipedia("machine learning")',
            lambda: search_wikipedia("machine learning"),
        ),
        (
            'trends_tool — search_trends("AI")',
            lambda: search_trends("AI"),
        ),
        (
            'yc_scraper — search_yc_companies("AI")',
            lambda: search_yc_companies("AI", limit=5),
        ),
        (
            'techcrunch_scraper — search_funding_articles("artificial intelligence")',
            lambda: search_funding_articles("artificial intelligence", limit=5),
        ),
    ]

    passed = 0
    failed = 0
    for label, fn in tests:
        if _run_case(label, fn):
            passed += 1
        else:
            failed += 1

    print()
    print(f"Summary: {passed} passed, {failed} failed")


if __name__ == "__main__":
    main()
