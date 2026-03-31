"""Smoke tests for the FastAPI agent service (no external APIs)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.agent_service import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"
    assert data.get("service") == "agent_service"
    assert "timestamp" in data


def test_root_returns_html_when_ui_present(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code in (200, 404)
    if response.status_code == 200:
        assert "text/html" in response.headers.get("content-type", "")


def test_tools_lists_catalog(client: TestClient) -> None:
    response = client.get("/tools")
    assert response.status_code == 200
    data = response.json()
    assert "tools" in data
    assert isinstance(data["tools"], list)
    assert data.get("count", 0) >= 1
