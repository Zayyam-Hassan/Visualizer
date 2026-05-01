"""Tests for the health and root endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_returns_200() -> None:
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "running"
    assert "service" in body


def test_health_returns_200() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body
    assert "device" in body
    assert "timestamp" in body


def test_health_schema_fields() -> None:
    response = client.get("/health")
    body = response.json()
    required = {"status", "app_name", "app_env", "model_loaded", "model_name", "device", "timestamp"}
    assert required.issubset(body.keys()), f"Missing keys: {required - body.keys()}"


def test_docs_accessible() -> None:
    response = client.get("/docs")
    assert response.status_code == 200
