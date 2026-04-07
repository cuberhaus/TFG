"""Tests for the TFG FastAPI backend.

These tests exercise endpoints that work without ML model dependencies.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code", "src"))

import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_models_list():
    r = client.get("/api/models")
    assert r.status_code == 200
    data = r.json()
    assert "models" in data
    assert isinstance(data["models"], list)


def test_losses_files():
    r = client.get("/api/losses/files")
    assert r.status_code == 200
    data = r.json()
    assert "files" in data
    assert isinstance(data["files"], list)


def test_prepare_status():
    r = client.get("/api/prepare/status")
    assert r.status_code == 200


def test_generate_status():
    r = client.get("/api/generate/status")
    assert r.status_code == 200


def test_train_status():
    r = client.get("/api/train/status")
    assert r.status_code == 200


def test_evaluate_status():
    r = client.get("/api/evaluate/status")
    assert r.status_code == 200


def test_hpo_status():
    r = client.get("/api/hpo/status")
    assert r.status_code == 200


def test_performance():
    r = client.get("/api/performance")
    assert r.status_code == 200


def test_dataset_train():
    r = client.get("/api/dataset/train")
    assert r.status_code == 200


def test_dataset_test():
    r = client.get("/api/dataset/test")
    assert r.status_code == 200
