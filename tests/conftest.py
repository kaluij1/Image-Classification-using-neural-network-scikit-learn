"""Shared fixtures. Tests never load best.pt or the KSDD2 zip."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "reports" / "baseline" / "metrics.json"


@pytest.fixture(scope="session")
def locked_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def locked_threshold(locked_metrics: dict) -> float:
    return float(locked_metrics["threshold_selection"]["threshold"])
