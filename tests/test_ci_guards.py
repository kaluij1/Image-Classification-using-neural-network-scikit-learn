"""CI stays cheap: no dataset, no weights, no retrain, no Docker rebuild."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"

FORBIDDEN_IN_WORKFLOW = (
    "download_ksdd2",
    "train_baseline",
    "run_error_analysis",
    "run_threshold_tradeoff",
    "docker compose",
    "docker-compose",
    "docker build",
    "best.pt",
    "KolektorSDD2",
)


def test_workflow_exists_and_targets_main() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "python -m pytest" in text
    assert "requirements-dev.txt" in text
    assert "3.13" in text
    assert "branches: [main]" in text


def test_workflow_does_not_train_score_or_download() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    for needle in FORBIDDEN_IN_WORKFLOW:
        assert needle not in text, f"CI workflow mentions {needle}"


def test_weights_and_dataset_are_not_tracked() -> None:
    tracked = subprocess.check_output(
        ["git", "ls-files"],
        cwd=ROOT,
        text=True,
    )
    assert "reports/baseline/metrics.json" in tracked
    for line in tracked.splitlines():
        assert not line.endswith(".pt")
        assert not line.endswith(".pth")
        assert not line.endswith(".zip")
        assert not line.startswith("data/raw/")
        assert "checkpoints/best.pt" not in line
