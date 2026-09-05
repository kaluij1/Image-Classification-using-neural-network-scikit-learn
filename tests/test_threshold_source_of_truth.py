"""Serving path must read the cutoff from metrics.json, not a literal."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVE_PATHS = (
    ROOT / "src" / "ksdd2_serve.py",
    ROOT / "src" / "ksdd2_api.py",
    ROOT / "scripts" / "serve_api.py",
)


def _threshold_literal_assignments(path: Path) -> list[float]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[float] = []
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            value = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
            value = node.value
        if value is None:
            continue
        if not any(_is_threshold_target(target) for target in targets):
            continue
        if isinstance(value, ast.Constant) and isinstance(value.value, (int, float)):
            found.append(float(value.value))
    return found


def _is_threshold_target(target: ast.expr) -> bool:
    if isinstance(target, ast.Name) and target.id == "threshold":
        return True
    return isinstance(target, ast.Attribute) and target.attr == "threshold"


def test_metrics_json_is_the_locked_46_49_row(locked_metrics: dict) -> None:
    selection = locked_metrics["threshold_selection"]
    assert selection["floor_met"] is True
    assert selection["min_recall"] == 46 / 49
    assert "highest_threshold_on_val_with_defective_recall" in selection["rule"]
    threshold = float(selection["threshold"])
    assert 0.0 < threshold < 1.0
    assert threshold != 0.5
    assert (
        threshold
        != locked_metrics["threshold_selection_precision_floor_reference"]["threshold"]
    )
    assert threshold != locked_metrics["threshold_selection_f1_reference"]["threshold"]


def test_serve_loads_threshold_selection_from_metrics_json() -> None:
    source = (ROOT / "src" / "ksdd2_serve.py").read_text(encoding="utf-8")
    assert 'report["threshold_selection"]' in source
    assert 'selection["threshold"]' in source
    assert "DEFAULT_METRICS" in source
    assert "metrics.json" in source


def test_serving_path_does_not_hardcode_an_operating_threshold(
    locked_threshold: float,
    locked_metrics: dict,
) -> None:
    alternates = {
        0.5,
        float(locked_metrics["threshold_selection_precision_floor_reference"]["threshold"]),
        float(locked_metrics["threshold_selection_f1_reference"]["threshold"]),
        locked_threshold,
    }
    for path in SERVE_PATHS:
        assigned = _threshold_literal_assignments(path)
        assert assigned == [], f"{path.name} hardcodes threshold {assigned}"
        text = path.read_text(encoding="utf-8")
        for value in alternates:
            assert f"{value}" not in text, f"{path.name} embeds {value}"
