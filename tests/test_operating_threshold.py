"""Locked 46/49 recall-floor rule on synthetic scores. No test-set scoring."""

from __future__ import annotations

import numpy as np

from ksdd2_metrics import (
    MIN_RECALL_FOR_THRESHOLD,
    VAL_RECALL_FLOOR_CAUGHT,
    VAL_RECALL_FLOOR_N,
    binary_metrics,
    select_operating_threshold,
    select_threshold_max_f1,
    select_threshold_max_recall,
    select_threshold_recall_floor,
)


def test_locked_floor_constants_are_46_of_49() -> None:
    assert VAL_RECALL_FLOOR_CAUGHT == 46
    assert VAL_RECALL_FLOOR_N == 49
    assert MIN_RECALL_FOR_THRESHOLD == 46 / 49


def _diverging_scores() -> tuple[np.ndarray, np.ndarray]:
    """Construct scores where 46/49, precision-floor, and max-F1 disagree.

    40 easy defectives at 0.80, 6 mid at 0.30, 3 hard at 0.08.
    Many ok parts sit just above 0.30 so the recall-floor cutoff has
    precision < 0.5. Highest cutoff that still catches 46/49 is 0.30.
    """
    y_true = np.concatenate(
        [
            np.ones(40),
            np.ones(6),
            np.ones(3),
            np.zeros(70),
            np.zeros(8),
        ]
    )
    y_score = np.concatenate(
        [
            np.full(40, 0.80),
            np.full(6, 0.30),
            np.full(3, 0.08),
            np.full(70, 0.32),
            np.full(8, 0.75),
        ]
    )
    return y_true, y_score


def test_operating_threshold_is_highest_cutoff_with_46_of_49() -> None:
    y_true, y_score = _diverging_scores()
    selected = select_operating_threshold(y_true, y_score)
    at = binary_metrics(y_true, y_score, selected["threshold"])

    assert selected["floor_met"] is True
    assert selected["min_recall"] == 46 / 49
    assert at["confusion_matrix"]["tp"] == 46
    assert at["confusion_matrix"]["fn"] == 3
    assert at["recall_defective"] >= 46 / 49
    assert selected["threshold"] == 0.30


def test_operating_rule_is_not_precision_floor_or_max_f1() -> None:
    y_true, y_score = _diverging_scores()
    operating = select_operating_threshold(y_true, y_score)
    precision_floor = select_threshold_max_recall(y_true, y_score)
    max_f1 = select_threshold_max_f1(y_true, y_score)

    assert operating["threshold"] != precision_floor["threshold"]
    assert operating["threshold"] != max_f1["threshold"]
    assert operating["threshold"] < precision_floor["threshold"]
    assert operating["threshold"] < max_f1["threshold"]


def test_higher_cutoff_breaks_the_46_of_49_floor() -> None:
    y_true, y_score = _diverging_scores()
    selected = select_operating_threshold(y_true, y_score)
    stricter = binary_metrics(y_true, y_score, selected["threshold"] + 0.01)
    assert stricter["confusion_matrix"]["tp"] < 46
    assert stricter["recall_defective"] < 46 / 49


def test_select_operating_threshold_delegates_to_recall_floor() -> None:
    y_true, y_score = _diverging_scores()
    direct = select_threshold_recall_floor(
        y_true, y_score, min_recall=MIN_RECALL_FOR_THRESHOLD
    )
    via_alias = select_operating_threshold(y_true, y_score)
    assert via_alias["threshold"] == direct["threshold"]
    assert via_alias["rule"] == direct["rule"]


def test_fallback_when_floor_cannot_be_met() -> None:
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([0.9, 0.2, 0.1, 0.05])
    selected = select_threshold_recall_floor(y_true, y_score, min_recall=1.1)
    assert selected["floor_met"] is False
    assert "fallback_max_recall" in selected["rule"]
