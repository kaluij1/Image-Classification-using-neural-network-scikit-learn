"""Binary metrics for defective-vs-ok. Positive class is defective.

Primary metric (PROBLEM.md): defective-class recall on official test,
at a threshold chosen on validation only.

Operating-point rule: on validation, maximize defective recall among
thresholds whose defective precision is at least 0.5. That prefers
missing fewer defects (FN cost more than FP) without inventing a
numeric recall target such as 95%. Ties go to higher precision.
Max-F1 is still computed as a high-precision reference; it is not
the decision threshold.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

MIN_PRECISION_FOR_RECALL = 0.5

POSITIVE_LABEL = 1


def binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64)
    y_pred = (y_score >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(x) for x in cm.ravel())
    per_class_recall = recall_score(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0
    )
    per_class_precision = precision_score(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0
    )

    metrics: dict[str, Any] = {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "n_defective": int(y_true.sum()),
        "n_ok": int((y_true == 0).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_defective": float(
            precision_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)
        ),
        "recall_defective": float(
            recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)
        ),
        "f1_defective": float(
            f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)
        ),
        "recall_ok": float(per_class_recall[0]),
        "recall_defective_from_per_class": float(per_class_recall[1]),
        "precision_ok": float(per_class_precision[0]),
        "confusion_matrix": {
            "labels": ["ok", "defective"],
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
        "accuracy_is_misleading": True,
    }

    if len(np.unique(y_true)) == 2:
        metrics["pr_auc_defective"] = float(average_precision_score(y_true, y_score))
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    else:
        metrics["pr_auc_defective"] = None
        metrics["roc_auc"] = None
    return metrics


def _pr_arrays(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    precision, recall, thresholds = precision_recall_curve(
        y_true, y_score, pos_label=POSITIVE_LABEL
    )
    return precision, recall, thresholds


def _empty_selection(rule: str) -> dict[str, Any]:
    return {
        "rule": rule,
        "threshold": 0.5,
        "f1_defective": 0.0,
        "recall_defective": 0.0,
        "precision_defective": 0.0,
        "n_thresholds_scanned": 0,
    }


def _selection_at(
    rule: str,
    precision: np.ndarray,
    recall: np.ndarray,
    thresholds: np.ndarray,
    y_true: np.ndarray,
    y_score: np.ndarray,
    index: int,
) -> dict[str, Any]:
    threshold = float(thresholds[index])
    pred = (y_score >= threshold).astype(int)
    return {
        "rule": rule,
        "threshold": threshold,
        "f1_defective": float(
            f1_score(y_true, pred, pos_label=POSITIVE_LABEL, zero_division=0)
        ),
        "recall_defective": float(recall[index]),
        "precision_defective": float(precision[index]),
        "n_thresholds_scanned": int(thresholds.size),
    }


def select_threshold_max_f1(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, Any]:
    """High-precision reference: max defective F1 on val."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64)
    precision, recall, thresholds = _pr_arrays(y_true, y_score)
    if thresholds.size == 0:
        return _empty_selection("max_defective_f1_on_val; tie_break_higher_recall")

    f1 = np.array(
        [
            f1_score(
                y_true,
                (y_score >= thresh).astype(int),
                pos_label=POSITIVE_LABEL,
                zero_division=0,
            )
            for thresh in thresholds
        ]
    )
    best = float(f1.max())
    candidates = np.where(np.isclose(f1, best))[0]
    best_i = int(candidates[np.argmax(recall[:-1][candidates])])
    return _selection_at(
        "max_defective_f1_on_val; tie_break_higher_recall",
        precision,
        recall,
        thresholds,
        y_true,
        y_score,
        best_i,
    )


def select_threshold_max_recall(
    y_true: np.ndarray,
    y_score: np.ndarray,
    min_precision: float = MIN_PRECISION_FOR_RECALL,
) -> dict[str, Any]:
    """Pick a recall-preferring threshold on validation only.

    Maximize defective recall among thresholds with defective precision
    at least ``min_precision``. Tie-break: higher precision. If nothing
    meets the floor, fall back to max F1.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64)
    precision, recall, thresholds = _pr_arrays(y_true, y_score)
    rule = (
        f"max_defective_recall_on_val_subject_to_precision>={min_precision:g}; "
        "tie_break_higher_precision"
    )
    if thresholds.size == 0:
        return _empty_selection(rule)

    feasible = np.where(precision[:-1] >= min_precision)[0]
    if feasible.size == 0:
        fallback = select_threshold_max_f1(y_true, y_score)
        fallback["rule"] = f"{rule}; fallback_max_f1"
        return fallback

    best_recall = float(recall[:-1][feasible].max())
    tied = feasible[np.isclose(recall[:-1][feasible], best_recall)]
    best_i = int(tied[np.argmax(precision[:-1][tied])])
    selected = _selection_at(
        rule, precision, recall, thresholds, y_true, y_score, best_i
    )
    selected["min_precision"] = float(min_precision)
    return selected


def pr_curve_points(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, list[float]]:
    precision, recall, thresholds = precision_recall_curve(
        y_true, y_score, pos_label=POSITIVE_LABEL
    )
    return {
        "precision": [float(x) for x in precision],
        "recall": [float(x) for x in recall],
        "thresholds": [float(x) for x in thresholds],
    }
