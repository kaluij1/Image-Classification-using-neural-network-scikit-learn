"""Write Phase 2 baseline reports and figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_history(history: list[dict[str, Any]], dest: Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="val")
    axes[0].set_title("BCE-with-logits loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["val_pr_auc"] for row in history], label="val PR-AUC")
    axes[1].plot(
        epochs,
        [row["val_recall_defective@0.5"] for row in history],
        label="val defective recall @ 0.5",
    )
    axes[1].plot(
        epochs,
        [row["val_f1_defective@0.5"] for row in history],
        label="val defective F1 @ 0.5",
    )
    axes[1].set_title("Validation metrics during training")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=120)
    plt.close(fig)


def plot_pr_curve(
    precision: list[float],
    recall: list[float],
    dest: Path,
    *,
    title: str,
    ap: float | None,
    operating_point: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    label = f"PR curve (AP={ap:.3f})" if ap is not None else "PR curve"
    ax.plot(recall, precision, label=label)
    if operating_point is not None:
        ax.scatter(
            [operating_point[0]],
            [operating_point[1]],
            zorder=3,
            label="val-chosen threshold",
        )
    ax.set_title(title)
    ax.set_xlabel("Recall (defective)")
    ax.set_ylabel("Precision (defective)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend()
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=120)
    plt.close(fig)


def plot_confusion_matrices(
    by_role: dict[str, dict[str, Any]],
    dest: Path,
    threshold: float,
) -> None:
    roles = [role for role in ("train", "val", "test") if role in by_role]
    fig, axes = plt.subplots(1, len(roles), figsize=(4.2 * len(roles), 3.8))
    if len(roles) == 1:
        axes = [axes]
    for ax, role in zip(axes, roles):
        cm = by_role[role]["confusion_matrix"]
        mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]], dtype=int)
        ax.imshow(mat, cmap="Blues")
        ax.set_title(f"{role} @ {threshold:.3f}")
        ax.set_xticks([0, 1], ["pred ok", "pred def"])
        ax.set_yticks([0, 1], ["true ok", "true def"])
        for (i, j), value in np.ndenumerate(mat):
            ax.text(j, i, str(value), ha="center", va="center")
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=120)
    plt.close(fig)


def write_metrics_markdown(report: dict[str, Any], dest: Path) -> None:
    counts = report["split_counts"]
    selection = report["threshold_selection"]
    chosen = report["metrics_at_chosen_threshold"]
    default = report["metrics_at_0.5"]
    lines = [
        "# Phase 2 — PyTorch baseline",
        "",
        f"Generated (UTC): {report['generated_at_utc']}",
        "",
        "Image-level defective vs ok. Positive = defective (hold for review).",
        "Official test was frozen. Validation was carved from official train only.",
        "Numbers below were computed from this run. Accuracy is reported and treated as misleading.",
        "",
        "## Setup",
        "",
        f"- Backbone: `{report['config']['backbone']}` (ImageNet pretrained={report['config']['pretrained']})",
        f"- Input (letterbox): {report['config']['input_width']}×{report['config']['input_height']} (W×H)",
        f"- Loss: BCEWithLogitsLoss, pos_weight={report['pos_weight']:.4f} (train fold only: n_ok/n_defective)",
        f"- Optimizer: AdamW, lr_head={report['config']['lr_head']}, lr_backbone={report['config']['lr_backbone']}",
        f"- Epochs: {report['config']['epochs']} (backbone frozen for first {report['config']['freeze_epochs']})",
        f"- Best checkpoint: epoch {report['best_epoch']} by validation PR-AUC ({report['best_val_pr_auc']:.4f})",
        f"- Device: {report['device']}",
        "",
        "## Splits",
        "",
        "| Role | N | Defective | OK | Source |",
        "|---|---:|---:|---:|---|",
    ]
    sources = {
        "train": "official train, stratified remainder",
        "val": "official train, stratified holdout",
        "test": "official test (frozen)",
    }
    for role in ("train", "val", "test"):
        row = counts[role]
        lines.append(
            f"| {role} | {row['n']} | {row['defective']} | {row['ok']} | {sources[role]} |"
        )
    lines.extend(
        [
            "",
            "No train/val/test ID overlap. Copy files `10301 (copy)` are outside the inventory.",
            "",
            "## Operating point",
            "",
            f"- Rule: {selection['rule']}",
            f"- Threshold chosen on **val only**: {selection['threshold']:.6f}",
            f"- Val at that threshold: precision={selection['precision_defective']:.4f}, "
            f"recall={selection['recall_defective']:.4f}, F1={selection['f1_defective']:.4f}",
            "- Test was scored once at this threshold after selection. Test was not used to pick it.",
            "",
            "## Metrics at the val-chosen threshold",
            "",
            "| Split | Accuracy* | Precision (def) | Recall (def) | F1 (def) | Recall (ok) | PR-AUC (def) | ROC-AUC | TP | FP | FN | TN |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for role in ("train", "val", "test"):
        m = chosen[role]
        cm = m["confusion_matrix"]
        pr = "n/a" if m["pr_auc_defective"] is None else f"{m['pr_auc_defective']:.4f}"
        roc = "n/a" if m["roc_auc"] is None else f"{m['roc_auc']:.4f}"
        lines.append(
            f"| {role} | {m['accuracy']:.4f} | {m['precision_defective']:.4f} | "
            f"{m['recall_defective']:.4f} | {m['f1_defective']:.4f} | {m['recall_ok']:.4f} | "
            f"{pr} | {roc} | {cm['tp']} | {cm['fp']} | {cm['fn']} | {cm['tn']} |"
        )
    lines.extend(
        [
            "",
            "\\*Accuracy is the wrong primary metric at ~10.7% prevalence.",
            "",
            "## Metrics at threshold 0.5 (reference only)",
            "",
            "| Split | Accuracy* | Precision (def) | Recall (def) | F1 (def) | PR-AUC (def) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for role in ("train", "val", "test"):
        m = default[role]
        pr = "n/a" if m["pr_auc_defective"] is None else f"{m['pr_auc_defective']:.4f}"
        lines.append(
            f"| {role} | {m['accuracy']:.4f} | {m['precision_defective']:.4f} | "
            f"{m['recall_defective']:.4f} | {m['f1_defective']:.4f} | {pr} |"
        )
    lines.extend(
        [
            "",
            "## Primary metric (as defined)",
            "",
            f"- Defective-class recall on official test at the val-chosen threshold: "
            f"**{chosen['test']['recall_defective']:.4f}** "
            f"({chosen['test']['confusion_matrix']['tp']}/"
            f"{chosen['test']['n_defective']} defective images).",
            f"- Threshold-free test PR-AUC (defective): **{chosen['test']['pr_auc_defective']:.4f}**.",
            "",
            "## Threshold tradeoff (same checkpoint, val-only selection)",
            "",
            "False negatives cost more than false positives, so the decision "
            "threshold maximizes validation defective recall among cutoffs "
            "with precision at least 0.5 (holds are not majority false alarms). "
            "Max-F1 is a high-precision reference only. Test was not used to pick "
            "the cutoff.",
            "",
        ]
    )
    f1_ref_sel = report.get("threshold_selection_f1_reference")
    f1_ref_metrics = report.get("metrics_at_f1_reference")
    if f1_ref_sel and f1_ref_metrics:
        lines.append(
            f"- Max-F1 reference ({f1_ref_sel['threshold']:.3f}): test "
            f"**{f1_ref_metrics['test']['confusion_matrix']['fn']} FN / "
            f"{f1_ref_metrics['test']['confusion_matrix']['fp']} FP**, "
            f"recall {f1_ref_metrics['test']['recall_defective']:.4f}, "
            f"precision {f1_ref_metrics['test']['precision_defective']:.4f}"
        )
    lines.extend(
        [
            f"- Chosen recall-preferring ({selection['threshold']:.3f}): test "
            f"**{chosen['test']['confusion_matrix']['fn']} FN / "
            f"{chosen['test']['confusion_matrix']['fp']} FP**, "
            f"recall {chosen['test']['recall_defective']:.4f}, "
            f"precision {chosen['test']['precision_defective']:.4f}",
            f"- Threshold 0.5 (untuned default): test "
            f"**{default['test']['confusion_matrix']['fn']} FN / "
            f"{default['test']['confusion_matrix']['fp']} FP**, "
            f"recall {default['test']['recall_defective']:.4f}, "
            f"precision {default['test']['precision_defective']:.4f}",
            "",
            "This is an operating-point choice, not a retrain.",
            "",
            "## Leakage / protocol notes",
            "",
            "- ImageNet mean/std only (not computed from KSDD2, and not from test).",
            "- `pos_weight` from the training fold only.",
            "- Augmentation on the training role only.",
            "- Threshold selected on validation only; official test frozen.",
            "- Training target is the image-level mask-derived label, not the mask pixels.",
            "",
        ]
    )
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
