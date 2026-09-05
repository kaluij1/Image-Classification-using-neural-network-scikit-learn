"""Train the Phase 2 MobileNetV3-Small baseline and write reports/baseline/."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ksdd2_inventory import find_dataset_root, load_inventory  # noqa: E402
from ksdd2_metrics import (  # noqa: E402
    binary_metrics,
    pr_curve_points,
    select_operating_threshold,
    select_threshold_max_f1,
    select_threshold_max_recall,
)
from ksdd2_report import (  # noqa: E402
    plot_confusion_matrices,
    plot_history,
    plot_pr_curve,
    write_json,
    write_metrics_markdown,
)
from ksdd2_splits import (  # noqa: E402
    load_manifest,
    make_split_manifest,
    manifest_counts,
    save_manifest,
    samples_by_role,
    validate_manifest,
)
from ksdd2_model import build_baseline  # noqa: E402
from ksdd2_train import (  # noqa: E402
    TrainConfig,
    collect_predictions,
    make_loader,
    resolve_device,
    train_baseline,
)


def _prediction_payload(pred: dict) -> dict:
    return {
        "sample_ids": list(pred["sample_ids"]),
        "y_true": [int(x) for x in pred["y_true"]],
        "y_score": [float(x) for x in pred["y_score"]],
    }


def _load_trained_result(
    role_samples: dict,
    config: TrainConfig,
    output_dir: Path,
) -> dict:
    import torch

    existing = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    checkpoint = output_dir / "checkpoints" / "best.pt"
    device = resolve_device()
    model = build_baseline(pretrained=False).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    eval_loaders = {
        role: make_loader(role_samples[role], train=False, config=config)
        for role in ("train", "val", "test")
    }
    return {
        "model": model,
        "device": existing.get("device", str(device)),
        "pos_weight": existing["pos_weight"],
        "history": existing["history"],
        "best_epoch": existing["best_epoch"],
        "best_val_pr_auc": existing["best_val_pr_auc"],
        "best_checkpoint": str(checkpoint),
        "eval_loaders": eval_loaders,
    }


def emit_baseline_report(
    result: dict,
    config: TrainConfig,
    counts: dict,
    output_dir: Path,
) -> None:
    device = next(result["model"].parameters()).device
    predictions = {
        role: collect_predictions(result["model"], result["eval_loaders"][role], device)
        for role in ("train", "val", "test")
    }

    selection = select_operating_threshold(
        predictions["val"]["y_true"], predictions["val"]["y_score"]
    )
    precision_floor_reference = select_threshold_max_recall(
        predictions["val"]["y_true"], predictions["val"]["y_score"]
    )
    f1_reference = select_threshold_max_f1(
        predictions["val"]["y_true"], predictions["val"]["y_score"]
    )
    threshold = selection["threshold"]
    metrics_chosen = {
        role: binary_metrics(pred["y_true"], pred["y_score"], threshold)
        for role, pred in predictions.items()
    }
    metrics_default = {
        role: binary_metrics(pred["y_true"], pred["y_score"], config.default_threshold)
        for role, pred in predictions.items()
    }
    metrics_precision_floor_reference = {
        role: binary_metrics(
            pred["y_true"], pred["y_score"], precision_floor_reference["threshold"]
        )
        for role, pred in predictions.items()
    }
    metrics_f1_reference = {
        role: binary_metrics(pred["y_true"], pred["y_score"], f1_reference["threshold"])
        for role, pred in predictions.items()
    }
    val_curve = pr_curve_points(predictions["val"]["y_true"], predictions["val"]["y_score"])
    test_curve = pr_curve_points(
        predictions["test"]["y_true"], predictions["test"]["y_score"]
    )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": result["device"],
        "config": asdict(config),
        "pos_weight": result["pos_weight"],
        "split_counts": counts,
        "best_epoch": result["best_epoch"],
        "best_val_pr_auc": result["best_val_pr_auc"],
        "best_checkpoint": result["best_checkpoint"],
        "threshold_selection": selection,
        "threshold_selection_precision_floor_reference": precision_floor_reference,
        "threshold_selection_f1_reference": f1_reference,
        "metrics_at_chosen_threshold": metrics_chosen,
        "metrics_at_0.5": metrics_default,
        "metrics_at_precision_floor_reference": metrics_precision_floor_reference,
        "metrics_at_f1_reference": metrics_f1_reference,
        "history": result["history"],
    }

    write_json(output_dir / "config.json", asdict(config))
    write_json(output_dir / "history.json", result["history"])
    write_json(output_dir / "metrics.json", report)
    write_json(output_dir / "val_pr_curve.json", val_curve)
    write_json(output_dir / "predictions_val.json", _prediction_payload(predictions["val"]))
    write_json(output_dir / "predictions_test.json", _prediction_payload(predictions["test"]))
    write_metrics_markdown(report, output_dir / "metrics.md")

    plot_history(result["history"], output_dir / "history.png")
    plot_pr_curve(
        val_curve["precision"],
        val_curve["recall"],
        output_dir / "pr_curve_val.png",
        title="Validation PR curve (defective)",
        ap=metrics_chosen["val"]["pr_auc_defective"],
        operating_point=(
            metrics_chosen["val"]["recall_defective"],
            metrics_chosen["val"]["precision_defective"],
        ),
    )
    plot_pr_curve(
        test_curve["precision"],
        test_curve["recall"],
        output_dir / "pr_curve_test.png",
        title="Official test PR curve (defective)",
        ap=metrics_chosen["test"]["pr_auc_defective"],
        operating_point=(
            metrics_chosen["test"]["recall_defective"],
            metrics_chosen["test"]["precision_defective"],
        ),
    )
    plot_confusion_matrices(
        metrics_chosen, output_dir / "confusion_matrices.png", threshold
    )

    test_m = metrics_chosen["test"]
    print(f"Wrote {output_dir}")
    print(
        f"Primary: test defective recall @ val threshold {threshold:.4f} = "
        f"{test_m['recall_defective']:.4f} "
        f"(PR-AUC={test_m['pr_auc_defective']:.4f})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports" / "baseline",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--freeze-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--refresh-split", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the split and run one train/val batch, then exit.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Reload checkpoints/best.pt and rewrite metrics; do not train.",
    )
    args = parser.parse_args()

    config = TrainConfig(
        seed=args.seed,
        val_fraction=args.val_fraction,
        batch_size=args.batch_size,
        epochs=1 if args.dry_run else args.epochs,
        freeze_epochs=0 if args.dry_run else args.freeze_epochs,
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data_root = find_dataset_root(args.data_root)
    samples = load_inventory(data_root)
    manifest_path = output_dir / "split_manifest.json"
    if manifest_path.is_file() and not args.refresh_split:
        manifest = load_manifest(manifest_path)
        validate_manifest(manifest, samples)
        print(f"Loaded split manifest: {manifest_path}")
    else:
        manifest = make_split_manifest(
            samples, seed=config.seed, val_fraction=config.val_fraction
        )
        save_manifest(manifest, manifest_path)
        print(f"Wrote split manifest: {manifest_path}")

    role_samples = samples_by_role(samples, manifest)
    counts = manifest_counts(manifest)
    print("Split counts:", json.dumps(counts))

    if args.dry_run:
        config.batch_size = min(config.batch_size, 4)
        from ksdd2_train import epoch_pass, make_loader, pos_weight_from_train, set_seed
        from ksdd2_model import build_baseline
        import torch.nn as nn

        device = resolve_device()
        set_seed(config.seed)
        model = build_baseline(pretrained=True).to(device)
        loader = make_loader(role_samples["train"][:8], train=True, config=config)
        val_loader = make_loader(role_samples["val"][:8], train=False, config=config)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight_from_train(role_samples["train"], device)
        )
        optimizer = torch_optimizer(model)
        train_loss = epoch_pass(model, loader, device, criterion, optimizer)
        val_loss = epoch_pass(model, val_loader, device, criterion, None)
        print(f"Dry run ok. device={device} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        return

    if args.eval_only:
        result = _load_trained_result(role_samples, config, output_dir)
    else:
        result = train_baseline(role_samples, config, output_dir)

    emit_baseline_report(result, config, counts, output_dir)


def torch_optimizer(model):
    import torch

    return torch.optim.AdamW(model.parameters(), lr=3e-4)


if __name__ == "__main__":
    main()
