"""Phase 3 error analysis of the frozen Phase 2 checkpoint.

Does not train, does not retune the threshold, and does not change the
backbone. Uses official GT masks for size / location / Grad-CAM overlap
only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ksdd2_analysis import (  # noqa: E402
    pick_fp_examples,
    pick_tp_examples,
    plot_area_bars,
    plot_cam_gallery,
    plot_case_gallery,
    plot_score_strip,
    summarize_cam,
    write_error_report_bundle,
)
from ksdd2_errors import (  # noqa: E402
    AREA_BINS,
    CAM_NEIGHBORHOOD_RADIUS_PX,
    TINY_AREA_CUTOFF,
    build_cases,
    confusion_counts,
    error_type,
    filter_cases,
    location_bin,
    mask_geometry,
    score_summary,
    stratify_by_area,
    stratify_by_location,
    tiny_defect_rows,
)
from ksdd2_gradcam import CAM_BIN_FRACTION, FeatureGradCAM, explain_sample  # noqa: E402
from ksdd2_inventory import Sample, find_dataset_root, load_inventory  # noqa: E402
from ksdd2_model import build_baseline  # noqa: E402
from ksdd2_splits import load_manifest, validate_manifest  # noqa: E402
from ksdd2_train import resolve_device  # noqa: E402
from ksdd2_transforms import eval_transforms  # noqa: E402

def _cm_counts(metrics_cm: dict) -> dict[str, int]:
    return {
        "TP": int(metrics_cm["tp"]),
        "FP": int(metrics_cm["fp"]),
        "FN": int(metrics_cm["fn"]),
        "TN": int(metrics_cm["tn"]),
    }


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _score_samples(
    model: torch.nn.Module,
    samples: list[Sample],
    device: torch.device,
    threshold: float,
    role_of: dict[tuple[str, str], str],
) -> list[dict]:
    transform = eval_transforms()
    model.eval()
    cases: list[dict] = []
    with torch.no_grad():
        for sample in samples:
            with Image.open(sample.image_path) as image:
                tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
            logit = float(model(tensor).squeeze().cpu())
            y_score = 1.0 / (1.0 + math.exp(-logit))
            y_true = 1 if sample.label == "defective" else 0
            y_pred = int(y_score >= threshold)
            geometry = mask_geometry(sample)
            cases.append(
                {
                    "sample_id": sample.sample_id,
                    "official_split": sample.split,
                    "role": role_of[(sample.split, sample.sample_id)],
                    "label": sample.label,
                    "y_true": y_true,
                    "y_score": y_score,
                    "y_pred": y_pred,
                    "error_type": error_type(y_true, y_pred),
                    "image_width": sample.width,
                    "image_height": sample.height,
                    **geometry,
                }
            )
    return cases


def _cam_public(item: dict, case: dict) -> dict:
    return {
        "sample_id": item["sample_id"],
        "error_type": case["error_type"],
        "y_score": case["y_score"],
        "area_fraction": case["area_fraction"],
        "area_bin": case["area_bin"],
        "metrics": item["metrics"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=ROOT / "reports" / "baseline",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports" / "error_analysis",
    )
    args = parser.parse_args()

    baseline_dir = args.baseline_dir
    metrics = _load_json(baseline_dir / "metrics.json")
    threshold = float(metrics["threshold_selection"]["threshold"])
    checkpoint = baseline_dir / "checkpoints" / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    data_root = find_dataset_root(args.data_root)
    samples = load_inventory(data_root)
    manifest = load_manifest(baseline_dir / "split_manifest.json")
    validate_manifest(manifest, samples)
    samples_by_id = {s.sample_id: s for s in samples}

    predictions_by_role = {
        "val": _load_json(baseline_dir / "predictions_val.json"),
        "test": _load_json(baseline_dir / "predictions_test.json"),
    }
    cases = build_cases(samples, manifest, predictions_by_role, threshold)

    test_cases = filter_cases(cases, role="test")
    val_cases = filter_cases(cases, role="val")
    test_cm = confusion_counts(test_cases)
    val_cm = confusion_counts(val_cases)
    expected_test = _cm_counts(
        metrics["metrics_at_chosen_threshold"]["test"]["confusion_matrix"]
    )
    expected_val = _cm_counts(
        metrics["metrics_at_chosen_threshold"]["val"]["confusion_matrix"]
    )
    if test_cm != expected_test:
        raise RuntimeError(f"Test confusion {test_cm} != metrics.json {expected_test}")
    if val_cm != expected_val:
        raise RuntimeError(f"Val confusion {val_cm} != metrics.json {expected_val}")

    test_def = [c for c in test_cases if c["label"] == "defective"]
    test_fn = sorted(
        filter_cases(test_cases, error="FN"), key=lambda c: c["area_fraction"]
    )
    test_fp = sorted(
        filter_cases(test_cases, error="FP"), key=lambda c: -c["y_score"]
    )
    test_tp = filter_cases(test_cases, error="TP")

    device = resolve_device()
    model = build_baseline(pretrained=False).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    model.eval()

    role_of = {
        (row["official_split"], row["sample_id"]): role
        for role in ("train", "val", "test")
        for row in manifest[role]
    }
    scored_ids = {(c["official_split"], c["sample_id"]) for c in cases}
    tiny_unscored = [
        s
        for s in samples
        if s.label == "defective"
        and (s.mask_foreground_pixels / (s.width * s.height)) < TINY_AREA_CUTOFF
        and (s.split, s.sample_id) not in scored_ids
    ]
    if tiny_unscored:
        cases.extend(_score_samples(model, tiny_unscored, device, threshold, role_of))

    tiny_rows = tiny_defect_rows(cases)
    tiny_test = [r for r in tiny_rows if r["role"] == "test"]
    tiny_payload = {
        "n_dataset": len(tiny_rows),
        "n_train": sum(1 for r in tiny_rows if r["role"] == "train"),
        "n_val": sum(1 for r in tiny_rows if r["role"] == "val"),
        "n_test": len(tiny_test),
        "n_test_fn": sum(1 for r in tiny_test if r["error_type"] == "FN"),
        "n_test_tp": sum(1 for r in tiny_test if r["error_type"] == "TP"),
        "rows": tiny_rows,
    }

    fp_cam_cases = pick_fp_examples(test_fp)
    tp_cam_cases = pick_tp_examples(test_tp)
    camber = FeatureGradCAM(model)
    try:
        fn_expl = [
            explain_sample(camber, samples_by_id[c["sample_id"]], device)
            for c in test_fn
        ]
        tp_expl = [
            explain_sample(camber, samples_by_id[c["sample_id"]], device)
            for c in tp_cam_cases
        ]
        fp_expl = [
            explain_sample(camber, samples_by_id[c["sample_id"]], device)
            for c in fp_cam_cases
        ]
    finally:
        camber.close()

    cases_by_id = {c["sample_id"]: c for c in cases}
    feature_hw = fn_expl[0]["metrics"]["cam_feature_hw"] if fn_expl else [0, 0]
    fp_peak_location = {"top": 0, "middle": 0, "bottom": 0}
    for item in fp_expl:
        loc = location_bin(item["metrics"]["peak_y_frac"])
        if loc is not None:
            fp_peak_location[loc] += 1

    output_dir = args.output_dir
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    plot_case_gallery(
        test_fn,
        samples_by_id,
        figures / "gallery_fn_test.png",
        title="Test false negatives — official GT overlay (yellow box padded 8 px)",
        cols=4,
        with_mask=True,
    )
    plot_case_gallery(
        test_fp,
        samples_by_id,
        figures / "gallery_fp_test.png",
        title="Test false positives — official GT is empty",
        cols=14 if len(test_fp) > 80 else 10,
        with_mask=False,
        thumb_size=(56, 140) if len(test_fp) > 80 else (72, 180),
    )
    plot_cam_gallery(
        fn_expl,
        cases_by_id,
        samples_by_id,
        figures / "gallery_gradcam_fn.png",
        title="Grad-CAM on all test FNs (yellow = GT edge)",
    )
    plot_cam_gallery(
        tp_expl,
        cases_by_id,
        samples_by_id,
        figures / "gallery_gradcam_tp.png",
        title="Grad-CAM on sampled test TPs (up to 2 per area bin)",
    )
    plot_cam_gallery(
        fp_expl,
        cases_by_id,
        samples_by_id,
        figures / "gallery_gradcam_fp.png",
        title="Grad-CAM on sampled test FPs (12 highest-score + 4 nearest threshold)",
    )
    plot_area_bars(
        stratify_by_area(test_def),
        figures / "area_stratification.png",
    )
    plot_score_strip(
        test_tp, test_fn, test_fp, threshold, figures / "score_strip.png"
    )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "best_epoch": metrics["best_epoch"],
        "best_val_pr_auc": metrics["best_val_pr_auc"],
        "threshold": threshold,
        "threshold_rule": metrics["threshold_selection"]["rule"],
        "cam_bin_fraction": CAM_BIN_FRACTION,
        "neighborhood_radius_px": CAM_NEIGHBORHOOD_RADIUS_PX,
        "area_bins": [name for _a, _b, name in AREA_BINS],
        "test_confusion": test_cm,
        "val_confusion": val_cm,
        "test_fn": test_fn,
        "test_fp": test_fp,
        "val_fn": filter_cases(val_cases, error="FN"),
        "test_fn_scores": score_summary(test_fn),
        "test_fp_scores": score_summary(test_fp),
        "test_fn_near_threshold": sum(
            1 for c in test_fn if c["y_score"] >= threshold - 0.05
        ),
        "test_fn_tiny": sum(1 for c in test_fn if c["area_fraction"] < TINY_AREA_CUTOFF),
        "test_fp_between_chosen_and_half": sum(
            1 for c in test_fp if c["y_score"] < 0.5
        ),
        "test_fp_high_conf": sum(1 for c in test_fp if c["y_score"] >= 0.8),
        "test_area_stratification": stratify_by_area(test_def),
        "test_location_stratification": stratify_by_location(test_def),
        "tiny_defects": tiny_payload,
        "gradcam": {
            "feature_hw": feature_hw,
            "fn": summarize_cam(fn_expl),
            "tp": summarize_cam(tp_expl),
            "fp": summarize_cam(fp_expl),
            "fp_peak_location": fp_peak_location,
            "examples": {
                "fn": [_cam_public(item, cases_by_id[item["sample_id"]]) for item in fn_expl],
                "tp": [_cam_public(item, cases_by_id[item["sample_id"]]) for item in tp_expl],
                "fp": [_cam_public(item, cases_by_id[item["sample_id"]]) for item in fp_expl],
            },
        },
    }
    write_error_report_bundle(report, output_dir)
    print(f"Wrote {output_dir}")
    print(
        f"Test FN={test_cm['FN']} FP={test_cm['FP']} | "
        f"tiny-dataset={tiny_payload['n_dataset']} "
        f"tiny-test-FN={tiny_payload['n_test_fn']}/{tiny_payload['n_test']}"
    )
    print(
        f"Grad-CAM FN hit {report['gradcam']['fn']['n_hit']}/"
        f"{report['gradcam']['fn']['n']}"
    )


if __name__ == "__main__":
    main()
