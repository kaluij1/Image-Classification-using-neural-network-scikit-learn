"""Figures and the Phase 3 written report. Counts only; no retuning."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from ksdd2_errors import AREA_BINS, TINY_AREA_CUTOFF, load_mask_binary
from ksdd2_inventory import Sample
from ksdd2_report import write_json

MASK_EDGE_PAD = 8


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _mask_edges(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    if not m.any():
        return m
    up = np.zeros_like(m)
    down = np.zeros_like(m)
    left = np.zeros_like(m)
    right = np.zeros_like(m)
    up[1:] = m[:-1]
    down[:-1] = m[1:]
    left[:, 1:] = m[:, :-1]
    right[:, :-1] = m[:, 1:]
    return m & ~(up & down & left & right)


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    bbox: dict[str, int] | None = None,
) -> np.ndarray:
    out = image.copy()
    hit = mask.astype(bool)
    if hit.any():
        out[hit, 0] = np.clip(out[hit, 0].astype(np.int16) + 110, 0, 255)
        out[hit, 1] = np.clip(out[hit, 1].astype(np.int16) - 30, 0, 255)
        edges = _mask_edges(hit)
        out[edges] = (255, 255, 0)
    if bbox is not None:
        pil = Image.fromarray(out)
        draw = ImageDraw.Draw(pil)
        draw.rectangle(
            [
                max(0, bbox["xmin"] - MASK_EDGE_PAD),
                max(0, bbox["ymin"] - MASK_EDGE_PAD),
                min(out.shape[1] - 1, bbox["xmax"] + MASK_EDGE_PAD),
                min(out.shape[0] - 1, bbox["ymax"] + MASK_EDGE_PAD),
            ],
            outline=(255, 220, 0),
            width=2,
        )
        out = np.asarray(pil)
    return out


def overlay_cam(
    image: np.ndarray,
    cam: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    image_f = image.astype(np.float32) / 255.0
    cam_n = np.clip(np.asarray(cam, dtype=np.float32), 0.0, 1.0)
    heatmap = plt.cm.jet(cam_n)[..., :3]
    blend = (0.55 * image_f + 0.45 * heatmap)
    out = np.clip(blend * 255.0, 0, 255).astype(np.uint8)
    if mask is not None and mask.any():
        edges = _mask_edges(mask.astype(bool))
        out[edges] = (255, 255, 0)
    return out


def _caption(case: dict[str, Any], *, with_area: bool) -> str:
    line = f"{case['sample_id']}  s={case['y_score']:.3f}"
    if with_area and case["label"] == "defective":
        line += f"  a={100.0 * case['area_fraction']:.3f}%"
    return line


def plot_case_gallery(
    cases: list[dict[str, Any]],
    samples_by_id: dict[str, Sample],
    dest: Path,
    *,
    title: str,
    cols: int,
    with_mask: bool,
    thumb_size: tuple[int, int] = (96, 240),
) -> None:
    if not cases:
        return
    n = len(cases)
    cols = min(cols, n)
    rows = int(np.ceil(n / cols))
    fig_w = 1.35 * cols
    fig_h = 3.15 * rows + 0.55
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes_flat = np.atleast_1d(axes).ravel()
    for ax in axes_flat:
        ax.axis("off")
    for ax, case in zip(axes_flat, cases):
        sample = samples_by_id[case["sample_id"]]
        image = _read_rgb(sample.image_path)
        if with_mask and case["label"] == "defective":
            mask = load_mask_binary(sample.mask_path)
            image = overlay_mask(image, mask, case.get("bbox"))
        thumb = Image.fromarray(image).resize(thumb_size, Image.Resampling.BILINEAR)
        ax.imshow(thumb)
        ax.set_title(_caption(case, with_area=with_mask), fontsize=7)
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=130)
    plt.close(fig)


def plot_cam_gallery(
    explanations: list[dict[str, Any]],
    cases_by_id: dict[str, dict[str, Any]],
    samples_by_id: dict[str, Sample],
    dest: Path,
    *,
    title: str,
) -> None:
    if not explanations:
        return
    n = len(explanations)
    cols = 4 if n > 4 else n
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.55 * cols, 3.35 * rows + 0.5))
    axes_flat = np.atleast_1d(axes).ravel()
    for ax in axes_flat:
        ax.axis("off")
    for ax, item in zip(axes_flat, explanations):
        sample = samples_by_id[item["sample_id"]]
        case = cases_by_id[item["sample_id"]]
        image = _read_rgb(sample.image_path)
        shown = overlay_cam(image, item["cam_original"], item["mask"])
        thumb = Image.fromarray(shown).resize((96, 240), Image.Resampling.BILINEAR)
        metrics = item["metrics"]
        extra = ""
        if case["label"] == "defective":
            extra = f"  hit={int(metrics['hit'])}"
        ax.imshow(thumb)
        ax.set_title(
            f"{item['sample_id']}  s={case['y_score']:.3f}{extra}",
            fontsize=7,
        )
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=130)
    plt.close(fig)


def plot_area_bars(rows: list[dict[str, Any]], dest: Path) -> None:
    names = [r["area_bin"] for r in rows]
    tp = [r["tp"] for r in rows]
    fn = [r["fn"] for r in rows]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(x - 0.18, tp, 0.36, label="TP")
    ax.bar(x + 0.18, fn, 0.36, label="FN")
    ax.set_xticks(x, names)
    ax.set_ylabel("Test defectives (count)")
    ax.set_title("Official test defectives by mask area, at the val-chosen threshold")
    ax.legend()
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=130)
    plt.close(fig)


def plot_score_strip(
    tps: list[dict[str, Any]],
    fns: list[dict[str, Any]],
    fps: list[dict[str, Any]],
    threshold: float,
    dest: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    rng = np.random.default_rng(0)

    def _strip(cases: list[dict[str, Any]], y: float, color: str, label: str) -> None:
        if not cases:
            return
        scores = np.array([c["y_score"] for c in cases], dtype=np.float64)
        jitter = y + rng.uniform(-0.08, 0.08, size=len(scores))
        ax.scatter(scores, jitter, s=18, c=color, alpha=0.75, label=label)

    _strip(tps, 2.0, "#2ca02c", f"TP n={len(tps)}")
    _strip(fns, 1.0, "#d62728", f"FN n={len(fns)}")
    _strip(fps, 0.0, "#ff7f0e", f"FP n={len(fps)}")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1, label="val threshold")
    ax.set_xlabel("Predicted P(defective)")
    ax.set_yticks([0, 1, 2], ["FP", "FN", "TP"])
    ax.set_xlim(-0.02, 1.02)
    ax.set_title("Official test scores at the frozen Phase 2 checkpoint")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=130)
    plt.close(fig)


def _fmt_frac(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.3f}%"


def pick_fp_examples(
    fps: list[dict[str, Any]],
    *,
    n_confident: int = 12,
    n_borderline: int = 4,
) -> list[dict[str, Any]]:
    ordered = sorted(fps, key=lambda c: c["y_score"], reverse=True)
    confident = ordered[:n_confident]
    taken = {c["sample_id"] for c in confident}
    borderline = [c for c in reversed(ordered) if c["sample_id"] not in taken][
        :n_borderline
    ]
    return confident + list(reversed(borderline))


def pick_tp_examples(
    tps: list[dict[str, Any]],
    *,
    per_bin: int = 2,
) -> list[dict[str, Any]]:
    picked: list[dict[str, Any]] = []
    for _low, _high, name in AREA_BINS:
        bucket = [c for c in tps if c["area_bin"] == name]
        bucket.sort(key=lambda c: c["area_fraction"])
        picked.extend(bucket[:per_bin])
    return picked


def _case_table_row(case: dict[str, Any]) -> str:
    loc = case["location_bin"] or "n/a"
    return (
        f"| {case['sample_id']} | {case['y_score']:.4f} | "
        f"{case['mask_foreground_pixels']} | {_fmt_pct(case['area_fraction'])} | "
        f"{case['area_bin']} | {case['letterbox_mask_pixels']} | {loc} |"
    )


def summarize_cam(explanations: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [item["metrics"] for item in explanations]
    if not rows:
        return {
            "n": 0,
            "n_hit": 0,
            "n_peak_in_mask": 0,
            "n_peak_in_neighborhood": 0,
            "n_cam_empty": 0,
            "median_mask_coverage": None,
            "median_cam_mass_in_mask": None,
        }
    coverages = [r["mask_coverage"] for r in rows if r["mask_coverage"] is not None]
    masses = [r["cam_mass_in_mask"] for r in rows if r["cam_mass_in_mask"] is not None]
    return {
        "n": len(rows),
        "n_hit": sum(1 for r in rows if r["hit"]),
        "n_peak_in_mask": sum(1 for r in rows if r["peak_in_mask"]),
        "n_peak_in_neighborhood": sum(1 for r in rows if r["peak_in_neighborhood"]),
        "n_cam_empty": sum(1 for r in rows if r["cam_empty"]),
        "median_mask_coverage": (
            float(np.median(coverages)) if coverages else None
        ),
        "median_cam_mass_in_mask": float(np.median(masses)) if masses else None,
    }


def write_error_markdown(report: dict[str, Any], dest: Path) -> None:
    test_cm = report["test_confusion"]
    val_cm = report["val_confusion"]
    threshold = report["threshold"]
    tiny = report["tiny_defects"]
    area_rows = report["test_area_stratification"]
    loc_rows = report["test_location_stratification"]
    fns = report["test_fn"]
    fps = report["test_fp"]
    fp_scores = report["test_fp_scores"]
    fn_scores = report["test_fn_scores"]
    cam = report["gradcam"]
    lines = [
        "# Phase 3 — Error analysis",
        "",
        f"Generated (UTC): {report['generated_at_utc']}",
        "",
        "Image-level defective vs ok at the locked val-only threshold "
        "(highest cutoff with at least 46/49 val defectives). "
        "Official test was not used to pick the threshold. Masks are used "
        "here for size, location, and Grad-CAM overlap only.",
        "",
        "## Setup",
        "",
        f"- Checkpoint: Phase 2 `reports/baseline/checkpoints/best.pt` "
        f"(epoch {report['best_epoch']}, val PR-AUC {report['best_val_pr_auc']:.4f})",
        f"- Threshold (val only, not retuned): {threshold:.6f}",
        f"- Device: {report['device']}",
        "- Grad-CAM: last `features` map of MobileNetV3-Small; backward on the defective logit",
        f"- High-CAM region: pixels ≥ {report['cam_bin_fraction']:g} × max(CAM)",
        f"- Peak neighborhood: {report['neighborhood_radius_px']} px dilation of the official mask",
        "",
        "## Confusion at the frozen operating point",
        "",
        "Recomputed from saved Phase 2 scores (`predictions_*.json`), not from a retrain.",
        "",
        f"- Test: TP {test_cm['TP']}, FP {test_cm['FP']}, FN {test_cm['FN']}, "
        f"TN {test_cm['TN']}",
        f"- Val (context): TP {val_cm['TP']}, FP {val_cm['FP']}, FN {val_cm['FN']}, "
        f"TN {val_cm['TN']}",
        "",
        "These match `reports/baseline/metrics.md`.",
        "",
        "## False negatives on official test (expensive)",
        "",
        f"n = {len(fns)}. Scores: min={fn_scores['min']:.4f}, "
        f"median={fn_scores['median']:.4f}, max={fn_scores['max']:.4f}.",
        f"Near-threshold FNs (score ≥ threshold − 0.05): "
        f"{report['test_fn_near_threshold']}.",
        "",
        "| ID | Score | Mask px | Area | Area bin | Letterbox mask px | Location |",
        "|---|---:|---:|---:|---|---:|---|",
    ]
    for case in sorted(fns, key=lambda c: c["area_fraction"]):
        lines.append(_case_table_row(case))
    lines.extend(
        [
            "",
            "Location is the mask centroid along image height (top / middle / bottom thirds). "
            "Letterbox mask px is the official mask after the same 224×448 letterbox the model sees.",
            "",
            "## Are the misses the tiny defects?",
            "",
            f"Phase 1 audit: {tiny['n_dataset']} defective masks in the whole zip "
            f"cover less than {100.0 * TINY_AREA_CUTOFF:.1f}% of the image.",
            f"Of those, {tiny['n_test']} official test, {tiny['n_val']} val, "
            f"{tiny['n_train']} train-role.",
            "Val/test scores are from the saved Phase 2 prediction files. "
            "Train-role tiny-defect scores were computed from the same frozen "
            "checkpoint (there is no `predictions_train.json`).",
            "",
            "| ID | Split | Role | Mask px | Area | Letterbox px | Score | Error |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in tiny["rows"]:
        score = "n/a" if row["y_score"] is None else f"{row['y_score']:.4f}"
        lines.append(
            f"| {row['sample_id']} | {row['official_split']} | {row['role']} | "
            f"{row['mask_foreground_pixels']} | {_fmt_pct(row['area_fraction'])} | "
            f"{row['letterbox_mask_pixels']} | {score} | {row['error_type']} |"
        )
    lines.extend(
        [
            "",
            f"Test defectives with area < 0.1%: {tiny['n_test']}. "
            f"Among them, FN = {tiny['n_test_fn']}, TP = {tiny['n_test_tp']}.",
            f"Of the {len(fns)} test FNs, {report['test_fn_tiny']} "
            f"{'has' if report['test_fn_tiny'] == 1 else 'have'} area < 0.1%.",
            "",
            "### Test defectives by mask area",
            "",
            "| Area bin | n | TP | FN | FN rate |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in area_rows:
        lines.append(
            f"| {row['area_bin']} | {row['n']} | {row['tp']} | {row['fn']} | "
            f"{_fmt_frac(row['fn_rate'])} |"
        )
    lines.extend(
        [
            "",
            "### Test defectives by mask centroid (height thirds)",
            "",
            "| Location | n | TP | FN | FN rate |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in loc_rows:
        lines.append(
            f"| {row['location_bin']} | {row['n']} | {row['tp']} | {row['fn']} | "
            f"{_fmt_frac(row['fn_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## False positives on official test (cheaper, numerous)",
            "",
            f"n = {len(fps)}. Official GT is empty. Scores: min={fp_scores['min']:.4f}, "
            f"median={fp_scores['median']:.4f}, max={fp_scores['max']:.4f}.",
            f"FPs with score in [{threshold:.3f}, 0.5): {report['test_fp_between_chosen_and_half']}.",
            f"FPs with score ≥ 0.8: {report['test_fp_high_conf']}.",
            "",
            "Highest-score FPs (first 12): "
            + ", ".join(
                f"{c['sample_id']} ({c['y_score']:.3f})"
                for c in sorted(fps, key=lambda x: x["y_score"], reverse=True)[:12]
            )
            + ".",
            "",
            "## Grad-CAM (same checkpoint)",
            "",
            "CAM is produced at the last backbone feature map "
            f"({cam['feature_hw'][0]}×{cam['feature_hw'][1]} for a 224×448 input) "
            "and upsampled. Overlap is measured on the original image after "
            "un-letterboxing. A coarse map makes pixel IoU pessimistic for "
            "tiny marks; hit / peak-in-neighborhood are the counts to read first.",
            "",
            f"- Test FN (all {cam['fn']['n']}): high-CAM overlaps mask in "
            f"{cam['fn']['n_hit']}/{cam['fn']['n']}; peak in mask "
            f"{cam['fn']['n_peak_in_mask']}/{cam['fn']['n']}; peak in "
            f"{report['neighborhood_radius_px']}px neighborhood "
            f"{cam['fn']['n_peak_in_neighborhood']}/{cam['fn']['n']}; "
            f"empty CAM {cam['fn']['n_cam_empty']}/{cam['fn']['n']}; "
            f"median mask coverage {_fmt_frac(cam['fn']['median_mask_coverage'])}; "
            f"median CAM mass in mask {_fmt_frac(cam['fn']['median_cam_mass_in_mask'])}.",
            f"- Test TP sample (n={cam['tp']['n']}, up to 2 per area bin): "
            f"overlap {cam['tp']['n_hit']}/{cam['tp']['n']}; peak in mask "
            f"{cam['tp']['n_peak_in_mask']}/{cam['tp']['n']}; peak in neighborhood "
            f"{cam['tp']['n_peak_in_neighborhood']}/{cam['tp']['n']}; "
            f"median mask coverage {_fmt_frac(cam['tp']['median_mask_coverage'])}; "
            f"median CAM mass in mask {_fmt_frac(cam['tp']['median_cam_mass_in_mask'])}.",
            f"- Test FP sample (n={cam['fp']['n']}; 12 highest-score + 4 nearest threshold): "
            f"empty CAM {cam['fp']['n_cam_empty']}/{cam['fp']['n']}. "
            "No GT region exists; peak height-third counts: "
            + ", ".join(
                f"{name}={cam['fp_peak_location'].get(name, 0)}"
                for name in ("top", "middle", "bottom")
            )
            + ".",
            "",
            "Per-example Grad-CAM rows (FN, then sampled TP, then sampled FP) are in "
            "`error_report.json` under `gradcam.examples`.",
            "",
            "## Files",
            "",
            "- `error_report.json` — computed tables",
            "- `figures/gallery_fn_test.png` — all test FNs with GT overlay "
            "(yellow box is padded 8 px for visibility)",
            "- `figures/gallery_fp_test.png` — all test FPs (empty GT)",
            "- `figures/gallery_gradcam_{fn,tp,fp}.png`",
            "- `figures/area_stratification.png`, `figures/score_strip.png`",
            "",
        ]
    )
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_error_report_bundle(report: dict[str, Any], output_dir: Path) -> None:
    write_json(output_dir / "error_report.json", report)
    write_error_markdown(report, output_dir / "error_report.md")
