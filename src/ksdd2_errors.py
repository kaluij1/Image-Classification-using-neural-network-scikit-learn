"""Join Phase 2 predictions with official masks for error analysis.

Masks are used for size and location only. They are not a training
target here and are not used to retune the threshold.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from ksdd2_inventory import Sample
from ksdd2_transforms import INPUT_HEIGHT, INPUT_WIDTH, LetterboxGeom, letterbox_geometry

# Same 0.1% / 0.5% cutoffs as the Phase 1 audit, then round bins above that.
AREA_BINS: tuple[tuple[float, float, str], ...] = (
    (0.0, 0.001, "<0.1%"),
    (0.001, 0.005, "0.1–0.5%"),
    (0.005, 0.02, "0.5–2%"),
    (0.02, 0.10, "2–10%"),
    (0.10, 1.01, "≥10%"),
)
TINY_AREA_CUTOFF = 0.001
CAM_NEIGHBORHOOD_RADIUS_PX = 16


def load_mask_binary(path) -> np.ndarray:
    with Image.open(path) as mask:
        arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr.max(axis=2)
    return arr > 0


def letterbox_mask(mask: np.ndarray, geom: LetterboxGeom) -> np.ndarray:
    image = Image.fromarray((mask.astype(np.uint8)) * 255)
    resized = image.resize((geom.new_w, geom.new_h), Image.Resampling.NEAREST)
    canvas = np.zeros((geom.dst_h, geom.dst_w), dtype=bool)
    canvas[
        geom.pad_y : geom.pad_y + geom.new_h,
        geom.pad_x : geom.pad_x + geom.new_w,
    ] = np.asarray(resized) > 0
    return canvas


def unletterbox_map(values: np.ndarray, geom: LetterboxGeom) -> np.ndarray:
    content = values[
        geom.pad_y : geom.pad_y + geom.new_h,
        geom.pad_x : geom.pad_x + geom.new_w,
    ]
    image = Image.fromarray(np.asarray(content, dtype=np.float32), mode="F")
    resized = image.resize((geom.src_w, geom.src_h), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def area_bin(area_fraction: float) -> str:
    for low, high, name in AREA_BINS:
        if low <= area_fraction < high:
            return name
    return AREA_BINS[-1][2]


def location_bin(centroid_y_frac: float | None) -> str | None:
    if centroid_y_frac is None:
        return None
    if centroid_y_frac < 1.0 / 3.0:
        return "top"
    if centroid_y_frac < 2.0 / 3.0:
        return "middle"
    return "bottom"


def error_type(y_true: int, y_pred: int) -> str:
    if y_true == 1 and y_pred == 1:
        return "TP"
    if y_true == 0 and y_pred == 1:
        return "FP"
    if y_true == 1 and y_pred == 0:
        return "FN"
    return "TN"


def mask_geometry(
    sample: Sample,
    *,
    input_width: int = INPUT_WIDTH,
    input_height: int = INPUT_HEIGHT,
) -> dict[str, Any]:
    mask = load_mask_binary(sample.mask_path)
    pixels = int(mask.sum())
    area_fraction = pixels / float(sample.width * sample.height)
    geom = letterbox_geometry(sample.width, sample.height, input_width, input_height)
    letterbox_pixels = int(letterbox_mask(mask, geom).sum())
    payload: dict[str, Any] = {
        "mask_foreground_pixels": pixels,
        "area_fraction": float(area_fraction),
        "area_bin": area_bin(area_fraction) if pixels > 0 else "none",
        "letterbox_mask_pixels": letterbox_pixels,
        "centroid_y_frac": None,
        "centroid_x_frac": None,
        "location_bin": None,
        "bbox": None,
    }
    if pixels == 0:
        return payload
    ys, xs = np.nonzero(mask)
    cy = float(ys.mean() / sample.height)
    cx = float(xs.mean() / sample.width)
    payload["centroid_y_frac"] = cy
    payload["centroid_x_frac"] = cx
    payload["location_bin"] = location_bin(cy)
    payload["bbox"] = {
        "ymin": int(ys.min()),
        "ymax": int(ys.max()),
        "xmin": int(xs.min()),
        "xmax": int(xs.max()),
    }
    return payload


def _prediction_index(payload: dict[str, Any]) -> dict[str, tuple[int, float]]:
    index: dict[str, tuple[int, float]] = {}
    for sample_id, y_true, y_score in zip(
        payload["sample_ids"], payload["y_true"], payload["y_score"]
    ):
        index[str(sample_id)] = (int(y_true), float(y_score))
    return index


def build_cases(
    samples: list[Sample],
    manifest: dict[str, Any],
    predictions_by_role: dict[str, dict[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
    role_of: dict[tuple[str, str], str] = {}
    for role in ("train", "val", "test"):
        for row in manifest[role]:
            role_of[(row["official_split"], row["sample_id"])] = role

    pred_index = {
        role: _prediction_index(payload)
        for role, payload in predictions_by_role.items()
    }
    cases: list[dict[str, Any]] = []
    for sample in samples:
        role = role_of[(sample.split, sample.sample_id)]
        if role not in pred_index or sample.sample_id not in pred_index[role]:
            continue
        y_true, y_score = pred_index[role][sample.sample_id]
        y_pred = int(y_score >= threshold)
        geometry = mask_geometry(sample)
        if y_true != (1 if sample.label == "defective" else 0):
            raise ValueError(
                f"Label mismatch for {sample.split}/{sample.sample_id}: "
                f"prediction y_true={y_true}, inventory={sample.label}"
            )
        cases.append(
            {
                "sample_id": sample.sample_id,
                "official_split": sample.split,
                "role": role,
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


def filter_cases(
    cases: list[dict[str, Any]],
    *,
    role: str | None = None,
    error: str | None = None,
) -> list[dict[str, Any]]:
    out = cases
    if role is not None:
        out = [c for c in out if c["role"] == role]
    if error is not None:
        out = [c for c in out if c["error_type"] == error]
    return out


def confusion_counts(cases: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for case in cases:
        counts[case["error_type"]] += 1
    return counts


def stratify_by_area(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[case["area_bin"]].append(case)
    rows: list[dict[str, Any]] = []
    for _low, _high, name in AREA_BINS:
        bucket = grouped.get(name, [])
        n_fn = sum(1 for c in bucket if c["error_type"] == "FN")
        n_tp = sum(1 for c in bucket if c["error_type"] == "TP")
        rows.append(
            {
                "area_bin": name,
                "n": len(bucket),
                "tp": n_tp,
                "fn": n_fn,
                "fn_rate": (n_fn / len(bucket)) if bucket else None,
            }
        )
    return rows


def stratify_by_location(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in ("top", "middle", "bottom"):
        bucket = [c for c in cases if c["location_bin"] == name]
        n_fn = sum(1 for c in bucket if c["error_type"] == "FN")
        n_tp = sum(1 for c in bucket if c["error_type"] == "TP")
        rows.append(
            {
                "location_bin": name,
                "n": len(bucket),
                "tp": n_tp,
                "fn": n_fn,
                "fn_rate": (n_fn / len(bucket)) if bucket else None,
            }
        )
    return rows


def score_summary(cases: list[dict[str, Any]]) -> dict[str, float | int | None]:
    if not cases:
        return {"n": 0, "min": None, "median": None, "max": None}
    scores = np.array([c["y_score"] for c in cases], dtype=np.float64)
    return {
        "n": int(len(cases)),
        "min": float(scores.min()),
        "median": float(np.median(scores)),
        "max": float(scores.max()),
    }


def tiny_defect_rows(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tiny = [
        c
        for c in cases
        if c["label"] == "defective" and c["area_fraction"] < TINY_AREA_CUTOFF
    ]
    tiny.sort(key=lambda c: (c["official_split"], int(c["sample_id"])))
    return [
        {
            "sample_id": c["sample_id"],
            "official_split": c["official_split"],
            "role": c["role"],
            "area_fraction": c["area_fraction"],
            "mask_foreground_pixels": c["mask_foreground_pixels"],
            "letterbox_mask_pixels": c["letterbox_mask_pixels"],
            "error_type": c["error_type"],
            "y_score": c["y_score"],
        }
        for c in tiny
    ]


def dilate_binary(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    size = 2 * radius + 1
    image = Image.fromarray((mask.astype(np.uint8)) * 255)
    dilated = image.filter(ImageFilter.MaxFilter(size))
    return np.asarray(dilated) > 0
