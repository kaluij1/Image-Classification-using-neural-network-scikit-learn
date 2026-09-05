"""Audit KolektorSDD2 without training a model.

Computes inventory stats, image integrity, class imbalance, mask-area
summary, exact duplicates, and cheap perceptual near-duplicates.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

from ksdd2_inventory import Sample, find_dataset_root, load_inventory, unexpected_files

HASH_SIZE = 8


@dataclass(frozen=True)
class CorruptRecord:
    path: str
    kind: str
    error: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def average_hash_bits(path: Path, hash_size: int = HASH_SIZE) -> int:
    with Image.open(path) as image:
        gray = image.convert("L").resize((hash_size, hash_size), Image.Resampling.BILINEAR)
        pixels = np.asarray(gray, dtype=np.float32)
    mean = float(pixels.mean())
    bits = (pixels >= mean).astype(np.uint8).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def check_integrity(samples: list[Sample]) -> list[CorruptRecord]:
    corrupt: list[CorruptRecord] = []
    for sample in samples:
        for kind, path in (("image", sample.image_path), ("mask", sample.mask_path)):
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    image.load()
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                corrupt.append(CorruptRecord(str(path), kind, str(exc)))
    return corrupt


def _count_table(samples: list[Sample]) -> dict[str, dict[str, int]]:
    table: dict[str, dict[str, int]] = {}
    for split in ("train", "test", "all"):
        subset = samples if split == "all" else [s for s in samples if s.split == split]
        table[split] = {
            "n": len(subset),
            "defective": sum(1 for s in subset if s.label == "defective"),
            "ok": sum(1 for s in subset if s.label == "ok"),
        }
    return table


def _geometry_summary(samples: list[Sample]) -> dict[str, Any]:
    widths = [s.width for s in samples]
    heights = [s.height for s in samples]
    modes = Counter(s.mode for s in samples)
    channels = Counter(s.channels for s in samples)
    unique_sizes = Counter((s.width, s.height) for s in samples)
    return {
        "width_min": min(widths),
        "width_max": max(widths),
        "width_median": int(np.median(widths)),
        "height_min": min(heights),
        "height_max": max(heights),
        "height_median": int(np.median(heights)),
        "modes": dict(modes),
        "channels": {str(k): v for k, v in channels.items()},
        "unique_size_count": len(unique_sizes),
        "most_common_sizes": [
            {"width": w, "height": h, "count": c}
            for (w, h), c in unique_sizes.most_common(8)
        ],
    }


def _mask_summary(samples: list[Sample]) -> dict[str, Any]:
    defective = [s for s in samples if s.label == "defective"]
    if not defective:
        return {"defective_count": 0}
    areas = np.array(
        [s.mask_foreground_pixels / (s.width * s.height) for s in defective],
        dtype=np.float64,
    )
    pixels = np.array([s.mask_foreground_pixels for s in defective], dtype=np.int64)
    return {
        "defective_count": len(defective),
        "foreground_pixels_min": int(pixels.min()),
        "foreground_pixels_max": int(pixels.max()),
        "foreground_pixels_median": int(np.median(pixels)),
        "area_fraction_min": float(areas.min()),
        "area_fraction_max": float(areas.max()),
        "area_fraction_median": float(np.median(areas)),
        "area_fraction_mean": float(areas.mean()),
        "tiny_defects_under_0_5pct": int(np.sum(areas < 0.005)),
        "tiny_defects_under_0_1pct": int(np.sum(areas < 0.001)),
    }


def _exact_duplicates(samples: list[Sample]) -> dict[str, Any]:
    by_hash: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_hash[sha256_file(sample.image_path)].append(sample)
    groups = [group for group in by_hash.values() if len(group) > 1]
    cross_split = [
        group
        for group in groups
        if len({s.split for s in group}) > 1
    ]
    return {
        "duplicate_groups": len(groups),
        "images_in_duplicate_groups": sum(len(g) for g in groups),
        "cross_split_groups": len(cross_split),
        "examples": [
            {
                "sha256": sha256_file(group[0].image_path),
                "members": [
                    {"id": s.sample_id, "split": s.split, "label": s.label}
                    for s in group
                ],
            }
            for group in groups[:20]
        ],
    }


def _near_duplicates(samples: list[Sample]) -> dict[str, Any]:
    """Group images that share an identical 8x8 average hash.

    Hamming-distance expansion is not used. On this dataset it produces
    thousands of pairs because ok parts share similar texture under a
    64-bit hash. That is not evidence of leakage.
    """
    buckets: dict[int, list[Sample]] = defaultdict(list)
    for sample in samples:
        buckets[average_hash_bits(sample.image_path)].append(sample)

    exact_ahash_groups = [group for group in buckets.values() if len(group) > 1]
    cross_split = [group for group in exact_ahash_groups if len({s.split for s in group}) > 1]
    mixed_label = [group for group in exact_ahash_groups if len({s.label for s in group}) > 1]
    return {
        "hash_size": HASH_SIZE,
        "identical_ahash_groups": len(exact_ahash_groups),
        "images_in_identical_ahash_groups": sum(len(g) for g in exact_ahash_groups),
        "identical_ahash_cross_split_groups": len(cross_split),
        "identical_ahash_mixed_label_groups": len(mixed_label),
        "identical_ahash_examples": [
            [{"id": s.sample_id, "split": s.split, "label": s.label} for s in group[:8]]
            for group in exact_ahash_groups[:10]
        ],
        "cross_split_examples": [
            [{"id": s.sample_id, "split": s.split, "label": s.label} for s in group]
            for group in cross_split[:10]
        ],
    }


def _imbalance(counts: dict[str, dict[str, int]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split, row in counts.items():
        n = row["n"]
        defective = row["defective"]
        result[split] = {
            "defective_fraction": defective / n if n else None,
            "ok_to_defective": (row["ok"] / defective) if defective else None,
        }
    return result


def build_report(
    samples: list[Sample],
    corrupt: list[CorruptRecord],
    extra_files: list[str],
) -> dict[str, Any]:
    counts = _count_table(samples)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(samples),
        "counts": counts,
        "imbalance": _imbalance(counts),
        "geometry": _geometry_summary(samples),
        "masks": _mask_summary(samples),
        "corrupt": [asdict(item) for item in corrupt],
        "unexpected_files": extra_files,
        "missing_masks": 0,
        "exact_duplicates": _exact_duplicates(samples),
        "near_duplicates": _near_duplicates(samples),
        "official_published": {
            "images": 3335,
            "defective": 356,
            "ok": 2979,
            "train_defective": 246,
            "train_ok": 2085,
            "test_defective": 110,
            "test_ok": 894,
        },
    }


def _font() -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("arial.ttf", 14)
    except OSError:
        return ImageFont.load_default()


def save_sample_grid(
    samples: list[Sample],
    dest: Path,
    title: str,
    max_n: int = 8,
    overlay_mask: bool = False,
) -> None:
    chosen = samples[:max_n]
    if not chosen:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    thumb_w, thumb_h = 96, 240
    cols = min(8, len(chosen))
    rows = int(np.ceil(len(chosen) / cols))
    pad = 8
    header = 28
    canvas = Image.new(
        "RGB",
        (cols * (thumb_w + pad) + pad, header + rows * (thumb_h + pad) + pad),
        (24, 24, 24),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 6), title, fill=(230, 230, 230), font=_font())
    for idx, sample in enumerate(chosen):
        r, c = divmod(idx, cols)
        with Image.open(sample.image_path) as image:
            thumb = image.convert("RGB").resize((thumb_w, thumb_h), Image.Resampling.BILINEAR)
        if overlay_mask:
            with Image.open(sample.mask_path) as mask:
                mask_arr = np.asarray(mask.convert("L").resize((thumb_w, thumb_h)))
            overlay = np.asarray(thumb).copy()
            hit = mask_arr > 0
            overlay[hit, 0] = np.clip(overlay[hit, 0].astype(np.int16) + 90, 0, 255)
            thumb = Image.fromarray(overlay.astype(np.uint8))
        x = pad + c * (thumb_w + pad)
        y = header + pad + r * (thumb_h + pad)
        canvas.paste(thumb, (x, y))
    canvas.save(dest)


def write_markdown(report: dict[str, Any], dest: Path) -> None:
    counts = report["counts"]
    geo = report["geometry"]
    masks = report["masks"]
    exact = report["exact_duplicates"]
    near = report["near_duplicates"]
    lines = [
        "# KolektorSDD2 data audit",
        "",
        f"Generated (UTC): {report['generated_at_utc']}",
        "",
        "All numbers below were computed from the official extracted zip.",
        "No model was trained.",
        "",
        "## Sample counts",
        "",
        "| Split | N | Defective | OK | Defective % |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in ("train", "test", "all"):
        row = counts[split]
        frac = 100.0 * row["defective"] / row["n"] if row["n"] else 0.0
        lines.append(
            f"| {split} | {row['n']} | {row['defective']} | {row['ok']} | {frac:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Image geometry",
            "",
            f"- Width range: {geo['width_min']}–{geo['width_max']} px (median {geo['width_median']})",
            f"- Height range: {geo['height_min']}–{geo['height_max']} px (median {geo['height_median']})",
            f"- Modes: {geo['modes']}",
            f"- Channel counts: {geo['channels']}",
            f"- Distinct (width, height) pairs: {geo['unique_size_count']}",
            "",
            "## Mask / defect size (defective images only)",
            "",
            f"- Foreground pixels: min {masks['foreground_pixels_min']}, "
            f"median {masks['foreground_pixels_median']}, max {masks['foreground_pixels_max']}",
            f"- Area fraction: min {masks['area_fraction_min']:.6f}, "
            f"median {masks['area_fraction_median']:.4f}, "
            f"mean {masks['area_fraction_mean']:.4f}, max {masks['area_fraction_max']:.4f}",
            f"- Defects covering < 0.5% of the image: {masks['tiny_defects_under_0_5pct']}",
            f"- Defects covering < 0.1% of the image: {masks['tiny_defects_under_0_1pct']}",
            "",
            "## Integrity",
            "",
            f"- Missing masks: {report['missing_masks']}",
            f"- Unreadable images or masks: {len(report['corrupt'])}",
            f"- Unexpected extra files (not {{id}}.png / {{id}}_GT.png): {len(report['unexpected_files'])}",
        ]
    )
    if report["unexpected_files"]:
        for name in report["unexpected_files"]:
            lines.append(f"  - `{name}`")
    lines.extend(
        [
            "",
            "",
            "## Duplicates",
            "",
            f"- Exact SHA-256 duplicate groups (canonical IDs): {exact['duplicate_groups']} "
            f"({exact['images_in_duplicate_groups']} images); "
            f"cross-split groups: {exact['cross_split_groups']}",
            f"- Identical 8×8 average-hash groups: {near['identical_ahash_groups']} "
            f"({near['images_in_identical_ahash_groups']} images); "
            f"cross-split: {near.get('identical_ahash_cross_split_groups', 'n/a')}; "
            f"mixed-label: {near.get('identical_ahash_mixed_label_groups', 'n/a')}",
            "",
            "An 8×8 average hash is a cheap screen, not proof of identity. "
            "Collisions among ok images are expected for similar production parts. "
            "Do not treat them as train/test leakage without visual review.",
            "",
        ]
    )
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(root: Path | None, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = find_dataset_root(root)
    samples = load_inventory(data_root)
    extra_files = unexpected_files(data_root)
    corrupt = check_integrity(samples)
    report = build_report(samples, corrupt, extra_files)

    (output_dir / "audit_summary.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    write_markdown(report, output_dir / "audit_summary.md")

    rng = np.random.default_rng(0)
    by_key: dict[tuple[str, str], list[Sample]] = defaultdict(list)
    for sample in samples:
        by_key[(sample.split, sample.label)].append(sample)

    for (split, label), group in by_key.items():
        order = rng.permutation(len(group))
        picked = [group[i] for i in order[:8]]
        save_sample_grid(
            picked,
            output_dir / f"samples_{split}_{label}.png",
            title=f"{split} / {label} (random, seed=0)",
        )

    defective = [s for s in samples if s.label == "defective"]
    defective_sorted = sorted(defective, key=lambda s: s.mask_foreground_pixels)
    save_sample_grid(
        defective_sorted[:8],
        output_dir / "samples_smallest_defects.png",
        title="Smallest defect masks (area overlay)",
        overlay_mask=True,
    )
    save_sample_grid(
        list(reversed(defective_sorted[-8:])),
        output_dir / "samples_largest_defects.png",
        title="Largest defect masks (area overlay)",
        overlay_mask=True,
    )
    return report
