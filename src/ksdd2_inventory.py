"""Discover KolektorSDD2 image/mask pairs and derive image-level labels.

Official release layout (ViCoS zip):

    data/raw/KolektorSDD2/
        train/{id}.png
        train/{id}_GT.png
        test/{id}.png
        test/{id}_GT.png

A sample is labeled defective if its ground-truth mask contains any
nonzero pixel; otherwise it is labeled ok. This matches the official
image-level positive/negative definition.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
import numpy as np

IMAGE_SUFFIX = ".png"
MASK_SUFFIX = "_GT.png"
SPLITS = ("train", "test")
SAMPLE_ID_RE = re.compile(r"^\d+$")
EXPECTED_FILE_RE = re.compile(r"^\d+(_GT)?\.png$")


@dataclass(frozen=True)
class Sample:
    sample_id: str
    split: str
    image_path: Path
    mask_path: Path
    label: str
    mask_foreground_pixels: int
    width: int
    height: int
    channels: int
    mode: str


def find_dataset_root(start: Path | None = None) -> Path:
    candidates: list[Path] = []
    if start is not None:
        candidates.append(start)
    here = Path(__file__).resolve().parents[1]
    candidates.extend(
        [
            here / "data" / "raw" / "KolektorSDD2",
            here / "data" / "raw",
        ]
    )
    for path in candidates:
        if (path / "train").is_dir() and (path / "test").is_dir():
            return path
        nested = path / "KolektorSDD2"
        if (nested / "train").is_dir() and (nested / "test").is_dir():
            return nested
    raise FileNotFoundError(
        "KolektorSDD2 root with train/ and test/ not found. "
        "Run scripts/download_ksdd2.py first."
    )


def _mask_foreground_pixels(mask_path: Path) -> int:
    with Image.open(mask_path) as mask:
        arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr.max(axis=2)
    return int(np.count_nonzero(arr))


def _image_geometry(image_path: Path) -> tuple[int, int, int, str]:
    with Image.open(image_path) as image:
        image.load()
        width, height = image.size
        mode = image.mode
        channels = len(image.getbands())
    return width, height, channels, mode


def iter_split_ids(split_dir: Path) -> list[str]:
    ids: list[str] = []
    for path in sorted(split_dir.glob(f"*{IMAGE_SUFFIX}")):
        if path.name.endswith(MASK_SUFFIX):
            continue
        if SAMPLE_ID_RE.match(path.stem):
            ids.append(path.stem)
    return ids


def unexpected_files(root: Path) -> list[str]:
    extra: list[str] = []
    for split in SPLITS:
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        for path in sorted(split_dir.iterdir()):
            if path.is_file() and not EXPECTED_FILE_RE.match(path.name):
                extra.append(str(path.relative_to(root)))
    return extra


def load_inventory(root: Path | None = None) -> list[Sample]:
    root = find_dataset_root(root)
    samples: list[Sample] = []
    missing_masks: list[str] = []
    for split in SPLITS:
        split_dir = root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")
        for sample_id in iter_split_ids(split_dir):
            image_path = split_dir / f"{sample_id}{IMAGE_SUFFIX}"
            mask_path = split_dir / f"{sample_id}{MASK_SUFFIX}"
            if not mask_path.is_file():
                missing_masks.append(str(image_path))
                continue
            fg = _mask_foreground_pixels(mask_path)
            width, height, channels, mode = _image_geometry(image_path)
            samples.append(
                Sample(
                    sample_id=sample_id,
                    split=split,
                    image_path=image_path,
                    mask_path=mask_path,
                    label="defective" if fg > 0 else "ok",
                    mask_foreground_pixels=fg,
                    width=width,
                    height=height,
                    channels=channels,
                    mode=mode,
                )
            )
    if missing_masks:
        raise FileNotFoundError(
            "Missing masks for: " + ", ".join(missing_masks[:10])
        )
    return samples
