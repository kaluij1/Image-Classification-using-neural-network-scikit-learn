"""Reproducible train/val split carved from official KSDD2 train only.

Official test is frozen. Validation is stratified by the image-level
label (defective vs ok). Test images, test masks, and test-derived
statistics are never used here.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split

from ksdd2_inventory import Sample

SEED = 42
VAL_FRACTION = 0.2
ROLES = ("train", "val", "test")


def _key(official_split: str, sample_id: str) -> str:
    return f"{official_split}/{sample_id}"


def make_split_manifest(
    samples: list[Sample],
    seed: int = SEED,
    val_fraction: float = VAL_FRACTION,
) -> dict[str, Any]:
    official_train = [s for s in samples if s.split == "train"]
    official_test = [s for s in samples if s.split == "test"]
    if not official_train or not official_test:
        raise ValueError("Inventory must include both official train and test samples.")

    train_ids = [s.sample_id for s in official_train]
    train_labels = [s.label for s in official_train]
    fit_ids, val_ids = train_test_split(
        train_ids,
        test_size=val_fraction,
        random_state=seed,
        stratify=train_labels,
    )

    by_official: dict[str, dict[str, Sample]] = {"train": {}, "test": {}}
    for sample in samples:
        by_official[sample.split][sample.sample_id] = sample

    def _rows(official_split: str, ids: list[str], role: str) -> list[dict[str, str]]:
        rows = []
        for sample_id in sorted(ids, key=int):
            sample = by_official[official_split][sample_id]
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "official_split": official_split,
                    "role": role,
                    "label": sample.label,
                }
            )
        return rows

    manifest = {
        "seed": seed,
        "val_fraction": val_fraction,
        "strategy": "stratified_on_official_train",
        "positive_class": "defective",
        "train": _rows("train", fit_ids, "train"),
        "val": _rows("train", val_ids, "val"),
        "test": _rows("test", [s.sample_id for s in official_test], "test"),
    }
    validate_manifest(manifest, samples)
    return manifest


def validate_manifest(manifest: dict[str, Any], samples: list[Sample]) -> None:
    inventory_keys = {_key(s.split, s.sample_id) for s in samples}
    role_keys: dict[str, set[str]] = {}
    for role in ROLES:
        rows = manifest[role]
        keys = {_key(row["official_split"], row["sample_id"]) for row in rows}
        if len(keys) != len(rows):
            raise ValueError(f"Duplicate IDs in role {role}")
        unknown = keys - inventory_keys
        if unknown:
            raise ValueError(f"Manifest IDs not in inventory ({role}): {sorted(unknown)[:5]}")
        role_keys[role] = keys

    if not role_keys["train"].isdisjoint(role_keys["val"]):
        raise ValueError("Leakage: train and val share sample keys")
    if not role_keys["train"].isdisjoint(role_keys["test"]):
        raise ValueError("Leakage: train and test share sample keys")
    if not role_keys["val"].isdisjoint(role_keys["test"]):
        raise ValueError("Leakage: val and test share sample keys")

    for row in manifest["train"] + manifest["val"]:
        if row["official_split"] != "train":
            raise ValueError("Train/val must be carved from official train only")
    for row in manifest["test"]:
        if row["official_split"] != "test":
            raise ValueError("Test role must be the official test split")

    union = role_keys["train"] | role_keys["val"] | role_keys["test"]
    if union != inventory_keys:
        raise ValueError("Manifest does not cover the canonical inventory exactly")


def manifest_counts(manifest: dict[str, Any]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for role in ROLES:
        labels = Counter(row["label"] for row in manifest[role])
        counts[role] = {
            "n": len(manifest[role]),
            "defective": labels.get("defective", 0),
            "ok": labels.get("ok", 0),
        }
    return counts


def samples_by_role(
    samples: list[Sample],
    manifest: dict[str, Any],
) -> dict[str, list[Sample]]:
    lookup = {_key(s.split, s.sample_id): s for s in samples}
    grouped: dict[str, list[Sample]] = {role: [] for role in ROLES}
    for role in ROLES:
        for row in manifest[role]:
            grouped[role].append(lookup[_key(row["official_split"], row["sample_id"])])
    return grouped


def save_manifest(manifest: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
