"""Image-level KSDD2 dataset. Masks are not the training target."""

from __future__ import annotations

from typing import Any, Callable

import torch
from PIL import Image
from torch.utils.data import Dataset

from ksdd2_inventory import Sample

LABEL_TO_INDEX = {"ok": 0, "defective": 1}


class KSDD2ClassificationDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image) if self.transform is not None else image
        return {
            "image": tensor,
            "label": torch.tensor(LABEL_TO_INDEX[sample.label], dtype=torch.float32),
            "sample_id": sample.sample_id,
            "official_split": sample.split,
        }
