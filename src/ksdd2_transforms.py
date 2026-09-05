"""Preprocess and light augmentation for the Phase 2 baseline.

Images are tall and variable (median 229x637, 601 shapes). A square
224x224 resize would squash height by ~3x and can erase the smallest
defects (8 masks cover <0.1% of the image). Letterboxing to a portrait
canvas keeps aspect ratio.

Augmentation is conservative: these are fixture-captured industrial
photos, not an in-the-wild set. No random crop — a crop can drop a
23-pixel defect entirely. No vertical flip — part orientation in the
fixture is treated as meaningful.
"""

from __future__ import annotations

from PIL import Image
from torchvision import transforms

# PIL size is (width, height). Matches the portrait capture roughly 1:2.
INPUT_WIDTH = 224
INPUT_HEIGHT = 448

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_PAD_RGB = tuple(int(round(c * 255)) for c in IMAGENET_MEAN)


class Letterbox:
    """Resize with aspect ratio preserved, pad to a fixed canvas."""

    def __init__(
        self,
        width: int = INPUT_WIDTH,
        height: int = INPUT_HEIGHT,
        fill: tuple[int, int, int] = _PAD_RGB,
    ) -> None:
        self.width = width
        self.height = height
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        src_w, src_h = image.size
        scale = min(self.width / src_w, self.height / src_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (self.width, self.height), self.fill)
        canvas.paste(resized, ((self.width - new_w) // 2, (self.height - new_h) // 2))
        return canvas


def train_transforms(
    width: int = INPUT_WIDTH,
    height: int = INPUT_HEIGHT,
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=4, fill=_PAD_RGB),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.08),
            Letterbox(width=width, height=height),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def eval_transforms(
    width: int = INPUT_WIDTH,
    height: int = INPUT_HEIGHT,
) -> transforms.Compose:
    return transforms.Compose(
        [
            Letterbox(width=width, height=height),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
