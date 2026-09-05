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

from dataclasses import dataclass

from PIL import Image
from torchvision import transforms

# PIL size is (width, height). Matches the portrait capture roughly 1:2.
INPUT_WIDTH = 224
INPUT_HEIGHT = 448

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_PAD_RGB = tuple(int(round(c * 255)) for c in IMAGENET_MEAN)


@dataclass(frozen=True)
class LetterboxGeom:
    src_w: int
    src_h: int
    dst_w: int
    dst_h: int
    scale: float
    new_w: int
    new_h: int
    pad_x: int
    pad_y: int


def letterbox_geometry(
    src_w: int,
    src_h: int,
    dst_w: int = INPUT_WIDTH,
    dst_h: int = INPUT_HEIGHT,
) -> LetterboxGeom:
    scale = min(dst_w / src_w, dst_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    return LetterboxGeom(
        src_w=src_w,
        src_h=src_h,
        dst_w=dst_w,
        dst_h=dst_h,
        scale=scale,
        new_w=new_w,
        new_h=new_h,
        pad_x=(dst_w - new_w) // 2,
        pad_y=(dst_h - new_h) // 2,
    )


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
        geom = letterbox_geometry(src_w, src_h, self.width, self.height)
        resized = image.resize((geom.new_w, geom.new_h), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (self.width, self.height), self.fill)
        canvas.paste(resized, (geom.pad_x, geom.pad_y))
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
