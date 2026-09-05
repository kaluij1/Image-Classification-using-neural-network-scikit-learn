"""Grad-CAM for the Phase 2 MobileNetV3-Small classifier.

Backpropagates the defective logit through the last backbone feature
map. This is analysis of the existing checkpoint, not a new model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ksdd2_errors import (
    CAM_NEIGHBORHOOD_RADIUS_PX,
    dilate_binary,
    letterbox_mask,
    load_mask_binary,
    unletterbox_map,
)
from ksdd2_inventory import Sample
from ksdd2_transforms import (
    INPUT_HEIGHT,
    INPUT_WIDTH,
    eval_transforms,
    letterbox_geometry,
)

CAM_BIN_FRACTION = 0.5


class FeatureGradCAM:
    """Grad-CAM on ``model.features[-1]`` (last MobileNetV3-Small conv)."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.model.eval()
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        target = model.features[-1]
        self._fwd_handle = target.register_forward_hook(self._on_forward)
        self._bwd_handle = target.register_full_backward_hook(self._on_backward)

    def _on_forward(self, _module, _inputs, output: torch.Tensor) -> None:
        self._activations = output

    def _on_backward(
        self, _module, _grad_input, grad_output: tuple[torch.Tensor, ...]
    ) -> None:
        self._gradients = grad_output[0]

    def close(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __call__(self, image: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        if image.ndim != 4:
            image = image.unsqueeze(0)
        self.model.zero_grad(set_to_none=True)
        logit = self.model(image).squeeze()
        logit.backward()
        if self._activations is None or self._gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations")
        activations = self._activations.detach()
        gradients = self._gradients.detach()
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam_map = cam.squeeze(0).squeeze(0).cpu().numpy().astype(np.float64)
        peak = float(cam_map.max())
        if peak > 0:
            cam_map = cam_map / peak
        return cam_map, np.array([int(d) for d in cam.shape[-2:]], dtype=int)


def upsample_cam(cam: np.ndarray, height: int, width: int) -> np.ndarray:
    tensor = torch.from_numpy(cam.astype(np.float32))[None, None]
    resized = F.interpolate(
        tensor, size=(height, width), mode="bilinear", align_corners=False
    )
    return resized.squeeze().cpu().numpy().astype(np.float32)


def overlap_metrics(
    cam_original: np.ndarray,
    mask: np.ndarray,
    neighborhood_radius: int = CAM_NEIGHBORHOOD_RADIUS_PX,
) -> dict[str, Any]:
    mask = mask.astype(bool)
    cam = np.asarray(cam_original, dtype=np.float64)
    cam_sum = float(cam.sum())
    peak = float(cam.max())
    payload: dict[str, Any] = {
        "cam_max": peak,
        "cam_empty": peak <= 0,
        "mask_pixels": int(mask.sum()),
        "high_cam_pixels": 0,
        "intersection_pixels": 0,
        "mask_coverage": None,
        "high_cam_precision": None,
        "cam_mass_in_mask": None,
        "hit": False,
        "peak_in_mask": False,
        "peak_in_neighborhood": False,
        "neighborhood_radius_px": neighborhood_radius,
        "peak_y": None,
        "peak_x": None,
        "peak_y_frac": None,
        "peak_x_frac": None,
    }
    if peak <= 0:
        return payload
    high = cam >= (CAM_BIN_FRACTION * peak)
    payload["high_cam_pixels"] = int(high.sum())
    ys, xs = np.unravel_index(int(cam.argmax()), cam.shape)
    payload["peak_y"] = int(ys)
    payload["peak_x"] = int(xs)
    payload["peak_y_frac"] = float(ys / cam.shape[0])
    payload["peak_x_frac"] = float(xs / cam.shape[1])
    if mask.any():
        intersection = np.logical_and(high, mask)
        payload["intersection_pixels"] = int(intersection.sum())
        payload["mask_coverage"] = float(intersection.sum() / mask.sum())
        payload["high_cam_precision"] = (
            float(intersection.sum() / high.sum()) if high.any() else 0.0
        )
        payload["cam_mass_in_mask"] = (
            float((cam * mask).sum() / cam_sum) if cam_sum > 0 else 0.0
        )
        payload["hit"] = bool(intersection.any())
        payload["peak_in_mask"] = bool(mask[ys, xs])
        payload["peak_in_neighborhood"] = bool(
            dilate_binary(mask, neighborhood_radius)[ys, xs]
        )
    return payload


def explain_sample(
    camber: FeatureGradCAM,
    sample: Sample,
    device: torch.device,
    *,
    input_width: int = INPUT_WIDTH,
    input_height: int = INPUT_HEIGHT,
) -> dict[str, Any]:
    transform = eval_transforms(input_width, input_height)
    with Image.open(sample.image_path) as image:
        rgb = image.convert("RGB")
        tensor = transform(rgb).to(device)
    cam_small, cam_hw = camber(tensor)
    cam_letterbox = upsample_cam(cam_small, input_height, input_width)
    geom = letterbox_geometry(sample.width, sample.height, input_width, input_height)
    cam_original = unletterbox_map(cam_letterbox, geom)
    mask = load_mask_binary(sample.mask_path)
    mask_letterbox = letterbox_mask(mask, geom)
    metrics = overlap_metrics(cam_original, mask)
    metrics["cam_feature_hw"] = [int(cam_hw[0]), int(cam_hw[1])]
    metrics["letterbox_mask_pixels"] = int(mask_letterbox.sum())
    return {
        "sample_id": sample.sample_id,
        "cam_original": cam_original,
        "cam_letterbox": cam_letterbox,
        "mask": mask,
        "metrics": metrics,
    }
