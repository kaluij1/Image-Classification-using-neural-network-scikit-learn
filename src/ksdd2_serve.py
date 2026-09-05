"""Load the frozen Phase 2 checkpoint and apply the locked threshold.

This is inference only. It does not train, retune, or read the official
test split. Positive = hold for review (PROBLEM.md).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from ksdd2_model import BACKBONE_NAME, build_baseline
from ksdd2_transforms import INPUT_HEIGHT, INPUT_WIDTH, eval_transforms
from ksdd2_train import resolve_device

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "reports" / "baseline" / "checkpoints" / "best.pt"
DEFAULT_METRICS = ROOT / "reports" / "baseline" / "metrics.json"

DECISION_HOLD = "hold_for_review"
DECISION_CONTINUE = "continue"
LABEL_DEFECTIVE = "defective"
LABEL_OK = "ok"


@dataclass(frozen=True)
class InspectionResult:
    score: float
    threshold: float
    label: str
    decision: str
    positive: bool
    width: int
    height: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Inspector:
    def __init__(
        self,
        checkpoint: Path = DEFAULT_CHECKPOINT,
        metrics_path: Path = DEFAULT_METRICS,
        device: torch.device | None = None,
    ) -> None:
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Missing checkpoint: {checkpoint}. "
                "Train or copy reports/baseline/checkpoints/best.pt."
            )
        if not metrics_path.is_file():
            raise FileNotFoundError(
                f"Missing metrics: {metrics_path}. "
                "Need the locked threshold from reports/baseline/metrics.json."
            )
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
        selection = report["threshold_selection"]
        self.threshold = float(selection["threshold"])
        self.rule = str(selection["rule"])
        self.best_epoch = int(report["best_epoch"])
        self.best_val_pr_auc = float(report["best_val_pr_auc"])
        self.checkpoint_path = checkpoint
        self.metrics_path = metrics_path
        self.device = device or resolve_device()
        self.model = build_baseline(pretrained=False).to(self.device)
        payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()
        self.transform = eval_transforms(INPUT_WIDTH, INPUT_HEIGHT)

    def meta(self) -> dict[str, Any]:
        return {
            "backbone": BACKBONE_NAME,
            "input_width": INPUT_WIDTH,
            "input_height": INPUT_HEIGHT,
            "threshold": self.threshold,
            "rule": self.rule,
            "best_epoch": self.best_epoch,
            "best_val_pr_auc": self.best_val_pr_auc,
            "checkpoint": str(self.checkpoint_path),
            "positive_means": DECISION_HOLD,
            "not_a_scrap_command": True,
        }

    @torch.no_grad()
    def predict_image(self, image: Image.Image) -> InspectionResult:
        rgb = image.convert("RGB")
        width, height = rgb.size
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        logit = float(self.model(tensor).squeeze().cpu())
        score = 1.0 / (1.0 + math.exp(-logit))
        positive = score >= self.threshold
        return InspectionResult(
            score=score,
            threshold=self.threshold,
            label=LABEL_DEFECTIVE if positive else LABEL_OK,
            decision=DECISION_HOLD if positive else DECISION_CONTINUE,
            positive=positive,
            width=width,
            height=height,
        )
