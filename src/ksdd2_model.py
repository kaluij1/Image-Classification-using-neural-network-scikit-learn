"""Small pretrained baseline: MobileNetV3-Small, one logit.

ImageNet weights initialize the backbone. The final linear layer is
replaced for binary defective-vs-ok. This is transfer learning, not a
from-scratch net.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

BACKBONE_NAME = "mobilenet_v3_small"


def build_baseline(pretrained: bool = True) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 1)
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = trainable


def parameter_groups(
    model: nn.Module,
    lr_backbone: float,
    lr_head: float,
) -> list[dict]:
    return [
        {"params": model.features.parameters(), "lr": lr_backbone},
        {"params": model.classifier.parameters(), "lr": lr_head},
    ]
