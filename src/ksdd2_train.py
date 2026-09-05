"""Training and evaluation loops for the Phase 2 baseline."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ksdd2_dataset import KSDD2ClassificationDataset
from ksdd2_inventory import Sample
from ksdd2_metrics import binary_metrics
from ksdd2_model import build_baseline, parameter_groups, set_backbone_trainable
from ksdd2_transforms import INPUT_HEIGHT, INPUT_WIDTH, eval_transforms, train_transforms


@dataclass
class TrainConfig:
    seed: int = 42
    val_fraction: float = 0.2
    input_width: int = INPUT_WIDTH
    input_height: int = INPUT_HEIGHT
    batch_size: int = 24
    epochs: int = 10
    freeze_epochs: int = 3
    lr_head: float = 3e-4
    lr_backbone: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 0
    backbone: str = "mobilenet_v3_small"
    pretrained: bool = True
    default_threshold: float = 0.5


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_loader(
    samples: list[Sample],
    *,
    train: bool,
    config: TrainConfig,
    generator: torch.Generator | None = None,
) -> DataLoader:
    transform = (
        train_transforms(config.input_width, config.input_height)
        if train
        else eval_transforms(config.input_width, config.input_height)
    )
    dataset = KSDD2ClassificationDataset(samples, transform=transform)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=train,
        num_workers=config.num_workers,
        generator=generator if train else None,
        drop_last=False,
    )


def pos_weight_from_train(samples: list[Sample], device: torch.device) -> torch.Tensor:
    n_def = sum(1 for s in samples if s.label == "defective")
    n_ok = sum(1 for s in samples if s.label == "ok")
    if n_def == 0:
        raise ValueError("Training fold has no defective samples")
    return torch.tensor([n_ok / n_def], dtype=torch.float32, device=device)


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    logits_out: list[np.ndarray] = []
    labels_out: list[np.ndarray] = []
    sample_ids: list[str] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=False)
        logits = model(images).squeeze(1)
        logits_out.append(logits.detach().cpu().numpy())
        labels_out.append(batch["label"].numpy())
        sample_ids.extend(batch["sample_id"])
    logits_np = np.concatenate(logits_out)
    scores = 1.0 / (1.0 + np.exp(-logits_np))
    labels = np.concatenate(labels_out).astype(int)
    return {
        "sample_ids": sample_ids,
        "y_true": labels,
        "y_score": scores.astype(np.float64),
        "logits": logits_np.astype(np.float64),
    }


def epoch_pass(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    n = 0
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)
        if train_mode:
            loss.backward()
            optimizer.step()
        batch_n = int(labels.shape[0])
        total_loss += float(loss.item()) * batch_n
        n += batch_n
    return total_loss / max(n, 1)


def train_baseline(
    role_samples: dict[str, list[Sample]],
    config: TrainConfig,
    output_dir: Path,
    device: torch.device | None = None,
) -> dict[str, Any]:
    device = device or resolve_device()
    set_seed(config.seed)
    generator = torch.Generator()
    generator.manual_seed(config.seed)

    loaders = {
        "train": make_loader(
            role_samples["train"], train=True, config=config, generator=generator
        ),
        "val": make_loader(role_samples["val"], train=False, config=config),
        "test": make_loader(role_samples["test"], train=False, config=config),
    }
    # Monitoring loaders use eval transforms so train metrics are not
    # inflated/deflated by augmentation.
    eval_loaders = {
        "train": make_loader(role_samples["train"], train=False, config=config),
        "val": loaders["val"],
        "test": loaders["test"],
    }

    model = build_baseline(pretrained=config.pretrained).to(device)
    pos_weight = pos_weight_from_train(role_samples["train"], device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        parameter_groups(model, config.lr_backbone, config.lr_head),
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pt"

    history: list[dict[str, Any]] = []
    best_val_pr_auc = -1.0
    best_epoch = 0

    for epoch in range(1, config.epochs + 1):
        set_backbone_trainable(model, trainable=epoch > config.freeze_epochs)
        train_loss = epoch_pass(model, loaders["train"], device, criterion, optimizer)
        val_loss = epoch_pass(model, loaders["val"], device, criterion, None)
        scheduler.step()

        val_pred = collect_predictions(model, eval_loaders["val"], device)
        val_at_default = binary_metrics(
            val_pred["y_true"], val_pred["y_score"], config.default_threshold
        )
        row = {
            "epoch": epoch,
            "backbone_frozen": epoch <= config.freeze_epochs,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_recall_defective@0.5": val_at_default["recall_defective"],
            "val_precision_defective@0.5": val_at_default["precision_defective"],
            "val_f1_defective@0.5": val_at_default["f1_defective"],
            "val_pr_auc": val_at_default["pr_auc_defective"],
            "val_accuracy@0.5": val_at_default["accuracy"],
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}/{config.epochs} "
            f"frozen={row['backbone_frozen']} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_pr_auc={row['val_pr_auc']:.4f} "
            f"val_rec@0.5={row['val_recall_defective@0.5']:.4f}",
            flush=True,
        )

        if val_at_default["pr_auc_defective"] is not None and (
            val_at_default["pr_auc_defective"] > best_val_pr_auc
        ):
            best_val_pr_auc = val_at_default["pr_auc_defective"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "config": asdict(config),
                    "val_pr_auc": best_val_pr_auc,
                },
                best_path,
            )

    payload = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    return {
        "model": model,
        "device": str(device),
        "pos_weight": float(pos_weight.item()),
        "history": history,
        "best_epoch": best_epoch,
        "best_val_pr_auc": best_val_pr_auc,
        "best_checkpoint": str(best_path),
        "eval_loaders": eval_loaders,
    }
