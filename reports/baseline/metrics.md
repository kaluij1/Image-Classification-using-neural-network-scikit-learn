# Phase 2 — PyTorch baseline

Generated (UTC): 2026-09-05T19:39:43.006450+00:00

Image-level defective vs ok. Positive = defective (hold for review).
Official test was frozen. Validation was carved from official train only.
Numbers below were computed from this run. Accuracy is reported and treated as misleading.

## Setup

- Backbone: `mobilenet_v3_small` (ImageNet pretrained=True)
- Input (letterbox): 224×448 (W×H)
- Loss: BCEWithLogitsLoss, pos_weight=8.4619 (train fold only: n_ok/n_defective)
- Optimizer: AdamW, lr_head=0.0003, lr_backbone=0.0001
- Epochs: 10 (backbone frozen for first 3)
- Best checkpoint: epoch 7 by validation PR-AUC (0.9072)
- Device: cpu

## Splits

| Role | N | Defective | OK | Source |
|---|---:|---:|---:|---|
| train | 1864 | 197 | 1667 | official train, stratified remainder |
| val | 467 | 49 | 418 | official train, stratified holdout |
| test | 1004 | 110 | 894 | official test (frozen) |

No train/val/test ID overlap. Copy files `10301 (copy)` are outside the inventory.

## Operating point

- Rule: highest_threshold_on_val_with_defective_recall>=0.938776; tie_break_higher_precision
- Threshold chosen on **val only**: 0.173938
- Val at that threshold: precision=0.3566, recall=0.9388, F1=0.5169
- Test was scored once at this threshold after selection. Test was not used to pick it.

## Metrics at the val-chosen threshold

| Split | Accuracy* | Precision (def) | Recall (def) | F1 (def) | Recall (ok) | PR-AUC (def) | ROC-AUC | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 0.7988 | 0.3405 | 0.9645 | 0.5033 | 0.7792 | 0.9290 | 0.9731 | 190 | 368 | 7 | 1299 |
| val | 0.8158 | 0.3566 | 0.9388 | 0.5169 | 0.8014 | 0.9072 | 0.9623 | 46 | 83 | 3 | 335 |
| test | 0.7878 | 0.3333 | 0.9364 | 0.4916 | 0.7696 | 0.9042 | 0.9627 | 103 | 206 | 7 | 688 |

\*Accuracy is the wrong primary metric at ~10.7% prevalence.

## Metrics at threshold 0.5 (reference only)

| Split | Accuracy* | Precision (def) | Recall (def) | F1 (def) | PR-AUC (def) |
|---|---:|---:|---:|---:|---:|
| train | 0.9431 | 0.6679 | 0.9188 | 0.7735 | 0.9290 |
| val | 0.9465 | 0.6875 | 0.8980 | 0.7788 | 0.9072 |
| test | 0.9253 | 0.6087 | 0.8909 | 0.7232 | 0.9042 |

## Primary metric (as defined)

- Defective-class recall on official test at the val-chosen threshold: **0.9364** (103/110 defective images).
- Threshold-free test PR-AUC (defective): **0.9042**.

## Threshold tradeoff (same checkpoint, val-only selection)

False negatives cost more than false positives, so the locked threshold is the highest validation cutoff that still catches at least 46 of 49 val defectives. The older max-recall rule with precision at least 0.5, max-F1, and the 0.5 default are references only. Test was not used to pick the cutoff.

- Max-F1 reference (0.860): test **24 FN / 7 FP**, recall 0.7818, precision 0.9247
- Precision-floor reference (0.473, max val recall with precision>=0.5): test **12 FN / 74 FP**, recall 0.8909, precision 0.5698
- Chosen recall floor 46/49 (0.174): test **7 FN / 206 FP**, recall 0.9364, precision 0.3333
- Threshold 0.5 (untuned default): test **12 FN / 63 FP**, recall 0.8909, precision 0.6087

This is an operating-point choice, not a retrain.

## Leakage / protocol notes

- ImageNet mean/std only (not computed from KSDD2, and not from test).
- `pos_weight` from the training fold only.
- Augmentation on the training role only.
- Threshold selected on validation only; official test frozen.
- Training target is the image-level mask-derived label, not the mask pixels.

