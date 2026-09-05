# Manufacturing surface inspection (KolektorSDD2)

Production-oriented image classification for industrial surface defects: **defective** vs **ok**.

This repo is a portfolio project (ML + mechanical engineering). It is **not** a plant-ready inspection system. A real line would still need calibration, drift monitoring, a review workflow, and a documented cost ratio.

Decision problem, costs, and metrics: **[PROBLEM.md](PROBLEM.md)**.

## What is implemented vs planned

| Status | Scope |
|---|---|
| Done | Dataset choice, problem definition, local data audit |
| Done | Phase 2 PyTorch baseline (MobileNetV3-Small, transfer learning) |
| Not started | Error analysis / Grad-CAM, FastAPI, Docker, CI |
| Not this project | Cloud deploy, multi-class defect typing, pixel-level training |

Computed baseline numbers: `reports/baseline/metrics.md`. Do not quote test metrics from anywhere else.

## Dataset

[KolektorSDD2](https://www.vicos.si/resources/kolektorsdd2/) — ViCoS Lab / Kolektor Group. **CC BY-NC-SA 4.0** (non-commercial).

Local audit of the official zip (canonical IDs only):

| Split | N | Defective | OK |
|---|---:|---:|---:|
| Official train | 2,331 | 246 | 2,085 |
| Official test | 1,004 | 110 | 894 |
| All | 3,335 | 356 | 2,979 |

Images are RGB, roughly 229×637, variable size. A part is **defective** if its `_GT.png` mask has any nonzero pixel. Two official `10301 (copy)` files are ignored (byte-identical to `10301`).

Images are **not** in git. Download them locally.

## Evaluation protocol

- **Positive** = defective = hold for review. False negatives cost more than false positives.
- **Primary metric:** defective-class recall on official test, at a threshold chosen on validation only.
- **Secondary:** precision, F1, confusion matrix, PR-AUC. Accuracy is reported and treated as misleading (~10.7% defective).
- Official **test is frozen**. Validation is a stratified 20% of official train (seed 42).

Implemented split (from `reports/baseline/split_manifest.json`): train 1,864 (197/1,667), val 467 (49/418), test 1,004 (110/894).

## Repository layout

```
PROBLEM.md                 Decision problem (not results)
src/ksdd2_inventory.py     Discover images/masks, derive labels
src/ksdd2_audit.py         Phase 1 audit
src/ksdd2_splits.py        Train/val/test roles, leakage checks
src/ksdd2_dataset.py       Image-level PyTorch dataset
src/ksdd2_transforms.py    Letterbox 224×448, light train aug
src/ksdd2_model.py         MobileNetV3-Small, one logit
src/ksdd2_train.py         Train/eval loops
src/ksdd2_metrics.py       Threshold selection + metrics
src/ksdd2_report.py        Baseline figures and markdown
scripts/download_ksdd2.py
scripts/run_data_audit.py
scripts/train_baseline.py
notebooks/01_data_exploration.ipynb
notebooks/02_baseline_review.ipynb
reports/exploration/       Computed audit
reports/baseline/          Split + training outputs (when the run finishes)
data/raw/                  Local dataset (gitignored)
```

## Setup

Python 3.13, from the repo root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

CPU-only PyTorch (if the default wheel wants CUDA):

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Data and audit

```powershell
python scripts/download_ksdd2.py
python scripts/run_data_audit.py
```

Then open `notebooks/01_data_exploration.ipynb`. Audit numbers: `reports/exploration/audit_summary.md`.

## Baseline training

```powershell
python scripts/train_baseline.py
```

Writes `reports/baseline/` (split manifest, history, metrics, plots, `checkpoints/best.pt`). Review with `notebooks/02_baseline_review.ipynb` only after `metrics.json` is present.

Design choices for this baseline:

- Transfer learning, not training from scratch
- Portrait letterbox (224×448) instead of a square resize that would crush height
- No random crop (can drop a 23-pixel defect)
- Class imbalance handled with `BCEWithLogitsLoss` `pos_weight` from the **training fold only**
- Checkpoint on validation PR-AUC
- Operating threshold from validation: max defective recall among cutoffs with precision ≥ 0.5 (FN cost more than FP). Max-F1 is kept as a reference only.

## Limitations

See [PROBLEM.md](PROBLEM.md) for the full list. Short version: binary labels only, small defects, controlled lighting, non-commercial license, image-level scores can be right for the wrong pixels.

## Citation

If you use KSDD2, cite Božič, Tabernik, and Skočaj, *Computers in Industry*, 2021.
