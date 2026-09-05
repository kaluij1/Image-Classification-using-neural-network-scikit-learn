# Phase 3 — Error analysis

Generated (UTC): 2026-09-05T20:13:08.943085+00:00

Image-level defective vs ok at the locked val-only threshold (highest cutoff with at least 46/49 val defectives). Official test was not used to pick the threshold. Masks are used here for size, location, and Grad-CAM overlap only.

## Setup

- Checkpoint: Phase 2 `reports/baseline/checkpoints/best.pt` (epoch 7, val PR-AUC 0.9072)
- Threshold (val only, not retuned): 0.173938
- Device: cpu
- Grad-CAM: last `features` map of MobileNetV3-Small; backward on the defective logit
- High-CAM region: pixels ≥ 0.5 × max(CAM)
- Peak neighborhood: 16 px dilation of the official mask

## Confusion at the frozen operating point

Recomputed from saved Phase 2 scores (`predictions_*.json`), not from a retrain.

- Test: TP 103, FP 206, FN 7, TN 688
- Val (context): TP 46, FP 83, FN 3, TN 335

These match `reports/baseline/metrics.md`.

## False negatives on official test (expensive)

n = 7. Scores: min=0.0337, median=0.1009, max=0.1408.
Near-threshold FNs (score ≥ threshold − 0.05): 1.

| ID | Score | Mask px | Area | Area bin | Letterbox mask px | Location |
|---|---:|---:|---:|---|---:|---|
| 20816 | 0.0533 | 181 | 0.123% | 0.1–0.5% | 94 | bottom |
| 20544 | 0.1210 | 199 | 0.134% | 0.1–0.5% | 100 | top |
| 20236 | 0.0358 | 367 | 0.252% | 0.1–0.5% | 180 | top |
| 20801 | 0.1169 | 429 | 0.302% | 0.1–0.5% | 207 | top |
| 20523 | 0.0337 | 951 | 0.644% | 0.5–2% | 461 | top |
| 20640 | 0.1009 | 1200 | 0.834% | 0.5–2% | 609 | top |
| 20281 | 0.1408 | 2181 | 1.463% | 0.5–2% | 1050 | top |

Location is the mask centroid along image height (top / middle / bottom thirds). Letterbox mask px is the official mask after the same 224×448 letterbox the model sees.

## Are the misses the tiny defects?

Phase 1 audit: 8 defective masks in the whole zip cover less than 0.1% of the image.
Of those, 1 official test, 0 val, 7 train-role.
Val/test scores are from the saved Phase 2 prediction files. Train-role tiny-defect scores were computed from the same frozen checkpoint (there is no `predictions_train.json`).

| ID | Split | Role | Mask px | Area | Letterbox px | Score | Error |
|---|---|---|---:|---:|---:|---:|---|
| 20228 | test | test | 87 | 0.059% | 41 | 0.2184 | TP |
| 10045 | train | train | 23 | 0.017% | 12 | 0.5519 | TP |
| 10677 | train | train | 64 | 0.043% | 30 | 0.9455 | TP |
| 10844 | train | train | 78 | 0.054% | 41 | 0.8901 | TP |
| 10874 | train | train | 92 | 0.063% | 43 | 0.9219 | TP |
| 11230 | train | train | 62 | 0.042% | 28 | 0.9936 | TP |
| 11643 | train | train | 137 | 0.093% | 66 | 0.9429 | TP |
| 11912 | train | train | 96 | 0.066% | 53 | 0.5818 | TP |

Test defectives with area < 0.1%: 1. Among them, FN = 0, TP = 1.
Of the 7 test FNs, 0 have area < 0.1%.

### Test defectives by mask area

| Area bin | n | TP | FN | FN rate |
|---|---:|---:|---:|---:|
| <0.1% | 1 | 1 | 0 | 0.0000 |
| 0.1–0.5% | 16 | 12 | 4 | 0.2500 |
| 0.5–2% | 42 | 39 | 3 | 0.0714 |
| 2–10% | 45 | 45 | 0 | 0.0000 |
| ≥10% | 6 | 6 | 0 | 0.0000 |

### Test defectives by mask centroid (height thirds)

| Location | n | TP | FN | FN rate |
|---|---:|---:|---:|---:|
| top | 38 | 32 | 6 | 0.1579 |
| middle | 43 | 43 | 0 | 0.0000 |
| bottom | 29 | 28 | 1 | 0.0345 |

## False positives on official test (cheaper, numerous)

n = 206. Official GT is empty. Scores: min=0.1814, median=0.3586, max=0.9713.
FPs with score in [0.174, 0.5): 143.
FPs with score ≥ 0.8: 11.

Highest-score FPs (first 12): 20649 (0.971), 20889 (0.923), 20719 (0.912), 20162 (0.911), 20229 (0.907), 20511 (0.895), 20328 (0.891), 20113 (0.849), 20066 (0.835), 20262 (0.829), 20149 (0.810), 20348 (0.791).

## Grad-CAM (same checkpoint)

CAM is produced at the last backbone feature map (14×7 for a 224×448 input) and upsampled. Overlap is measured on the original image after un-letterboxing. A coarse map makes pixel IoU pessimistic for tiny marks; hit / peak-in-neighborhood are the counts to read first.

- Test FN (all 7): high-CAM overlaps mask in 5/7; peak in mask 2/7; peak in 16px neighborhood 5/7; empty CAM 0/7; median mask coverage 0.4511; median CAM mass in mask 0.0436.
- Test TP sample (n=9, up to 2 per area bin): overlap 8/9; peak in mask 6/9; peak in neighborhood 8/9; median mask coverage 0.7324; median CAM mass in mask 0.1208.
- Test FP sample (n=16; 12 highest-score + 4 nearest threshold): empty CAM 0/16. No GT region exists; peak height-third counts: top=3, middle=8, bottom=5.

Per-example Grad-CAM rows (FN, then sampled TP, then sampled FP) are in `error_report.json` under `gradcam.examples`.

## Files

- `error_report.json` — computed tables
- `figures/gallery_fn_test.png` — all test FNs with GT overlay (yellow box is padded 8 px for visibility)
- `figures/gallery_fp_test.png` — all test FPs (empty GT)
- `figures/gallery_gradcam_{fn,tp,fp}.png`
- `figures/area_stratification.png`, `figures/score_strip.png`

