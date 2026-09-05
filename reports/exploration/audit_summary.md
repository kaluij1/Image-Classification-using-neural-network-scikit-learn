# KolektorSDD2 data audit

Generated (UTC): 2026-09-05T05:18:43.754422+00:00

All numbers below were computed from the official extracted zip.
No model was trained.

## Sample counts

| Split | N | Defective | OK | Defective % |
|---|---:|---:|---:|---:|
| train | 2331 | 246 | 2085 | 10.55 |
| test | 1004 | 110 | 894 | 10.96 |
| all | 3335 | 356 | 2979 | 10.67 |

## Image geometry

- Width range: 184–241 px (median 229)
- Height range: 597–665 px (median 637)
- Modes: {'RGB': 3335}
- Channel counts: {'3': 3335}
- Distinct (width, height) pairs: 601

## Mask / defect size (defective images only)

- Foreground pixels: min 23, median 2273, max 43869
- Area fraction: min 0.000171, median 0.0156, mean 0.0281, max 0.3016
- Defects covering < 0.5% of the image: 62
- Defects covering < 0.1% of the image: 8

## Integrity

- Missing masks: 0
- Unreadable images or masks: 0
- Unexpected extra files (not {id}.png / {id}_GT.png): 2
  - `train\10301 (copy).png`
  - `train\10301_GT (copy).png`


## Duplicates

- Exact SHA-256 duplicate groups (canonical IDs): 0 (0 images); cross-split groups: 0
- Identical 8×8 average-hash groups: 23 (47 images); 9 groups span train/test (ok/ok); 1 group mixes labels.
- A Hamming 1–4 expansion of that 8×8 hash produced 17,992 pairs (7,581 cross-split). That volume means the expansion is too coarse for this visually similar industrial set and is **not** treated as leakage.

Average-hash collisions are a cheap screen, not proof of identity. Do not treat them as train/test leakage without visual review. The official `10301 (copy)` files are byte-identical to `10301` and are excluded from the canonical inventory, after which counts match the published split exactly.

