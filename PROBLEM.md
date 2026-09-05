# Problem definition — KolektorSDD2 surface inspection

This document defines the decision problem. It is not a results report and not a setup guide. Counts that refer to the local copy were computed from the official ViCoS zip (see `reports/exploration/audit_summary.md`).

How to run the repo: [README.md](README.md).

## Problem

Build an image-level classifier that, given a single inspection photo of a Kolektor production item, predicts whether the item shows a visible surface defect.

This is a **production-style** quality-inspection problem, not a claim of a plant-ready system. The industrial decision we are approximating is the first gate on a line: **hold the part for review** versus **let it continue**.

Dataset: [Kolektor Surface-Defect Dataset 2 (KSDD2)](https://www.vicos.si/resources/kolektorsdd2/), released by ViCoS Lab (University of Ljubljana) with imagery and annotations from Kolektor Group d.o.o. License: [CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/) (non-commercial; contact the authors for commercial use).

Citation:

> Božič, J., Tabernik, D., and Skočaj, D. “Mixed supervision for surface-defect detection: from weakly to fully supervised learning.” *Computers in Industry*, 2021.

## What constitutes a defect

A **defective** image is one whose official ground-truth mask contains at least one nonzero pixel. An **ok** image has an empty (all-zero) mask.

The dataset authors describe the marked regions as visible surface defects on a production item: scratches, minor spots, and larger surface imperfections. Defect *types are not labeled*. This project therefore does **not** claim to distinguish scratch vs spot vs other morphology.

We do not invent additional defect taxonomy, severity grades, or functional-failure labels. Pixel masks exist and are used in Phase 3 for size, location, and Grad-CAM overlap only. They are not a training target.

## Operational meaning of a positive prediction

**Positive = defective.**

A positive model output means: treat the item as suspect. In this project that is an **image-level flag** (hold / send to human review). The Phase 4 API exposes that same flag. It is not an automatic scrap command, not a bounding box, and not a process-control interlock.

A negative output means: the image is consistent with the ok class under the chosen threshold.

## False-positive cost

A false positive holds or re-inspects a good part.

Typical costs in this setting: extra labor, line delay, and (if the hold is treated as a reject) scrap of a conforming item. That is wasteful, but the part itself is not defective.

For this project, false positives are the **cheaper** error.

## False-negative cost

A false negative lets a defective part continue downstream or ship.

Typical costs: rework later, assembly failure, warranty, customer return, and — depending on the end use of the component — possible safety or reliability exposure. The official pages describe Kolektor production items; they do not name a specific SKU in the dataset card, so we do not invent one.

For this project, false negatives are the **more expensive** error. The operating point should prefer missing fewer defects over maximizing raw accuracy.

## Primary evaluation metric

**Recall of the defective class** on the official test set, at a threshold chosen on a validation split carved from official train only.

Rationale: recall is the quantity that moves with missed defects. Accuracy is the wrong primary metric at ~10.7% prevalence (a constant “ok” predictor is already ~89% accurate).

Do not invent a numeric recall target (for example “95%”) without a validation curve that justifies it.

Locked operating point: on validation, the highest threshold that still catches at least 46 of 49 val defectives (recall ≥ 46/49). That is a recall-floor choice after a computed tradeoff: false negatives are treated as the more expensive error, so more holds of good parts are accepted. The earlier max-recall-among-precision≥0.5 rule and max-F1 are references only. Test is not used to pick the cutoff.

## Secondary metrics

Report these only after they have been computed:

- Precision of the defective class at the same operating point
- F1 of the defective class at that point
- Confusion matrix
- Average precision (PR-AUC) for the defective class — threshold-free model comparison
- ROC-AUC as supporting context only (prevalence-insensitive; not the decision metric)
- Accuracy, reported and treated as misleading

## Important limitations

- **Binary only.** Types, severity, and functional impact are unlabeled.
- **Classification discards location.** The native annotation is a pixel mask. An image-level model can be right for the wrong pixels.
- **Small defects.** On this copy, defective masks range from 23 to 43,869 foreground pixels (area fraction 0.017%–30.2%, median 1.56%). Eight defects cover less than 0.1% of the image. Aggressive resizing can erase the signal.
- **Variable geometry.** All 3,335 images are RGB, but there are 601 distinct `(width, height)` pairs (width 184–241, height 597–665). A single input size is a design choice, not a property of the data.
- **Official zip messiness.** The release includes `train/10301 (copy).png` and `train/10301_GT (copy).png`, byte-identical to `10301`. They are excluded from the inventory. Canonical counts then match the published 3,335 / 246+2085 / 110+894 split exactly.
- **Near-duplicate screen is coarse.** SHA-256 found no duplicates among canonical IDs. An 8×8 average hash produced 23 collision groups (47 images): 9 groups span train/test (all ok/ok in those groups) and 1 group mixes labels. That is expected for similar parts under a 64-bit hash, not proof of leakage. A Hamming 1–4 expansion produced 17,992 pairs and is discarded as too coarse.
- **Controlled capture.** Lighting and framing are industrial but not a survey of plant variation (new SKUs, dirt, lighting drift).
- **License.** CC BY-NC-SA 4.0. Fine for a portfolio; not a commercial deployment dataset without permission.
- **Not production-ready.** A real line would also need calibration, drift monitoring, human-in-the-loop review design, traceability, and a documented cost ratio — none of which this dataset provides.

## Split policy

- Freeze the official **test** split (1,004 images).
- Carve **validation** only from official **train** (2,331 images), stratified by defective/ok.
- Do not use test images, test masks, or test-derived statistics for training or threshold selection.
- Ignore the two `*(copy)*` files.

The splitter that implements this policy is `src/ksdd2_splits.py` (seed 42, 20% val from official train).
