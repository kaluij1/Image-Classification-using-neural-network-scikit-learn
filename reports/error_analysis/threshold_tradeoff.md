# Threshold tradeoff — 46/49 locked

Generated (UTC): 2026-09-05T19:43:13.768047+00:00

Same Phase 2 checkpoint and saved scores. Every named cutoff below was chosen on **validation only**. Official test was scored after the fact. The locked row is the 46/49 recall floor. Nothing is wired into an API.

Locked rule: highest val threshold that still catches 46 of 49 val defectives. False negatives are treated as more expensive than false positives, so more holds of good parts are accepted.

## Table

| Point | Threshold | Locked | Val TP/FP/FN/TN | Val rec | Val prec | Test TP/FP/FN/TN | Test rec | Test prec |
|---|---:|:---:|---:|---:|---:|---:|---:|---:|
| prec>=0.5 (reference) | 0.4727 |  | 45/21/4/397 | 0.9184 | 0.6818 | 98/74/12/820 | 0.8909 | 0.5698 |
| default 0.5 | 0.5000 |  | 44/20/5/398 | 0.8980 | 0.6875 | 98/63/12/831 | 0.8909 | 0.6087 |
| max-F1 | 0.8602 |  | 42/5/7/413 | 0.8571 | 0.8936 | 86/7/24/887 | 0.7818 | 0.9247 |
| val rec>=90% (min FP) | 0.4727 |  | 45/21/4/397 | 0.9184 | 0.6818 | 98/74/12/820 | 0.8909 | 0.5698 |
| val rec>=46/49 | 0.1739 | yes | 46/83/3/335 | 0.9388 | 0.3566 | 103/206/7/688 | 0.9364 | 0.3333 |
| val rec>=47/49 | 0.1157 |  | 47/119/2/299 | 0.9592 | 0.2831 | 106/298/4/596 | 0.9636 | 0.2624 |
| val rec>=48/49 | 0.0821 |  | 48/159/1/259 | 0.9796 | 0.2319 | 107/371/3/523 | 0.9727 | 0.2238 |
| val rec>=49/49 | 0.0176 |  | 49/327/0/91 | 1.0000 | 0.1303 | 110/699/0/195 | 1.0000 | 0.1360 |
| max rec, prec>=0.4 | 0.4727 |  | 45/21/4/397 | 0.9184 | 0.6818 | 98/74/12/820 | 0.8909 | 0.5698 |
| max rec, prec>=0.3 | 0.1739 |  | 46/83/3/335 | 0.9388 | 0.3566 | 103/206/7/688 | 0.9364 | 0.3333 |

Locked = the operating point in `PROBLEM.md` and `reports/baseline/metrics.md`.

## Notes

- **prec>=0.5 (reference)**: max_defective_recall_on_val_subject_to_precision>=0.5; tie_break_higher_precision. Earlier Phase 2 rule. Kept as a reference only.
- **default 0.5**: fixed_threshold. Untuned default. Reference only.
- **max-F1**: max_defective_f1_on_val; tie_break_higher_recall. High-precision reference. Not the decision rule.
- **val rec>=90% (min FP)**: highest_threshold_on_val_with_defective_recall>=0.9; tie_break_higher_precision. Highest val cutoff that still catches ≥45/49.
- **val rec>=46/49**: highest_threshold_on_val_with_defective_recall>=0.938776; tie_break_higher_precision. Locked operating point. Highest val cutoff with at least 46/49.
- **val rec>=47/49**: highest_threshold_on_val_with_defective_recall>=0.959184; tie_break_higher_precision. Val recall floor 47/49.
- **val rec>=48/49**: highest_threshold_on_val_with_defective_recall>=0.979592; tie_break_higher_precision. Val recall floor 48/49.
- **val rec>=49/49**: highest_threshold_on_val_with_defective_recall>=1; tie_break_higher_precision. Catch every val defective. Most FPs.
- **max rec, prec>=0.4**: max_defective_recall_on_val_subject_to_precision>=0.4; tie_break_higher_precision. Same Phase 2 rule with a lower precision floor.
- **max rec, prec>=0.3**: max_defective_recall_on_val_subject_to_precision>=0.3; tie_break_higher_precision. Same Phase 2 rule with precision floor 0.3.

## Fixed lower probes (not selected on val)

These cutoffs are just lower numbers than 0.4727. They are included so the 99/110 crossing is visible. They are **not** a validation rule and must not be chosen because a particular test FN sits next to them.

| Point | Threshold | Locked | Val TP/FP/FN/TN | Val rec | Val prec | Test TP/FP/FN/TN | Test rec | Test prec |
|---|---:|:---:|---:|---:|---:|---:|---:|---:|
| probe 0.45 | 0.4500 |  | 45/24/4/394 | 0.9184 | 0.6522 | 98/77/12/817 | 0.8909 | 0.5600 |
| probe 0.4 | 0.4000 |  | 45/32/4/386 | 0.9184 | 0.5844 | 100/91/10/803 | 0.9091 | 0.5236 |
| probe 0.35 | 0.3500 |  | 45/40/4/378 | 0.9184 | 0.5294 | 101/106/9/788 | 0.9182 | 0.4879 |
| probe 0.3 | 0.3000 |  | 45/51/4/367 | 0.9184 | 0.4688 | 102/124/8/770 | 0.9273 | 0.4513 |
| probe 0.25 | 0.2500 |  | 45/58/4/360 | 0.9184 | 0.4369 | 102/156/8/738 | 0.9273 | 0.3953 |
| probe 0.2 | 0.2000 |  | 45/73/4/345 | 0.9184 | 0.3814 | 103/189/7/705 | 0.9364 | 0.3527 |

Test was not used to pick any val-selected cutoff. The 99/110 line on the scatter is a readout, not a selection rule.

