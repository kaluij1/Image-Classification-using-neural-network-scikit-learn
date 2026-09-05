"""Val-only threshold tradeoff on the frozen Phase 2 checkpoint.

Does not train and does not retune on test. The locked row is the
46/49 recall floor. Writes the comparison table; does not start an API.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ksdd2_metrics import (  # noqa: E402
    binary_metrics,
    select_threshold_max_f1,
    select_threshold_max_recall,
    select_threshold_recall_floor,
)
from ksdd2_report import write_json  # noqa: E402

# 90% of 49 val defectives is 44.1, so 45/49 is the first integer that meets it.
VAL_N_DEFECTIVE = 49
TEST_N_DEFECTIVE = 110


def _load_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray(payload["y_true"], dtype=int), np.asarray(
        payload["y_score"], dtype=np.float64
    )


def _row(
    name: str,
    selection: dict,
    val_xy: tuple[np.ndarray, np.ndarray],
    test_xy: tuple[np.ndarray, np.ndarray],
    *,
    locked: bool = False,
    notes: str = "",
) -> dict:
    threshold = float(selection["threshold"])
    val_m = binary_metrics(*val_xy, threshold)
    test_m = binary_metrics(*test_xy, threshold)
    return {
        "name": name,
        "rule": selection["rule"],
        "threshold": threshold,
        "locked": locked,
        "notes": notes,
        "val": val_m,
        "test": test_m,
    }


def _cm(metrics: dict) -> dict:
    return metrics["confusion_matrix"]


def _md_row(row: dict) -> str:
    val = row["val"]
    test = row["test"]
    vcm = _cm(val)
    tcm = _cm(test)
    flag = "yes" if row["locked"] else ""
    return (
        f"| {row['name']} | {row['threshold']:.4f} | {flag} | "
        f"{vcm['tp']}/{vcm['fp']}/{vcm['fn']}/{vcm['tn']} | "
        f"{val['recall_defective']:.4f} | {val['precision_defective']:.4f} | "
        f"{tcm['tp']}/{tcm['fp']}/{tcm['fn']}/{tcm['tn']} | "
        f"{test['recall_defective']:.4f} | {test['precision_defective']:.4f} |"
    )


def plot_tradeoff(rows: list[dict], dest: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    test_fp = [row["test"]["confusion_matrix"]["fp"] for row in rows]
    test_fn = [row["test"]["confusion_matrix"]["fn"] for row in rows]
    labels = [row["name"] for row in rows]
    ax.scatter(test_fp, test_fn, s=40)
    for x, y, label in zip(test_fp, test_fn, labels):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=7)
    ax.axhline(TEST_N_DEFECTIVE - np.ceil(0.90 * TEST_N_DEFECTIVE), linestyle="--", color="gray")
    ax.set_xlabel("Test FP")
    ax.set_ylabel("Test FN")
    ax.set_title("Official test tradeoff (thresholds chosen on val only)")
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=130)
    plt.close(fig)


def write_markdown(payload: dict, dest: Path) -> None:
    lines = [
        "# Threshold tradeoff — 46/49 locked",
        "",
        f"Generated (UTC): {payload['generated_at_utc']}",
        "",
        "Same Phase 2 checkpoint and saved scores. Every named cutoff below "
        "was chosen on **validation only**. Official test was scored after "
        "the fact. The locked row is the 46/49 recall floor. Nothing is "
        "wired into an API.",
        "",
        "Locked rule: highest val threshold that still catches 46 of 49 "
        "val defectives. False negatives are treated as more expensive than "
        "false positives, so more holds of good parts are accepted.",
        "",
        "## Table",
        "",
        "| Point | Threshold | Locked | Val TP/FP/FN/TN | Val rec | Val prec | Test TP/FP/FN/TN | Test rec | Test prec |",
        "|---|---:|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(_md_row(row))
    lines.extend(
        [
            "",
            "Locked = the operating point in `PROBLEM.md` and `reports/baseline/metrics.md`.",
            "",
            "## Notes",
            "",
        ]
    )
    for row in payload["rows"]:
        lines.append(f"- **{row['name']}**: {row['rule']}. {row['notes']}".rstrip())
    lines.extend(
        [
            "",
            "## Fixed lower probes (not selected on val)",
            "",
            "These cutoffs are just lower numbers than 0.4727. They are "
            "included so the 99/110 crossing is visible. They are **not** "
            "a validation rule and must not be chosen because a particular "
            "test FN sits next to them.",
            "",
            "| Point | Threshold | Locked | Val TP/FP/FN/TN | Val rec | Val prec | Test TP/FP/FN/TN | Test rec | Test prec |",
            "|---|---:|:---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["probe_rows"]:
        lines.append(_md_row(row))
    lines.extend(
        [
            "",
            "Test was not used to pick any val-selected cutoff. The 99/110 "
            "line on the scatter is a readout, not a selection rule.",
            "",
        ]
    )
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=ROOT / "reports" / "baseline",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports" / "error_analysis",
    )
    args = parser.parse_args()

    val_xy = _load_xy(args.baseline_dir / "predictions_val.json")
    test_xy = _load_xy(args.baseline_dir / "predictions_test.json")

    chosen = select_threshold_max_recall(*val_xy, min_precision=0.5)
    rows = [
        _row(
            "prec>=0.5 (reference)",
            chosen,
            val_xy,
            test_xy,
            notes="Earlier Phase 2 rule. Kept as a reference only.",
        ),
        _row(
            "default 0.5",
            {"rule": "fixed_threshold", "threshold": 0.5},
            val_xy,
            test_xy,
            notes="Untuned default. Reference only.",
        ),
        _row(
            "max-F1",
            select_threshold_max_f1(*val_xy),
            val_xy,
            test_xy,
            notes="High-precision reference. Not the decision rule.",
        ),
        _row(
            "val rec>=90% (min FP)",
            select_threshold_recall_floor(*val_xy, min_recall=0.90),
            val_xy,
            test_xy,
            notes="Highest val cutoff that still catches ≥45/49.",
        ),
        _row(
            "val rec>=46/49",
            select_threshold_recall_floor(*val_xy, min_recall=46 / VAL_N_DEFECTIVE),
            val_xy,
            test_xy,
            locked=True,
            notes="Locked operating point. Highest val cutoff with at least 46/49.",
        ),
        _row(
            "val rec>=47/49",
            select_threshold_recall_floor(*val_xy, min_recall=47 / VAL_N_DEFECTIVE),
            val_xy,
            test_xy,
            notes="Val recall floor 47/49.",
        ),
        _row(
            "val rec>=48/49",
            select_threshold_recall_floor(*val_xy, min_recall=48 / VAL_N_DEFECTIVE),
            val_xy,
            test_xy,
            notes="Val recall floor 48/49.",
        ),
        _row(
            "val rec>=49/49",
            select_threshold_recall_floor(*val_xy, min_recall=1.0),
            val_xy,
            test_xy,
            notes="Catch every val defective. Most FPs.",
        ),
        _row(
            "max rec, prec>=0.4",
            select_threshold_max_recall(*val_xy, min_precision=0.4),
            val_xy,
            test_xy,
            notes="Same Phase 2 rule with a lower precision floor.",
        ),
        _row(
            "max rec, prec>=0.3",
            select_threshold_max_recall(*val_xy, min_precision=0.3),
            val_xy,
            test_xy,
            notes="Same Phase 2 rule with precision floor 0.3.",
        ),
    ]
    probe_rows = [
        _row(
            f"probe {threshold:g}",
            {"rule": "fixed_threshold_probe_not_selected_on_val", "threshold": threshold},
            val_xy,
            test_xy,
            notes="Fixed probe. Not a val selection rule. Not locked.",
        )
        for threshold in (0.45, 0.40, 0.35, 0.30, 0.25, 0.20)
    ]

    # Drop exact-threshold duplicates after the first occurrence, keep names.
    seen: set[float] = set()
    unique_for_plot: list[dict] = []
    for row in rows:
        key = round(row["threshold"], 6)
        if key not in seen:
            seen.add(key)
            unique_for_plot.append(row)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_predictions": {
            "val": str(args.baseline_dir / "predictions_val.json"),
            "test": str(args.baseline_dir / "predictions_test.json"),
        },
        "locked": True,
        "locked_name": "val rec>=46/49",
        "api_updated": False,
        "target_note": (
            "90% of official-test defectives = 99/110. "
            "Val-selected thresholds were not chosen to hit that test count. "
            "Probe rows are fixed cutoffs, not a selection rule."
        ),
        "rows": rows,
        "probe_rows": probe_rows,
    }
    write_json(output_dir / "threshold_tradeoff.json", payload)
    write_markdown(payload, output_dir / "threshold_tradeoff.md")
    plot_tradeoff(unique_for_plot, output_dir / "figures" / "threshold_tradeoff.png")
    print(f"Wrote {output_dir / 'threshold_tradeoff.md'}")
    for row in rows + probe_rows:
        tcm = _cm(row["test"])
        print(
            f"{row['name']:24s} thr={row['threshold']:.4f}  "
            f"test {tcm['tp']}/{tcm['fp']}/{tcm['fn']}/{tcm['tn']}  "
            f"rec={row['test']['recall_defective']:.4f}"
        )


if __name__ == "__main__":
    main()
