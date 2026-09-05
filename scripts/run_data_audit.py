"""Run the Phase 1 KolektorSDD2 audit and write reports/exploration/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ksdd2_audit import run_audit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports" / "exploration",
    )
    args = parser.parse_args()
    report = run_audit(args.data_root, args.output_dir)
    counts = report["counts"]["all"]
    print(
        f"Audited {counts['n']} images "
        f"({counts['defective']} defective, {counts['ok']} ok)."
    )
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
