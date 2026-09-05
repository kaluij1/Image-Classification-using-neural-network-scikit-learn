"""Serve the locked Phase 2 inspector over HTTP, or score one image.

Does not train. Does not retune the threshold. Bind --host 0.0.0.0 in Docker.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ksdd2_serve import DEFAULT_CHECKPOINT, DEFAULT_METRICS, Inspector  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument(
        "--check",
        type=Path,
        default=None,
        help="Score one local image and exit (no server).",
    )
    args = parser.parse_args()

    inspector = Inspector(checkpoint=args.checkpoint, metrics_path=args.metrics)
    if args.check is not None:
        from PIL import Image

        with Image.open(args.check) as image:
            image.load()
            result = inspector.predict_image(image)
        print(json.dumps(result.as_dict(), indent=2))
        return

    import uvicorn
    from ksdd2_api import create_app

    app = create_app(inspector)
    print(
        f"Locked threshold {inspector.threshold:.6f}  "
        f"open http://{args.host}:{args.port}/"
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
