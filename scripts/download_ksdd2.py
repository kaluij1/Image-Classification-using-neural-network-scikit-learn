"""Download the official KolektorSDD2 release from ViCoS.

Source: https://www.vicos.si/resources/kolektorsdd2/
Redirects to: https://data.vicos.si/datasets/KSDD/KolektorSDD2.zip
License: CC BY-NC-SA 4.0 (non-commercial; contact authors for commercial use).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

OFFICIAL_URL = "https://go.vicos.si/kolektorsdd2"
EXPECTED_SIZE_BYTES = 853_126_555


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == EXPECTED_SIZE_BYTES:
        print(f"Already present: {dest} ({dest.stat().st_size} bytes)")
        return

    print(f"Downloading {url} -> {dest}")
    with urllib.request.urlopen(url) as response, dest.open("wb") as out:
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if downloaded % (50 * 1024 * 1024) < 1024 * 1024:
                print(f"  {downloaded / 1_048_576:.0f} MiB")

    size = dest.stat().st_size
    print(f"Downloaded {size} bytes")
    if size != EXPECTED_SIZE_BYTES:
        print(
            f"Warning: size {size} != published Content-Length {EXPECTED_SIZE_BYTES}",
            file=sys.stderr,
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive} -> {dest}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)
    print("Extraction complete")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory for the zip and extracted files",
    )
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()

    archive = args.data_dir / "KolektorSDD2.zip"
    extract_dir = args.data_dir / "KolektorSDD2"
    download(OFFICIAL_URL, archive)
    print(f"SHA-256: {sha256_file(archive)}")
    if not args.skip_extract:
        extract(archive, extract_dir)


if __name__ == "__main__":
    main()
