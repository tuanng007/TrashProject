"""Download and extract the TrashNet dataset into the project data folder."""

from __future__ import annotations

import argparse
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable
from urllib.request import urlopen

CHUNK_SIZE = 1 << 15
TRASHNET_URL = "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"
EXPECTED_HASH = None  # set to known md5 if you want to enforce integrity


def iter_chunks(stream) -> Iterable[bytes]:
    while True:
        chunk = stream.read(CHUNK_SIZE)
        if not chunk:
            break
        yield chunk


def download_zip(url: str, destination: Path) -> None:
    print(f"Downloading TrashNet from {url} ...")
    with urlopen(url) as response, destination.open("wb") as handle:
        total = int(response.headers.get("Content-Length", "0") or 0)
        downloaded = 0
        for chunk in iter_chunks(response):
            handle.write(chunk)
            downloaded += len(chunk)
            if total:
                percent = downloaded / total * 100
                sys.stdout.write(f"\r  -> {percent:5.1f}%")
                sys.stdout.flush()
        sys.stdout.write("\n")
    if EXPECTED_HASH:
        import hashlib

        checksum = hashlib.md5(destination.read_bytes()).hexdigest()
        if checksum != EXPECTED_HASH:
            raise RuntimeError(f"Checksum mismatch: expected {EXPECTED_HASH}, got {checksum}")


def extract_zip(zip_path: Path, output_dir: Path, *, overwrite: bool) -> None:
    print(f"Extracting to {output_dir} ...")
    with zipfile.ZipFile(zip_path) as archive:
        members = [name for name in archive.namelist() if name.startswith("dataset-resized/")]
        if not members:
            raise RuntimeError("Archive does not contain 'dataset-resized/' root.")
        for member in members:
            info = archive.getinfo(member)
            if info.is_dir():
                continue
            relative = Path(*Path(member).parts[1:])
            target = output_dir / relative
            if target.exists() and not overwrite:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as src, target.open("wb") as dst:
                dst.write(src.read())


def dataset_exists(output_dir: Path) -> bool:
    classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    return all((output_dir / cls).exists() for cls in classes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the TrashNet dataset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/trashnet"),
        help="Directory where the dataset will be extracted.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download the archive even if the dataset is already present.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files that already exist when extracting the archive.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if dataset_exists(args.output_dir) and not args.force:
        print(f"Dataset already present under {args.output_dir}. Use --force to re-download.")
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = Path(tmp_dir) / "trashnet.zip"
        download_zip(TRASHNET_URL, zip_path)
        extract_zip(zip_path, args.output_dir, overwrite=args.overwrite)

    print("TrashNet dataset is ready.")


if __name__ == "__main__":
    main()
