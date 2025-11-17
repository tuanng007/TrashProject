"""Download the TACO dataset into the project data folder.

This script fetches the official TACO v0.1 release (images + annotations)
and extracts it under the given output directory.

Note: TACO là dataset detection/segmentation; để huấn luyện classifier
theo cấu trúc `train/<class_name>/*.jpg` bạn vẫn cần thêm bước xử lý
(crop theo bounding box, chuẩn hóa nhãn...). Script này chỉ lo phần
tải và giải nén dữ liệu gốc.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable
from urllib.request import urlopen

CHUNK_SIZE = 1 << 15
TACO_IMAGES_URL = "https://github.com/pedropro/TACO/releases/download/v0.1/data.zip"
TACO_ANN_URL = "https://github.com/pedropro/TACO/releases/download/v0.1/annotations.json"


def iter_chunks(stream) -> Iterable[bytes]:
    while True:
        chunk = stream.read(CHUNK_SIZE)
        if not chunk:
            break
        yield chunk


def download_file(url: str, destination: Path) -> None:
    print(f"Downloading from {url} ...")
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


def extract_zip(zip_path: Path, output_dir: Path, *, overwrite: bool) -> None:
    print(f"Extracting images to {output_dir} ...")
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            target = output_dir / member.filename
            if target.exists() and not overwrite:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, target.open("wb") as dst:
                dst.write(src.read())


def dataset_exists(output_dir: Path) -> bool:
    images_dir = output_dir / "data"
    ann_file = output_dir / "annotations.json"
    return images_dir.exists() and ann_file.exists()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the TACO dataset (v0.1).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/taco"),
        help="Directory where images and annotations will be stored.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the dataset structure already exists.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files when extracting images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if dataset_exists(args.output_dir) and not args.force:
        print(f"TACO dataset already present under {args.output_dir}. Use --force to re-download.")
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        images_zip = tmp_dir_path / "taco_data.zip"
        ann_path = args.output_dir / "annotations.json"

        # Download images archive
        download_file(TACO_IMAGES_URL, images_zip)
        extract_zip(images_zip, args.output_dir / "data", overwrite=args.overwrite)

        # Download annotations
        print(f"Downloading annotations to {ann_path} ...")
        download_file(TACO_ANN_URL, ann_path)

    print("TACO dataset is ready.")


if __name__ == "__main__":
    main()

