from __future__ import annotations

import argparse
import shutil
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def collect_images(root: Path):
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]


def copy_images(images, out_dir: Path, prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(images):
        dst = out_dir / f"{prefix}_{i:06d}{src.suffix.lower()}"
        shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", required=True, help="Root folder containing original ExDark images")
    parser.add_argument("--enhanced_root", required=True, help="Root folder containing enhanced ExDark images")
    parser.add_argument("--out_root", required=True, help="Output binary dataset root")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    enhanced_root = Path(args.enhanced_root)
    out_root = Path(args.out_root)

    raw_images = collect_images(raw_root)
    enhanced_images = collect_images(enhanced_root)

    print(f"Found {len(raw_images)} raw images")
    print(f"Found {len(enhanced_images)} enhanced images")

    raw_out = out_root / "raw"
    enhanced_out = out_root / "enhanced"

    copy_images(raw_images, raw_out, "raw")
    copy_images(enhanced_images, enhanced_out, "enh")

    print(f"Saved binary dataset to: {out_root}")


if __name__ == "__main__":
    main()