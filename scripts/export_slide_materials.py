from __future__ import annotations
from pathlib import Path
import csv
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]

RUNS_DIR = ROOT / "runs" / "full_project"
EXPORT_DIR = ROOT / "presentation_assets"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FOLDERS = [
    "enhance_shallow_rgb",
    "enhance_deep_rgb",
    "enhance_unet_rgb",
    "enhance_resunet_rgb",
    "enhance_unet_gray",
]

def read_last_metrics(csv_path: Path):
    if not csv_path.exists():
        return None
    with open(csv_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None

def export_summary_csv():
    out_csv = EXPORT_DIR / "enhancement_summary.csv"
    rows_out = []

    for folder in MODEL_FOLDERS:
        metrics_path = RUNS_DIR / folder / "metrics.csv"
        row = read_last_metrics(metrics_path)
        if row is not None:
            rows_out.append({
                "model_run": folder,
                "epoch": row["epoch"],
                "train_loss": row["train_loss"],
                "val_psnr": row["val_psnr"],
                "val_ssim": row["val_ssim"],
            })

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model_run", "epoch", "train_loss", "val_psnr", "val_ssim"]
        )
        writer.writeheader()
        writer.writerows(rows_out)

    return rows_out

def export_bar_charts(rows_out):
    names = [r["model_run"] for r in rows_out]
    psnr = [float(r["val_psnr"]) for r in rows_out]
    ssim = [float(r["val_ssim"]) for r in rows_out]

    plt.figure(figsize=(9, 4))
    plt.bar(names, psnr)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("PSNR")
    plt.title("Validation PSNR by Model")
    plt.tight_layout()
    plt.savefig(EXPORT_DIR / "psnr_bar.png", dpi=200)
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.bar(names, ssim)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("SSIM")
    plt.title("Validation SSIM by Model")
    plt.tight_layout()
    plt.savefig(EXPORT_DIR / "ssim_bar.png", dpi=200)
    plt.close()

def export_example_grid():
    images = []
    labels = []

    for folder in MODEL_FOLDERS:
        triplet_dir = RUNS_DIR / folder / "sample_triplets"
        if triplet_dir.exists():
            files = sorted(triplet_dir.glob("sample_*.png"))
            if files:
                images.append(Image.open(files[0]).convert("RGB"))
                labels.append(folder)

    if not images:
        return

    widths, heights = zip(*(im.size for im in images))
    max_w = max(widths)
    total_h = sum(h + 40 for h in heights)

    canvas = Image.new("RGB", (max_w, total_h), "white")
    draw = ImageDraw.Draw(canvas)

    y = 0
    for label, im in zip(labels, images):
        canvas.paste(im, (0, y))
        draw.text((10, y + im.size[1] + 5), label, fill="black")
        y += im.size[1] + 40

    canvas.save(EXPORT_DIR / "model_comparison_montage.png")

def main():
    rows_out = export_summary_csv()
    if rows_out:
        export_bar_charts(rows_out)
    export_example_grid()
    print(f"Saved presentation assets to: {EXPORT_DIR}")

if __name__ == "__main__":
    main()