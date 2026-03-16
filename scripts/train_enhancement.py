from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as skimage_ssim
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms
import torchvision.utils as vutils


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import config
except ImportError:
    config = None


# Utility / config helpers

def get_cfg(name: str, default):
    if config is None:
        return default
    return getattr(config, name, default)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_device(device_arg: Optional[str]) -> torch.device:
    if device_arg is not None:
        if device_arg.startswith("cuda") and not torch.cuda.is_available():
            print("Requested CUDA but CUDA is not available. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Dataset loading

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return sorted(files)


def find_case_insensitive_subdir(parent: Path, name: str) -> Optional[Path]:
    if not parent.exists():
        return None
    for child in parent.iterdir():
        if child.is_dir() and child.name.lower() == name.lower():
            return child
    return None


def match_image_pairs(low_dir: Path, high_dir: Path) -> List[Tuple[Path, Path]]:
    low_files = list_images(low_dir)
    high_files = list_images(high_dir)

    if not low_files or not high_files:
        return []

    high_by_stem = {p.stem: p for p in high_files}
    high_by_name = {p.name: p for p in high_files}

    pairs: List[Tuple[Path, Path]] = []
    for low_path in low_files:
        match = high_by_name.get(low_path.name)
        if match is None:
            match = high_by_stem.get(low_path.stem)
        if match is not None:
            pairs.append((low_path, match))
    return pairs


class PairedImageDataset(Dataset):
    def __init__(self, pairs: Sequence[Tuple[Path, Path]], image_size: int = 256, input_mode: str = "rgb"):
        self.pairs = list(pairs)
        self.image_size = image_size
        self.input_mode = input_mode

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        low_path, high_path = self.pairs[idx]

        low_img = Image.open(low_path).convert("RGB")
        high_img = Image.open(high_path).convert("RGB")

        low_tensor = self.transform(low_img)
        high_tensor = self.transform(high_img)

        if self.input_mode == "gray":
            low_tensor = transforms.functional.rgb_to_grayscale(low_tensor, num_output_channels=1)
            high_tensor = transforms.functional.rgb_to_grayscale(high_tensor, num_output_channels=1)

        return low_tensor, high_tensor, str(low_path), str(high_path)


def build_lol_dataset(root: Path, split: str, image_size: int, input_mode: str) -> Optional[Dataset]:
    """
    LOL expected:
    root/
      our485/low, our485/high
      eval15/low, eval15/high
    """
    if not root.exists():
        return None

    split_dir_name = "our485" if split == "train" else "eval15"
    split_dir = root / split_dir_name
    if not split_dir.exists():
        return None

    low_dir = split_dir / "low"
    high_dir = split_dir / "high"
    pairs = match_image_pairs(low_dir, high_dir)

    if not pairs:
        return None

    return PairedImageDataset(pairs, image_size=image_size, input_mode=input_mode)


def build_loli_dataset(root: Path, split: str, image_size: int, input_mode: str) -> Optional[Dataset]:
    """
    LoLI-Street expected:
    root/
      Train|train/low, high
      Val|val/low, high
      Test|test/low, high
    """
    if not root.exists():
        return None

    split_map = {
        "train": "train",
        "val": "val",
        "test": "test",
    }
    split_dir = find_case_insensitive_subdir(root, split_map[split])
    if split_dir is None:
        return None

    low_dir = find_case_insensitive_subdir(split_dir, "low")
    high_dir = find_case_insensitive_subdir(split_dir, "high")
    if low_dir is None or high_dir is None:
        return None

    pairs = match_image_pairs(low_dir, high_dir)
    if not pairs:
        return None

    return PairedImageDataset(pairs, image_size=image_size, input_mode=input_mode)


def build_combined_split(
    lol_root: Optional[Path],
    loli_root: Optional[Path],
    split: str,
    image_size: int,
    input_mode: str,
) -> Dataset:
    datasets: List[Dataset] = []

    if lol_root is not None:
        ds = build_lol_dataset(lol_root, split, image_size, input_mode)
        if ds is not None:
            datasets.append(ds)

    if loli_root is not None:
        ds = build_loli_dataset(loli_root, split, image_size, input_mode)
        if ds is not None:
            datasets.append(ds)

    if not datasets:
        raise ValueError(f"No paired datasets available for split '{split}'")

    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def count_dataset_items(ds: Dataset) -> int:
    return len(ds)


# Models

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        x = self.relu(x + residual)
        return x


class ShallowCNN(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_ch, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class DeepCNN(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_ch, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch: int, base: int = 32):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base * 4, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)

        self.out_conv = nn.Conv2d(base, in_ch, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return torch.sigmoid(self.out_conv(d1))


class ResUNet(nn.Module):
    def __init__(self, in_ch: int, base: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.enc1 = ResidualBlock(base)
        self.down1 = nn.Conv2d(base, base * 2, kernel_size=3, stride=2, padding=1)

        self.enc2 = ResidualBlock(base * 2)
        self.down2 = nn.Conv2d(base * 2, base * 4, kernel_size=3, stride=2, padding=1)

        self.enc3 = ResidualBlock(base * 4)
        self.down3 = nn.Conv2d(base * 4, base * 8, kernel_size=3, stride=2, padding=1)

        self.bottleneck = ResidualBlock(base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(base * 8)

        self.up2 = nn.ConvTranspose2d(base * 8, base * 2, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(base * 4)

        self.up1 = nn.ConvTranspose2d(base * 4, base, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(base * 2)

        self.out_conv = nn.Conv2d(base * 2, in_ch, kernel_size=1)

    def forward(self, x):
        x0 = self.stem(x)
        e1 = self.enc1(x0)

        x1 = self.down1(e1)
        e2 = self.enc2(x1)

        x2 = self.down2(e2)
        e3 = self.enc3(x2)

        x3 = self.down3(e3)
        b = self.bottleneck(x3)

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return torch.sigmoid(self.out_conv(d1))


def build_model(model_name: str, input_mode: str) -> nn.Module:
    in_ch = 3 if input_mode == "rgb" else 1
    if model_name == "shallow":
        return ShallowCNN(in_ch)
    if model_name == "deep":
        return DeepCNN(in_ch)
    if model_name == "unet":
        return UNet(in_ch)
    if model_name == "resunet":
        return ResUNet(in_ch)
    raise ValueError(f"Unsupported model: {model_name}")


# Metrics / losses

def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps ** 2))


def ssim_torch(x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu_x = F.avg_pool2d(x, window_size, stride=1, padding=window_size // 2)
    mu_y = F.avg_pool2d(y, window_size, stride=1, padding=window_size // 2)

    sigma_x = F.avg_pool2d(x * x, window_size, stride=1, padding=window_size // 2) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, window_size, stride=1, padding=window_size // 2) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=window_size // 2) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    )
    return ssim_map.mean()


def compute_total_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    l1_weight: float,
    charbonnier_weight: float,
    ssim_weight: float,
) -> torch.Tensor:
    total = 0.0
    if l1_weight > 0:
        total = total + l1_weight * F.l1_loss(pred, target)
    if charbonnier_weight > 0:
        total = total + charbonnier_weight * charbonnier_loss(pred, target)
    if ssim_weight > 0:
        total = total + ssim_weight * (1.0 - ssim_torch(pred, target))
    return total


def batch_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target, reduction="none")
    mse = mse.flatten(1).mean(dim=1)
    psnr = 10.0 * torch.log10(1.0 / torch.clamp(mse, min=1e-10))
    return float(psnr.mean().item())


def tensor_to_image_np(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().clamp(0, 1)
    if x.ndim == 3:
        x = x.permute(1, 2, 0).numpy()
    else:
        raise ValueError("Expected CHW tensor")
    return x


def batch_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.detach().cpu().clamp(0, 1)
    target = target.detach().cpu().clamp(0, 1)

    vals: List[float] = []
    for i in range(pred.size(0)):
        pred_np = tensor_to_image_np(pred[i])
        target_np = tensor_to_image_np(target[i])

        if pred_np.shape[2] == 1:
            pred_np = pred_np[..., 0]
            target_np = target_np[..., 0]
            score = skimage_ssim(target_np, pred_np, data_range=1.0)
        else:
            score = skimage_ssim(target_np, pred_np, data_range=1.0, channel_axis=-1)

        vals.append(float(score))
    return float(np.mean(vals))


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    all_psnr: List[float] = []
    all_ssim: List[float] = []

    for low_img, high_img, _, _ in loader:
        low_img = low_img.to(device, non_blocking=True)
        high_img = high_img.to(device, non_blocking=True)

        pred = model(low_img)
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)

        all_psnr.append(batch_psnr(pred, high_img))
        all_ssim.append(batch_ssim(pred, high_img))

    mean_psnr = float(np.mean(all_psnr)) if all_psnr else 0.0
    mean_ssim = float(np.mean(all_ssim)) if all_ssim else 0.0
    return mean_psnr, mean_ssim


# Saving slide materials

def save_sample_triplets(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_dir: Path,
    max_images: int = 8,
) -> None:
    ensure_dir(save_dir)
    model.eval()

    saved = 0
    with torch.no_grad():
        for low_img, high_img, _, _ in loader:
            low_img = low_img.to(device, non_blocking=True)
            high_img = high_img.to(device, non_blocking=True)
            pred = model(low_img)

            for i in range(low_img.size(0)):
                triplet = torch.stack([
                    low_img[i].cpu(),
                    pred[i].cpu(),
                    high_img[i].cpu(),
                ])
                vutils.save_image(
                    triplet,
                    save_dir / f"sample_{saved:03d}.png",
                    nrow=3,
                    normalize=True,
                )
                saved += 1
                if saved >= max_images:
                    return


def save_curves(
    save_dir: Path,
    train_losses: List[float],
    val_psnrs: List[float],
    val_ssims: List[float],
) -> None:
    plots_dir = save_dir / "plots"
    ensure_dir(plots_dir)

    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(train_losses) + 1), train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Training Loss")
    plt.tight_layout()
    plt.savefig(plots_dir / "train_loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(val_psnrs) + 1), val_psnrs)
    plt.xlabel("Epoch")
    plt.ylabel("Validation PSNR")
    plt.title("Validation PSNR")
    plt.tight_layout()
    plt.savefig(plots_dir / "val_psnr_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(val_ssims) + 1), val_ssims)
    plt.xlabel("Epoch")
    plt.ylabel("Validation SSIM")
    plt.title("Validation SSIM")
    plt.tight_layout()
    plt.savefig(plots_dir / "val_ssim_curve.png", dpi=200)
    plt.close()


def save_run_config(args: argparse.Namespace, save_dir: Path) -> None:
    with open(save_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)


# Training

def parse_args() -> argparse.Namespace:
    default_lol = get_cfg("LOL_PATH", None)
    default_loli = get_cfg("LOLI_PATH", None)
    default_image_size = int(get_cfg("IMAGE_SIZE", 256))
    default_batch_size = int(get_cfg("BATCH_SIZE", 8))
    default_epochs = int(get_cfg("EPOCHS", 20))
    default_lr = float(get_cfg("LEARNING_RATE", 1e-4))
    default_num_workers = int(get_cfg("NUM_WORKERS", 4))
    default_device = str(get_cfg("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))

    parser = argparse.ArgumentParser()
    parser.add_argument("--lol_root", type=str, default=default_lol)
    parser.add_argument("--loli_root", type=str, default=default_loli)
    parser.add_argument("--model", type=str, required=True, choices=["shallow", "deep", "unet", "resunet"])
    parser.add_argument("--input_mode", type=str, default="rgb", choices=["rgb", "gray"])
    parser.add_argument("--image_size", type=int, default=default_image_size)
    parser.add_argument("--batch_size", type=int, default=default_batch_size)
    parser.add_argument("--epochs", type=int, default=default_epochs)
    parser.add_argument("--lr", type=float, default=default_lr)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=default_num_workers)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--l1_weight", type=float, default=1.0)
    parser.add_argument("--charbonnier_weight", type=float, default=0.2)
    parser.add_argument("--ssim_weight", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    save_dir = Path(args.save_dir)
    ensure_dir(save_dir)
    ensure_dir(save_dir / "checkpoints")
    ensure_dir(save_dir / "sample_triplets")
    save_run_config(args, save_dir)

    device = resolve_device(args.device)
    amp_enabled = device.type == "cuda" and args.model != "resunet"

    print("=" * 72)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"AMP enabled: {amp_enabled}")
    print("=" * 72)

    lol_root = Path(args.lol_root) if args.lol_root else None
    loli_root = Path(args.loli_root) if args.loli_root else None

    train_ds = build_combined_split(
        lol_root=lol_root,
        loli_root=loli_root,
        split="train",
        image_size=args.image_size,
        input_mode=args.input_mode,
    )
    val_ds = build_combined_split(
        lol_root=lol_root,
        loli_root=loli_root,
        split="val",
        image_size=args.image_size,
        input_mode=args.input_mode,
    )

    print(f"Train samples: {count_dataset_items(train_ds)}")
    print(f"Val samples:   {count_dataset_items(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = build_model(args.model, args.input_mode).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=amp_enabled)

    metrics_csv = save_dir / "metrics.csv"
    best_psnr = -float("inf")

    train_losses: List[float] = []
    val_psnrs: List[float] = []
    val_ssims: List[float] = []

    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_psnr", "val_ssim"])

        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=True)
            for low_img, high_img, _, _ in pbar:
                low_img = low_img.to(device, non_blocking=True)
                high_img = high_img.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=amp_enabled):
                    pred = model(low_img)
                    loss = compute_total_loss(
                        pred,
                        high_img,
                        l1_weight=args.l1_weight,
                        charbonnier_weight=args.charbonnier_weight,
                        ssim_weight=args.ssim_weight,
                    )

                if not torch.isfinite(loss):
                    raise ValueError("Non-finite loss detected during training.")

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                running_loss += float(loss.item())
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            avg_train_loss = running_loss / max(len(train_loader), 1)
            val_psnr, val_ssim = evaluate_model(model, val_loader, device)

            train_losses.append(avg_train_loss)
            val_psnrs.append(val_psnr)
            val_ssims.append(val_ssim)

            writer.writerow([epoch + 1, avg_train_loss, val_psnr, val_ssim])
            f.flush()

            print(
                f"Epoch {epoch + 1:03d} | "
                f"train_loss={avg_train_loss:.4f} | "
                f"val_psnr={val_psnr:.3f} | "
                f"val_ssim={val_ssim:.4f}"
            )

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "val_psnr": val_psnr,
                "val_ssim": val_ssim,
                "args": vars(args),
            }

            torch.save(checkpoint, save_dir / "checkpoints" / f"epoch_{epoch + 1:03d}.pt")
            torch.save(checkpoint, save_dir / "checkpoints" / "last.pt")

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save(checkpoint, save_dir / "checkpoints" / "best.pt")

    save_curves(save_dir, train_losses, val_psnrs, val_ssims)
    save_sample_triplets(model, val_loader, device, save_dir / "sample_triplets", max_images=8)

    print("=" * 72)
    print(f"Finished training: {args.model} | {args.input_mode}")
    print(f"Best validation PSNR: {max(val_psnrs):.3f}")
    print(f"Best validation SSIM: {max(val_ssims):.4f}")
    print(f"Saved outputs to: {save_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()