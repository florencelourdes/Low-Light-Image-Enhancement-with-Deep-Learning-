from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SimpleBinaryCNN(nn.Module):
    def __init__(self, in_ch: int = 3, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        loss_sum += loss.item()

    acc = correct / total if total else 0.0
    avg_loss = loss_sum / max(len(loader), 1)
    return avg_loss, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=getattr(config, "BATCH_SIZE", 8))
    parser.add_argument("--image_size", type=int, default=getattr(config, "IMAGE_SIZE", 256))
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--num_workers", type=int, default=getattr(config, "NUM_WORKERS", 4))
    args = parser.parse_args()

    seed_everything(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"

    print("=" * 72)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"AMP enabled: {amp_enabled}")
    print("=" * 72)

    tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    full_ds = datasets.ImageFolder(args.data_root, transform=tf)
    print("Classes:", full_ds.classes)

    n_total = len(full_ds)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        full_ds,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = SimpleBinaryCNN(in_ch=3, num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler("cuda", enabled=amp_enabled)

    train_losses = []
    val_losses = []
    val_accs = []
    best_val_acc = -1.0

    with open(save_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_acc"])

        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
            for x, y in pbar:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with autocast("cuda", enabled=amp_enabled):
                    logits = model(x)
                    loss = criterion(logits, y)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            train_loss = running_loss / max(len(train_loader), 1)
            val_loss, val_acc = evaluate(model, val_loader, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            writer.writerow([epoch + 1, train_loss, val_loss, val_acc])
            f.flush()

            print(f"Epoch {epoch+1:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

            ckpt = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "classes": full_ds.classes,
                "args": vars(args),
            }
            torch.save(ckpt, save_dir / "last.pt")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(ckpt, save_dir / "best.pt")

    test_loss, test_acc = evaluate(model, test_loader, device)

    with open(save_dir / "test_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["test_loss", "test_acc"])
        writer.writerow([test_loss, test_acc])

    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="train_loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Binary Classifier Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(val_accs) + 1), val_accs)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Binary Classifier Accuracy")
    plt.tight_layout()
    plt.savefig(save_dir / "val_acc_curve.png", dpi=200)
    plt.close()

    print("=" * 72)
    print(f"Final test accuracy: {test_acc:.4f}")
    print(f"Saved outputs to: {save_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()