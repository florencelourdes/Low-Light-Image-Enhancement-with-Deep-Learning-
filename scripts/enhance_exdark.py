from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config

from lowlight_project.models import build_enhancer
from lowlight_project.utils import ensure_dir, get_device, print_runtime_banner

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Run a trained enhancement model over ExDark and save enhanced images')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--model', required=True, choices=['shallow', 'deep', 'unet', 'resunet'])
    p.add_argument('--input_root', default=config.EXDARK_PATH, help='ExDark root with train/val/test or class folders')
    p.add_argument('--output_root', required=True)
    p.add_argument('--input_mode', default='rgb', choices=['rgb', 'gray'])
    p.add_argument('--image_size', type=int, default=config.CLASSIFIER_IMAGE_SIZE)
    p.add_argument('--device', default='')
    return p.parse_args()


def build_transform(image_size: int, input_mode: str):
    if input_mode == 'gray':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


def save_tensor_img(tensor: torch.Tensor, path: Path):
    arr = tensor.detach().cpu().clamp(0, 1)
    img = transforms.ToPILImage()(arr)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def main() -> None:
    args = parse_args()
    device = get_device(args.device or None)
    print_runtime_banner(device)
    channels = 1 if args.input_mode == 'gray' else 3
    model = build_enhancer(args.model, channels, channels).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    tf = build_transform(args.image_size, args.input_mode)

    input_root = Path(args.input_root)
    output_root = ensure_dir(args.output_root)

    for img_path in input_root.rglob('*'):
        if not img_path.is_file() or img_path.suffix.lower() not in IMG_EXTS:
            continue
        rel = img_path.relative_to(input_root)
        img = Image.open(img_path).convert('RGB')
        if args.input_mode == 'gray':
            img = img.convert('L')
        x = tf(img).unsqueeze(0).to(device, non_blocking=True)
        with torch.no_grad():
            pred = model(x)[0]
        save_tensor_img(pred, output_root / rel)
        print(f'Saved {output_root / rel}')


if __name__ == '__main__':
    main()
