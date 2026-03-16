from __future__ import annotations
from train_enhancement import build_model

import argparse

import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config

from lowlight_project.datasets import CombinedPairedDatasets, build_loli_street_datasets, build_lol_datasets
from lowlight_project.metrics import psnr, ssim
from lowlight_project.models import build_enhancer
from lowlight_project.utils import amp_enabled, dataloader_kwargs, ensure_dir, get_device, print_runtime_banner, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Evaluate enhancement model on LOL / LoLI-Street splits')
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--model', type=str, required=True, choices=['shallow', 'deep', 'unet', 'resunet'])
    p.add_argument('--lol_root', type=str, default=config.LOL_PATH)
    p.add_argument('--loli_root', type=str, default=config.LOLI_PATH)
    p.add_argument('--input_mode', type=str, default='rgb', choices=['rgb', 'gray'])
    p.add_argument('--image_size', type=int, default=config.IMAGE_SIZE)
    p.add_argument('--batch_size', type=int, default=config.BATCH_SIZE)
    p.add_argument('--save_dir', type=str, required=True)
    p.add_argument('--device', type=str, default='')
    p.add_argument('--num_workers', type=int, default=config.NUM_WORKERS)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    save_dir = ensure_dir(args.save_dir)
    device = get_device(args.device or None)
    print_runtime_banner(device)
    use_amp = amp_enabled(device)

    datasets_by_name = {}
    if args.lol_root:
        datasets_by_name['lol'] = build_lol_datasets(args.lol_root, args.image_size, args.input_mode)
    if args.loli_root:
        datasets_by_name['loli'] = build_loli_street_datasets(args.loli_root, args.image_size, args.input_mode)
    combined = CombinedPairedDatasets(datasets_by_name)
    test_ds = combined.get_split('test')
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **dataloader_kwargs(args.num_workers))

    channels = 1 if args.input_mode == 'gray' else 3
    model = build_model(args.model, args.input_mode).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt.get("model_state"))
    if state_dict is None:
        raise KeyError(f"Checkpoint does not contain model weights. Keys found: {list(ckpt.keys())}")
    model.load_state_dict(state_dict)
    model.eval()

    psnr_values, ssim_values = [], []
    preview = []
    with torch.no_grad():
        for low, high, _ in loader:
            low = low.to(device, non_blocking=True)
            high = high.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                pred = model(low)
            for i in range(pred.shape[0]):
                psnr_values.append(psnr(pred[i:i+1].float(), high[i:i+1].float()))
                ssim_values.append(ssim(pred[i:i+1].float(), high[i:i+1].float()))
                if len(preview) < 12:
                    preview.extend([low[i].cpu(), pred[i].float().cpu(), high[i].cpu()])

    metrics = {
        'psnr': sum(psnr_values) / max(1, len(psnr_values)),
        'ssim': sum(ssim_values) / max(1, len(ssim_values)),
        'num_samples': len(psnr_values),
        'device': str(device),
        'gpu_name': config.pretty_device_name(),
    }
    save_json(metrics, save_dir / 'metrics.json')
    print(metrics)

    if preview:
        save_image(make_grid(preview, nrow=3), save_dir / 'qualitative_grid.png')


if __name__ == '__main__':
    main()
