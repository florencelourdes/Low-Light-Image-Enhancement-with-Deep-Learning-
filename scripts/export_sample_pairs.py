from __future__ import annotations

import argparse
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config

from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from lowlight_project.datasets import CombinedPairedDatasets, build_loli_street_datasets, build_lol_datasets
from lowlight_project.utils import dataloader_kwargs, ensure_dir


def parse_args():
    p = argparse.ArgumentParser(description='Export sample low/high pairs for presentation slides')
    p.add_argument('--lol_root', default=config.LOL_PATH)
    p.add_argument('--loli_root', default=config.LOLI_PATH)
    p.add_argument('--input_mode', default='rgb', choices=['rgb', 'gray'])
    p.add_argument('--image_size', type=int, default=config.IMAGE_SIZE)
    p.add_argument('--split', default='train', choices=['train', 'val', 'test'])
    p.add_argument('--num_pairs', type=int, default=4)
    p.add_argument('--save_path', required=True)
    return p.parse_args()


def main():
    args = parse_args()
    datasets_by_name = {}
    if args.lol_root:
        datasets_by_name['lol'] = build_lol_datasets(args.lol_root, args.image_size, args.input_mode)
    if args.loli_root:
        datasets_by_name['loli'] = build_loli_street_datasets(args.loli_root, args.image_size, args.input_mode)
    combined = CombinedPairedDatasets(datasets_by_name)
    ds = combined.get_split(args.split)
    loader = DataLoader(ds, batch_size=args.num_pairs, shuffle=False, **dataloader_kwargs(0))
    low, high, _ = next(iter(loader))
    rows = []
    for i in range(min(args.num_pairs, low.shape[0])):
        rows.extend([low[i], high[i]])
    save_path = Path(args.save_path)
    ensure_dir(save_path.parent)
    save_image(make_grid(rows, nrow=2), save_path)


if __name__ == '__main__':
    main()
