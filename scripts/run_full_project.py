from __future__ import annotations

import argparse
from pathlib import Path

import sys
import os
import subprocess

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import config

def run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Run the full CS615 project pipeline')
    p.add_argument('--lol_root', default=config.LOL_PATH)
    p.add_argument('--loli_root', default=config.LOLI_PATH)
    p.add_argument('--exdark_root', default=config.EXDARK_PATH)
    p.add_argument('--out_root', default=str(config.RUNS_DIR / 'full_project'))
    p.add_argument('--epochs_enhance', type=int, default=config.EPOCHS_ENHANCEMENT)
    p.add_argument('--epochs_classifier', type=int, default=config.EPOCHS_CLASSIFIER)
    p.add_argument('--batch_size', type=int, default=config.BATCH_SIZE)
    p.add_argument('--device', default='')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)

    common_enhance = [
        sys.executable, 'scripts/train_enhancement.py',
        '--lol_root', args.lol_root,
        '--loli_root', args.loli_root,
        '--epochs', str(args.epochs_enhance),
        '--batch_size', str(args.batch_size),
    ]
    if args.device:
        common_enhance.extend(['--device', args.device])

    for model_name, input_mode in [
        ('shallow', 'rgb'),
        ('deep', 'rgb'),
        ('unet', 'rgb'),
        ('resunet', 'rgb'),
        ('deep', 'gray'),
        ('unet', 'gray'),
    ]:
        save_dir = root / f'enhance_{model_name}_{input_mode}'
        cmd = common_enhance + ['--model', model_name, '--input_mode', input_mode, '--save_dir', str(save_dir)]
        run(cmd)

    best_rgb_model = 'resunet'
    best_ckpt = root / 'enhance_resunet_rgb_retry' / 'checkpoints' / 'best.pt'
    eval_cmd = [
        sys.executable, 'scripts/evaluate_enhancement.py',
        '--checkpoint', str(best_ckpt),
        '--model', best_rgb_model,
        '--lol_root', args.lol_root,
        '--loli_root', args.loli_root,
        '--input_mode', 'rgb',
        '--save_dir', str(root / 'eval_best_rgb'),
    ]
    if args.device:
        eval_cmd.extend(['--device', args.device])
    run(eval_cmd)

    sample_cmd = [
        sys.executable, 'scripts/export_sample_pairs.py',
        '--lol_root', args.lol_root,
        '--loli_root', args.loli_root,
        '--save_path', str(root / 'sample_pairs.png'),
    ]
    run(sample_cmd)

    enhance_exdark_cmd = [
        sys.executable, 'scripts/enhance_exdark.py',
        '--checkpoint', str(best_ckpt),
        '--model', best_rgb_model,
        '--input_root', args.exdark_root,
        '--output_root', str(root / 'exdark_enhanced'),
        '--input_mode', 'rgb',
    ]
    if args.device:
        enhance_exdark_cmd.extend(['--device', args.device])
    run(enhance_exdark_cmd)

    for data_root, tag in [(args.exdark_root, 'raw'), (str(root / 'exdark_enhanced'), 'enhanced')]:
        clf_cmd = [
            sys.executable, 'scripts/train_classifier.py',
            '--class_root', data_root,
            '--epochs', str(args.epochs_classifier),
            '--save_dir', str(root / f'classifier_{tag}'),
            '--model', 'simple',
        ]
        if args.device:
            clf_cmd.extend(['--device', args.device])
        run(clf_cmd)


if __name__ == '__main__':
    main()
