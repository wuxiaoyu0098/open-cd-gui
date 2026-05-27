# Copyright (c) Open-CD. All rights reserved.
"""Open-CD wrapper for the external Change3D project."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CHANGE3D_ROOT = PROJECT_DIR / 'Change3D'


def _resolve_change3d_root(path: str | None) -> Path:
    root = Path(path or os.environ.get('CHANGE3D_ROOT',
                                       str(DEFAULT_CHANGE3D_ROOT))).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Change3D root not found: {root}')
    return root


def _resolve_external_path(root: Path, path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str((root / p).resolve())


def _run(root: Path, script: str, args: list[str]) -> int:
    script_path = root / script
    if not script_path.is_file():
        raise FileNotFoundError(f'Change3D script not found: {script_path}')
    cmd = [sys.executable, '-u', str(script_path)] + args
    print('$ ' + ' '.join(cmd), flush=True)
    env = os.environ.copy()
    env['PYTHONPATH'] = str(root) + os.pathsep + env.get('PYTHONPATH', '')
    proc = subprocess.run(cmd, cwd=str(root), env=env)
    return int(proc.returncode)


def _add_common_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--change3d-root', default=None)
    parser.add_argument('--file-root', required=True)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--exp-name', default='')
    parser.add_argument('--max-steps', type=int, default=-1)
    parser.add_argument('--max-epochs', type=int, default=120)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--pretrained', default='model/X3D_L.pyth')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--devices', default='1')
    parser.add_argument('--accelerator', default='gpu')
    parser.add_argument('--strategy', default='auto')
    parser.add_argument('--precision', type=int, default=16)
    parser.add_argument('--val-check-interval', type=float, default=1.0)
    parser.add_argument('--check-val-every-n-epoch', type=int, default=1)
    parser.add_argument('--train-log-interval', type=int, default=50)
    parser.add_argument('--in-height', type=int, default=512)
    parser.add_argument('--in-width', type=int, default=512)


def _train_bcd(args: argparse.Namespace) -> int:
    root = _resolve_change3d_root(args.change3d_root)
    pretrained = _resolve_external_path(root, args.pretrained)
    forwarded = [
        '--dataset', args.dataset,
        '--file_root', args.file_root,
        '--pre_dir', args.pre_dir,
        '--post_dir', args.post_dir,
        '--label_dir', args.label_dir,
        '--in_height', str(args.in_height),
        '--in_width', str(args.in_width),
        '--num_perception_frame', '1',
        '--num_class', '1',
        '--max_epochs', str(args.max_epochs),
        '--batch_size', str(args.batch_size),
        '--num_workers', str(args.num_workers),
        '--lr', str(args.lr),
        '--pretrained', pretrained,
        '--save_dir', args.work_dir,
        '--work_dirs', args.work_dir,
        '--exp_name', args.exp_name or 'Change3D_BCD',
        '--gpu_id', str(args.gpu_id),
        '--accelerator', args.accelerator,
        '--devices', str(args.devices),
        '--strategy', args.strategy,
        '--precision', str(args.precision),
        '--val_check_interval', str(args.val_check_interval),
        '--check_val_every_n_epoch', str(args.check_val_every_n_epoch),
        '--train_log_interval', str(args.train_log_interval),
    ]
    if args.max_steps is not None:
        forwarded += ['--max_steps', str(args.max_steps)]
    if args.resume:
        forwarded += ['--resume', args.resume]
    return _run(root, r'scripts\train_BCD.py', forwarded)


def _train_mcd(args: argparse.Namespace) -> int:
    root = _resolve_change3d_root(args.change3d_root)
    pretrained = _resolve_external_path(root, args.pretrained)
    forwarded = [
        '--dataset', args.dataset,
        '--file_root', args.file_root,
        '--im1_dir', args.im1_dir,
        '--im2_dir', args.im2_dir,
        '--label_dir', args.label_dir,
        '--in_height', str(args.in_height),
        '--in_width', str(args.in_width),
        '--num_perception_frame', '1',
        '--num_class', str(args.num_class),
        '--ignore_index', str(args.ignore_index),
        '--max_epochs', str(args.max_epochs),
        '--batch_size', str(args.batch_size),
        '--num_workers', str(args.num_workers),
        '--lr', str(args.lr),
        '--pretrained', pretrained,
        '--save_dir', args.work_dir,
        '--work_dirs', args.work_dir,
        '--exp_name', args.exp_name or 'Change3D_MCD',
        '--gpu_id', str(args.gpu_id),
        '--accelerator', args.accelerator,
        '--devices', str(args.devices),
        '--strategy', args.strategy,
        '--precision', str(args.precision),
        '--val_check_interval', str(args.val_check_interval),
        '--check_val_every_n_epoch', str(args.check_val_every_n_epoch),
        '--train_log_interval', str(args.train_log_interval),
        '--class_weight_mode', args.class_weight_mode,
        '--dice_weight', str(args.dice_weight),
    ]
    if args.max_steps is not None:
        forwarded += ['--max_steps', str(args.max_steps)]
    if args.resume:
        forwarded += ['--resume', args.resume]
    return _run(root, r'scripts\train_MCD_v1.py', forwarded)


def _add_common_infer_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--change3d-root', default=None)
    parser.add_argument('--tif1', required=True)
    parser.add_argument('--tif2', required=True)
    parser.add_argument('--pt-path', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--out-name', default='change3d')
    parser.add_argument('--tile-size', type=int, default=512)
    parser.add_argument('--overlap', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--remove-holes-area', type=int, default=200000)
    parser.add_argument('--remove-objects-min-size', type=int, default=800)
    parser.add_argument('--min-area-mu', type=float, default=0.1)
    parser.add_argument('--memmap-dir', default='')


def _infer_bcd(args: argparse.Namespace) -> int:
    root = _resolve_change3d_root(args.change3d_root)
    forwarded = [
        '--tif1', args.tif1,
        '--tif2', args.tif2,
        '--pt_path', args.pt_path,
        '--out_dir', args.out_dir,
        '--out_name', args.out_name,
        '--tile_size', str(args.tile_size),
        '--overlap', str(args.overlap),
        '--batch_size', str(args.batch_size),
        '--prob_threshold', str(args.prob_threshold),
        '--remove_holes_area', str(args.remove_holes_area),
        '--remove_objects_min_size', str(args.remove_objects_min_size),
        '--min_area_mu', str(args.min_area_mu),
        '--gpu_id', str(args.gpu_id),
        '--device', args.device,
    ]
    if args.memmap_dir:
        forwarded += ['--memmap_dir', args.memmap_dir]
    return _run(root, r'scripts\predict_BCD.py', forwarded)


def _infer_mcd(args: argparse.Namespace) -> int:
    root = _resolve_change3d_root(args.change3d_root)
    forwarded = [
        '--tif1', args.tif1,
        '--tif2', args.tif2,
        '--mcd_pt_path', args.pt_path,
        '--out_dir', args.out_dir,
        '--out_name', args.out_name,
        '--tile_size', str(args.tile_size),
        '--overlap', str(args.overlap),
        '--batch_size', str(args.batch_size),
        '--num_classes', str(args.num_classes),
        '--export_class_ids', args.export_class_ids,
        '--remove_holes_area', str(args.remove_holes_area),
        '--remove_objects_min_size', str(args.remove_objects_min_size),
        '--min_area_mu', str(args.min_area_mu),
        '--gpu_id', str(args.gpu_id),
        '--device', args.device,
        '--memmap_dir', args.memmap_dir,
    ]
    return _run(root, r'scripts\predict_MCD.py', forwarded)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser('Change3D Open-CD wrapper')
    subparsers = parser.add_subparsers(dest='command', required=True)

    train_bcd = subparsers.add_parser('train-bcd')
    _add_common_train_args(train_bcd)
    train_bcd.add_argument('--dataset', default='Change3D-CD')
    train_bcd.add_argument('--pre-dir', default='A')
    train_bcd.add_argument('--post-dir', default='B')
    train_bcd.add_argument('--label-dir', default='label')
    train_bcd.set_defaults(func=_train_bcd)

    train_mcd = subparsers.add_parser('train-mcd')
    _add_common_train_args(train_mcd)
    train_mcd.add_argument('--dataset', default='Change3D-MCD')
    train_mcd.add_argument('--im1-dir', default='A')
    train_mcd.add_argument('--im2-dir', default='B')
    train_mcd.add_argument('--label-dir', default='label')
    train_mcd.add_argument('--num-class', type=int, default=31)
    train_mcd.add_argument('--ignore-index', type=int, default=255)
    train_mcd.add_argument('--class-weight-mode',
                           default='median_freq_nonzero')
    train_mcd.add_argument('--dice-weight', type=float, default=1.0)
    train_mcd.set_defaults(func=_train_mcd)

    infer_bcd = subparsers.add_parser('infer-bcd')
    _add_common_infer_args(infer_bcd)
    infer_bcd.add_argument('--prob-threshold', type=float, default=0.5)
    infer_bcd.set_defaults(func=_infer_bcd)

    infer_mcd = subparsers.add_parser('infer-mcd')
    _add_common_infer_args(infer_mcd)
    infer_mcd.add_argument('--num-classes', type=int, default=31)
    infer_mcd.add_argument('--export-class-ids', default='')
    infer_mcd.set_defaults(func=_infer_mcd)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
