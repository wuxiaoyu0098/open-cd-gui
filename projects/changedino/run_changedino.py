# Copyright (c) Open-CD. All rights reserved.
"""Open-CD wrapper for the external ChangeDINO project."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CHANGEDINO_ROOT = PROJECT_DIR / 'ChangeDINO'
LAUNCH_CWD = Path.cwd().resolve()


def _resolve_changedino_root(path: str | None) -> Path:
    root = Path(path or os.environ.get('CHANGEDINO_ROOT',
                                       str(DEFAULT_CHANGEDINO_ROOT))).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'ChangeDINO root not found: {root}')
    return root


def _run(root: Path, script: str, args: list[str]) -> int:
    script_path = root / script
    if not script_path.is_file():
        raise FileNotFoundError(f'ChangeDINO script not found: {script_path}')
    cmd = [sys.executable, '-u', str(script_path)] + args
    print('$ ' + ' '.join(cmd), flush=True)
    env = os.environ.copy()
    env['PYTHONPATH'] = str(root) + os.pathsep + env.get('PYTHONPATH', '')
    env.setdefault('PYTHONUTF8', '1')
    env.setdefault('PYTHONIOENCODING', 'utf-8')
    env.setdefault('NO_COLOR', '1')
    env.setdefault('PYTHONNOUSERSITE', '1')
    proc = subprocess.run(cmd, cwd=str(root), env=env)
    return int(proc.returncode)


def _resolve_launch_path(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str((LAUNCH_CWD / p).resolve())


def _train(args: argparse.Namespace) -> int:
    root = _resolve_changedino_root(args.changedino_root)
    forwarded = [
        '--gpu_ids', args.gpu_ids,
        '--name', args.name,
        '--dataroot', _resolve_launch_path(args.dataroot),
        '--dataset', args.dataset,
        '--checkpoint_dir', _resolve_launch_path(args.checkpoint_dir),
        '--result_dir', _resolve_launch_path(args.result_dir),
        '--vis_path', args.vis_path,
        '--phase', 'train',
        '--backbone', args.backbone,
        '--fpn', args.fpn,
        '--fpn_channels', str(args.fpn_channels),
        '--deform_groups', str(args.deform_groups),
        '--gamma_mode', args.gamma_mode,
        '--beta_mode', args.beta_mode,
        '--alpha', str(args.alpha),
        '--gamma', str(args.gamma),
        '--batch_size', str(args.batch_size),
        '--num_epochs', str(args.num_epochs),
        '--num_workers', str(args.num_workers),
        '--lr', str(args.lr),
        '--weight_decay', str(args.weight_decay),
        '--train_size', str(args.train_size),
        '--accelerator', args.accelerator,
        '--devices', str(args.devices),
        '--strategy', args.strategy,
        '--precision', str(args.precision),
        '--save_ckpt_mode', args.save_ckpt_mode,
    ]
    if args.val_batch_size:
        forwarded += ['--val_batch_size', str(args.val_batch_size)]
    if args.resume:
        forwarded += ['--resume', args.resume]
    return _run(root, 'trainval.py', forwarded)


def _big_infer(args: argparse.Namespace) -> int:
    root = _resolve_changedino_root(args.changedino_root)
    forwarded = [
        '--tif1', _resolve_launch_path(args.tif1),
        '--tif2', _resolve_launch_path(args.tif2),
        '--checkpoint_dir', _resolve_launch_path(args.checkpoint_dir),
        '--name', args.name,
        '--out_base_dir', _resolve_launch_path(args.out_base_dir),
        '--tile_size', str(args.tile_size),
        '--overlap', str(args.overlap),
        '--batch_size', str(args.batch_size),
        '--gpu_ids', args.gpu_ids,
        '--binarize_mode', args.binarize_mode,
        '--prob_threshold', str(args.prob_threshold),
        '--logit_threshold', str(args.logit_threshold),
        '--smooth_kernel_size', str(args.smooth_kernel_size),
        '--remove_holes_area', str(args.remove_holes_area),
        '--remove_objects_min_size', str(args.remove_objects_min_size),
        '--min_area_mu', str(args.min_area_mu),
        '--epochs', args.epochs,
    ]
    if args.no_amp:
        forwarded.append('--no_amp')
    if args.smooth_overlap:
        forwarded.append('--smooth_overlap')
    if args.fill_all_holes:
        forwarded.append('--fill_all_holes')
    if args.strict_crs:
        forwarded.append('--strict_crs')
    if args.weights:
        forwarded.append('--weights')
        forwarded.extend(args.weights)
    if args.memmap_dir:
        forwarded += ['--memmap_dir', _resolve_launch_path(args.memmap_dir)]
    if args.suppress_green_growth:
        forwarded += [
            '--suppress_green_growth',
            '--exg_delta_thr', str(args.exg_delta_thr),
            '--exg2_min_thr', str(args.exg2_min_thr),
        ]
    if args.no_auto_align_if_size_mismatch:
        forwarded.append('--no-auto_align_if_size_mismatch')
    if args.align_script:
        forwarded += ['--align_script', _resolve_launch_path(args.align_script)]
    return _run(root, '2.predict_big_pt_flow_v3.py', forwarded)


def _test(args: argparse.Namespace) -> int:
    root = _resolve_changedino_root(args.changedino_root)
    checkpoint_dir = Path(_resolve_launch_path(args.checkpoint_dir))
    name = args.name
    if args.checkpoint:
        ckpt = Path(_resolve_launch_path(args.checkpoint))
        if ckpt.is_file():
            save_dir = checkpoint_dir / name
            save_dir.mkdir(parents=True, exist_ok=True)
            expected = save_dir / f'{name}_{args.backbone}_best.pth'
            if ckpt.resolve() != expected.resolve():
                shutil.copy2(ckpt, expected)
        elif ckpt.is_dir():
            checkpoint_dir = ckpt
    forwarded = [
        '--gpu_ids', args.gpu_ids,
        '--name', name,
        '--dataroot', _resolve_launch_path(args.dataroot),
        '--dataset', args.dataset,
        '--checkpoint_dir', str(checkpoint_dir),
        '--result_dir', _resolve_launch_path(args.result_dir),
        '--vis_path', args.vis_path,
        '--batch_size', str(args.batch_size),
        '--num_workers', str(args.num_workers),
        '--train_size', str(args.train_size),
        '--test_phase', args.test_phase,
        '--backbone', args.backbone,
        '--fpn', args.fpn,
        '--fpn_channels', str(args.fpn_channels),
        '--deform_groups', str(args.deform_groups),
        '--gamma_mode', args.gamma_mode,
        '--beta_mode', args.beta_mode,
        '--alpha', str(args.alpha),
        '--gamma', str(args.gamma),
        '--lr', str(args.lr),
        '--weight_decay', str(args.weight_decay),
    ]
    return _run(root, 'test.py', forwarded)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser('ChangeDINO Open-CD wrapper')
    subparsers = parser.add_subparsers(dest='command', required=True)

    train = subparsers.add_parser('train')
    train.add_argument('--changedino-root', default=None)
    train.add_argument('--gpu-ids', default='0')
    train.add_argument('--name', default='ChangeDINO')
    train.add_argument('--dataroot', required=True)
    train.add_argument('--dataset', required=True)
    train.add_argument('--checkpoint-dir', required=True)
    train.add_argument('--result-dir', default='results/changedino')
    train.add_argument('--vis-path', default='vis')
    train.add_argument('--backbone', default='mobilenetv2')
    train.add_argument('--fpn', default='fpn')
    train.add_argument('--fpn-channels', type=int, default=128)
    train.add_argument('--deform-groups', type=int, default=4)
    train.add_argument('--gamma-mode', default='SE')
    train.add_argument('--beta-mode', default='contextgatedconv')
    train.add_argument('--alpha', type=float, default=0.25)
    train.add_argument('--gamma', type=int, default=4)
    train.add_argument('--batch-size', type=int, default=4)
    train.add_argument('--val-batch-size', type=int, default=None)
    train.add_argument('--num-epochs', type=int, default=100)
    train.add_argument('--num-workers', type=int, default=0)
    train.add_argument('--lr', type=float, default=5e-4)
    train.add_argument('--weight-decay', type=float, default=5e-4)
    train.add_argument('--train-size', type=int, default=512)
    train.add_argument('--accelerator', default='gpu')
    train.add_argument('--devices', default='1')
    train.add_argument('--strategy', default='auto')
    train.add_argument('--precision', default='16')
    train.add_argument('--save-ckpt-mode', default='state_dict')
    train.add_argument('--resume', default=None)
    train.set_defaults(func=_train)

    infer = subparsers.add_parser('big-infer')
    infer.add_argument('--changedino-root', default=None)
    infer.add_argument('--tif1', required=True)
    infer.add_argument('--tif2', required=True)
    infer.add_argument('--checkpoint-dir', required=True)
    infer.add_argument('--name', default='')
    infer.add_argument('--out-base-dir', required=True)
    infer.add_argument('--tile-size', type=int, default=512)
    infer.add_argument('--overlap', type=int, default=128)
    infer.add_argument('--batch-size', type=int, default=16)
    infer.add_argument('--gpu-ids', default='0')
    infer.add_argument('--no-amp', action='store_true')
    infer.add_argument('--binarize-mode', default='logit_diff',
                       choices=['logit_diff', 'prob'])
    infer.add_argument('--prob-threshold', type=float, default=0.7)
    infer.add_argument('--logit-threshold', type=float, default=0.5)
    infer.add_argument('--smooth-overlap', action='store_true')
    infer.add_argument('--smooth-kernel-size', type=int, default=3)
    infer.add_argument('--remove-holes-area', type=int, default=200000)
    infer.add_argument('--remove-objects-min-size', type=int, default=800)
    infer.add_argument('--fill-all-holes', action='store_true')
    infer.add_argument('--min-area-mu', type=float, default=0.1)
    infer.add_argument('--strict-crs', action='store_true')
    infer.add_argument('--weights', nargs='*', default=None)
    infer.add_argument('--epochs', default='')
    infer.add_argument('--memmap-dir', default=None)
    infer.add_argument('--suppress-green-growth', action='store_true')
    infer.add_argument('--exg-delta-thr', type=float, default=0.10)
    infer.add_argument('--exg2-min-thr', type=float, default=0.05)
    infer.add_argument('--no-auto-align-if-size-mismatch',
                       action='store_true')
    infer.add_argument('--align-script', default='')
    infer.set_defaults(func=_big_infer)

    test = subparsers.add_parser('test')
    test.add_argument('--changedino-root', default=None)
    test.add_argument('--gpu-ids', default='0')
    test.add_argument('--name', default='ChangeDINO_BCD')
    test.add_argument('--dataroot', required=True)
    test.add_argument('--dataset', required=True)
    test.add_argument('--checkpoint-dir', required=True)
    test.add_argument('--checkpoint', default='')
    test.add_argument('--result-dir', required=True)
    test.add_argument('--vis-path', default='vis')
    test.add_argument('--batch-size', type=int, default=4)
    test.add_argument('--num-workers', type=int, default=0)
    test.add_argument('--train-size', type=int, default=512)
    test.add_argument('--test-phase', default='test')
    test.add_argument('--backbone', default='mobilenetv2')
    test.add_argument('--fpn', default='fpn')
    test.add_argument('--fpn-channels', type=int, default=128)
    test.add_argument('--deform-groups', type=int, default=4)
    test.add_argument('--gamma-mode', default='SE')
    test.add_argument('--beta-mode', default='contextgatedconv')
    test.add_argument('--alpha', type=float, default=0.25)
    test.add_argument('--gamma', type=int, default=4)
    test.add_argument('--lr', type=float, default=5e-4)
    test.add_argument('--weight-decay', type=float, default=5e-4)
    test.set_defaults(func=_test)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
