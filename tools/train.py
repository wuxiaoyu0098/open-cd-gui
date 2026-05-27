# Copyright (c) Open-CD. All rights reserved.
import argparse
import logging
import os
import os.path as osp
import subprocess
import sys
from collections import OrderedDict

import torch
import mmengine.runner.runner as mmengine_runner
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from opencd.registry import RUNNERS

# Import Open-CD modules explicitly so custom registries are populated when
# this script is launched directly from the GUI or command line.
import opencd.datasets  # noqa: F401,E402
import opencd.engine.hooks  # noqa: F401,E402
import opencd.evaluation  # noqa: F401,E402
import opencd.models  # noqa: F401,E402
import opencd.visualization  # noqa: F401,E402
from mmengine.registry import HOOKS as MMENGINE_HOOKS
from mmengine.registry import VISBACKENDS as MMENGINE_VISBACKENDS
from mmengine.registry import VISUALIZERS as MMENGINE_VISUALIZERS
from opencd.engine.hooks import CDVisualizationHook
from opencd.visualization import CDLocalVisBackend, CDLocalVisualizer


def safe_collect_env():
    try:
        from mmengine.utils.dl_utils.collect_env import collect_env
        return collect_env()
    except Exception as exc:
        return OrderedDict([
            ('sys.platform', os.name),
            ('Python', os.sys.version.replace('\n', '')),
            ('PyTorch', torch.__version__),
            ('CUDA available', str(torch.cuda.is_available())),
            ('Environment collection warning', repr(exc)),
        ])


mmengine_runner.collect_env = safe_collect_env
MMENGINE_VISUALIZERS.register_module(module=CDLocalVisualizer, force=True)
MMENGINE_VISBACKENDS.register_module(module=CDLocalVisBackend, force=True)
MMENGINE_HOOKS.register_module(module=CDVisualizationHook, force=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def _add_cli_arg(cmd, name, value):
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            cmd.append(name)
        return
    cmd.extend([name, str(value)])


def _run_change3d_train(args, cfg):
    c3d_cfg = cfg.change3d
    task = str(c3d_cfg.get('task', 'bcd')).lower()
    command = 'train-mcd' if task == 'mcd' else 'train-bcd'
    project_root = osp.abspath(osp.join(osp.dirname(__file__), '..'))
    wrapper = osp.join(project_root, 'projects', 'change3d',
                       'run_change3d.py')

    work_dir = cfg.work_dir
    cmd = [sys.executable, '-u', wrapper, command]
    _add_cli_arg(cmd, '--change3d-root', c3d_cfg.get('change3d_root'))
    _add_cli_arg(cmd, '--file-root', c3d_cfg.get('file_root'))
    _add_cli_arg(cmd, '--work-dir', work_dir)
    _add_cli_arg(cmd, '--exp-name', c3d_cfg.get('exp_name'))
    max_steps = c3d_cfg.get('max_steps', None)
    if max_steps is not None:
        _add_cli_arg(cmd, '--max-steps', max_steps)
    _add_cli_arg(cmd, '--max-epochs', c3d_cfg.get('max_epochs'))
    _add_cli_arg(cmd, '--batch-size', c3d_cfg.get('batch_size'))
    _add_cli_arg(cmd, '--num-workers', c3d_cfg.get('num_workers'))
    _add_cli_arg(cmd, '--lr', c3d_cfg.get('lr'))
    _add_cli_arg(cmd, '--pretrained', c3d_cfg.get('pretrained'))
    _add_cli_arg(cmd, '--gpu-id', c3d_cfg.get('gpu_id', 0))
    _add_cli_arg(cmd, '--devices', c3d_cfg.get('devices', 1))
    _add_cli_arg(cmd, '--accelerator', c3d_cfg.get('accelerator', 'gpu'))
    _add_cli_arg(cmd, '--strategy', c3d_cfg.get('strategy', 'auto'))
    _add_cli_arg(cmd, '--precision', c3d_cfg.get('precision', 16))
    _add_cli_arg(cmd, '--val-check-interval',
                 c3d_cfg.get('val_check_interval'))
    _add_cli_arg(cmd, '--check-val-every-n-epoch',
                 c3d_cfg.get('check_val_every_n_epoch'))
    _add_cli_arg(cmd, '--train-log-interval',
                 c3d_cfg.get('train_log_interval'))
    _add_cli_arg(cmd, '--in-height', c3d_cfg.get('in_height'))
    _add_cli_arg(cmd, '--in-width', c3d_cfg.get('in_width'))

    resume_path = c3d_cfg.get('resume')
    if resume_path:
        _add_cli_arg(cmd, '--resume', resume_path)

    if command == 'train-mcd':
        _add_cli_arg(cmd, '--dataset', c3d_cfg.get('dataset', 'Change3D-MCD'))
        _add_cli_arg(cmd, '--im1-dir', c3d_cfg.get('im1_dir', 'A'))
        _add_cli_arg(cmd, '--im2-dir', c3d_cfg.get('im2_dir', 'B'))
        _add_cli_arg(cmd, '--label-dir', c3d_cfg.get('label_dir', 'label'))
        _add_cli_arg(cmd, '--num-class', c3d_cfg.get('num_class'))
        _add_cli_arg(cmd, '--ignore-index', c3d_cfg.get('ignore_index'))
        _add_cli_arg(cmd, '--class-weight-mode',
                     c3d_cfg.get('class_weight_mode'))
        _add_cli_arg(cmd, '--dice-weight', c3d_cfg.get('dice_weight'))
    else:
        _add_cli_arg(cmd, '--dataset', c3d_cfg.get('dataset', 'Change3D-CD'))
        _add_cli_arg(cmd, '--pre-dir', c3d_cfg.get('pre_dir', 'A'))
        _add_cli_arg(cmd, '--post-dir', c3d_cfg.get('post_dir', 'B'))
        _add_cli_arg(cmd, '--label-dir', c3d_cfg.get('label_dir', 'label'))

    print('$ ' + ' '.join(cmd), flush=True)
    return subprocess.run(cmd, cwd=project_root).returncode


def _run_changedino_train(args, cfg):
    cdino_cfg = cfg.changedino
    project_root = osp.abspath(osp.join(osp.dirname(__file__), '..'))
    wrapper = osp.join(project_root, 'projects', 'changedino',
                       'run_changedino.py')

    work_dir = cfg.work_dir
    cmd = [sys.executable, '-u', wrapper, 'train']
    _add_cli_arg(cmd, '--changedino-root', cdino_cfg.get('changedino_root'))
    _add_cli_arg(cmd, '--gpu-ids', cdino_cfg.get('gpu_ids', '0'))
    _add_cli_arg(cmd, '--name', cdino_cfg.get('name', 'ChangeDINO_BCD'))
    _add_cli_arg(cmd, '--dataroot', cdino_cfg.get('dataroot'))
    _add_cli_arg(cmd, '--dataset', cdino_cfg.get('dataset'))
    _add_cli_arg(cmd, '--checkpoint-dir',
                 cdino_cfg.get('checkpoint_dir', work_dir))
    _add_cli_arg(cmd, '--result-dir', cdino_cfg.get('result_dir'))
    _add_cli_arg(cmd, '--vis-path', cdino_cfg.get('vis_path'))
    _add_cli_arg(cmd, '--backbone', cdino_cfg.get('backbone'))
    _add_cli_arg(cmd, '--fpn', cdino_cfg.get('fpn'))
    _add_cli_arg(cmd, '--fpn-channels', cdino_cfg.get('fpn_channels'))
    _add_cli_arg(cmd, '--deform-groups', cdino_cfg.get('deform_groups'))
    _add_cli_arg(cmd, '--gamma-mode', cdino_cfg.get('gamma_mode'))
    _add_cli_arg(cmd, '--beta-mode', cdino_cfg.get('beta_mode'))
    _add_cli_arg(cmd, '--alpha', cdino_cfg.get('alpha'))
    _add_cli_arg(cmd, '--gamma', cdino_cfg.get('gamma'))
    _add_cli_arg(cmd, '--batch-size', cdino_cfg.get('batch_size'))
    _add_cli_arg(cmd, '--val-batch-size', cdino_cfg.get('val_batch_size'))
    _add_cli_arg(cmd, '--num-epochs', cdino_cfg.get('num_epochs'))
    _add_cli_arg(cmd, '--num-workers', cdino_cfg.get('num_workers'))
    _add_cli_arg(cmd, '--lr', cdino_cfg.get('lr'))
    _add_cli_arg(cmd, '--weight-decay', cdino_cfg.get('weight_decay'))
    _add_cli_arg(cmd, '--train-size', cdino_cfg.get('train_size'))
    _add_cli_arg(cmd, '--accelerator', cdino_cfg.get('accelerator', 'gpu'))
    _add_cli_arg(cmd, '--devices', cdino_cfg.get('devices', 1))
    _add_cli_arg(cmd, '--strategy', cdino_cfg.get('strategy', 'auto'))
    _add_cli_arg(cmd, '--precision', cdino_cfg.get('precision', 16))
    _add_cli_arg(cmd, '--save-ckpt-mode',
                 cdino_cfg.get('save_ckpt_mode', 'state_dict'))
    resume_path = cdino_cfg.get('resume')
    if resume_path:
        _add_cli_arg(cmd, '--resume', resume_path)

    print('$ ' + ' '.join(cmd), flush=True)
    return subprocess.run(cmd, cwd=project_root).returncode


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    if cfg.get('change3d', None) is not None:
        raise SystemExit(_run_change3d_train(args, cfg))
    if cfg.get('changedino', None) is not None:
        raise SystemExit(_run_changedino_train(args, cfg))

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # resume training
    cfg.resume = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()
