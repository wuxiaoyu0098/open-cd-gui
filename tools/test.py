# Copyright (c) Open-CD. All rights reserved.
import argparse
import os
import os.path as osp
import subprocess
import sys
from collections import OrderedDict

# PyTorch >= 2.6 defaults torch.load(weights_only=True), which cannot load
# MMEngine checkpoints containing training metadata such as HistoryBuffer.
os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')

import torch
import mmengine.runner.runner as mmengine_runner
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

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


# TODO: support fuse_conv_bn, visualization, and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='Open-CD test (and eval) a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help=('if specified, the evaluation metric results will be dumped'
              'into the directory as json'))
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
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
    parser.add_argument(
        '--tta', action='store_true', help='Test time augmentation')
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


def _run_change3d_test(args, cfg):
    c3d_cfg = cfg.change3d
    task = str(c3d_cfg.get('task', 'bcd')).lower()
    infer_cfg = c3d_cfg.get('infer', {})
    command = 'infer-mcd' if task == 'mcd' else 'infer-bcd'
    project_root = osp.abspath(osp.join(osp.dirname(__file__), '..'))
    wrapper = osp.join(project_root, 'projects', 'change3d',
                       'run_change3d.py')

    out_dir = args.show_dir or args.work_dir or infer_cfg.get('out_dir')
    if not out_dir:
        out_dir = osp.join(project_root, 'results', 'change3d', command)

    cmd = [sys.executable, '-u', wrapper, command]
    _add_cli_arg(cmd, '--change3d-root', c3d_cfg.get('change3d_root'))
    _add_cli_arg(cmd, '--tif1', infer_cfg.get('tif1'))
    _add_cli_arg(cmd, '--tif2', infer_cfg.get('tif2'))
    _add_cli_arg(cmd, '--pt-path', args.checkpoint)
    _add_cli_arg(cmd, '--out-dir', out_dir)
    _add_cli_arg(cmd, '--out-name', infer_cfg.get('out_name'))
    _add_cli_arg(cmd, '--tile-size', infer_cfg.get('tile_size'))
    _add_cli_arg(cmd, '--overlap', infer_cfg.get('overlap'))
    _add_cli_arg(cmd, '--batch-size', infer_cfg.get('batch_size'))
    _add_cli_arg(cmd, '--gpu-id', infer_cfg.get('gpu_id', 0))
    _add_cli_arg(cmd, '--device', infer_cfg.get('device', 'cuda'))
    _add_cli_arg(cmd, '--remove-holes-area',
                 infer_cfg.get('remove_holes_area'))
    _add_cli_arg(cmd, '--remove-objects-min-size',
                 infer_cfg.get('remove_objects_min_size'))
    _add_cli_arg(cmd, '--min-area-mu', infer_cfg.get('min_area_mu'))
    _add_cli_arg(cmd, '--memmap-dir', infer_cfg.get('memmap_dir'))

    if command == 'infer-mcd':
        _add_cli_arg(cmd, '--num-classes', infer_cfg.get('num_classes'))
        _add_cli_arg(cmd, '--export-class-ids',
                     infer_cfg.get('export_class_ids'))
    else:
        _add_cli_arg(cmd, '--prob-threshold',
                     infer_cfg.get('prob_threshold'))

    print('$ ' + ' '.join(cmd), flush=True)
    return subprocess.run(cmd, cwd=project_root).returncode


def _run_changedino_test(args, cfg):
    cdino_cfg = cfg.changedino
    infer_cfg = cdino_cfg.get('infer', {})
    project_root = osp.abspath(osp.join(osp.dirname(__file__), '..'))
    wrapper = osp.join(project_root, 'projects', 'changedino',
                       'run_changedino.py')

    if infer_cfg.get('tif1') and infer_cfg.get('tif2'):
        return _run_changedino_big_infer(args, cfg, wrapper, infer_cfg,
                                         project_root)

    out_dir = args.show_dir or args.work_dir or cdino_cfg.get('result_dir')
    if not out_dir:
        out_dir = osp.join(project_root, 'results', 'changedino', 'test')

    checkpoint_dir = cdino_cfg.get('checkpoint_dir') or osp.dirname(
        args.checkpoint)

    cmd = [sys.executable, '-u', wrapper, 'test']
    _add_cli_arg(cmd, '--changedino-root', cdino_cfg.get('changedino_root'))
    _add_cli_arg(cmd, '--gpu-ids', cdino_cfg.get('gpu_ids', '0'))
    _add_cli_arg(cmd, '--name', cdino_cfg.get('name', 'ChangeDINO_BCD'))
    _add_cli_arg(cmd, '--dataroot', cdino_cfg.get('dataroot'))
    _add_cli_arg(cmd, '--dataset', cdino_cfg.get('dataset'))
    _add_cli_arg(cmd, '--checkpoint-dir', checkpoint_dir)
    _add_cli_arg(cmd, '--checkpoint', args.checkpoint)
    _add_cli_arg(cmd, '--result-dir', out_dir)
    _add_cli_arg(cmd, '--vis-path', cdino_cfg.get('vis_path'))
    _add_cli_arg(cmd, '--batch-size', cdino_cfg.get('batch_size'))
    _add_cli_arg(cmd, '--num-workers', cdino_cfg.get('num_workers'))
    _add_cli_arg(cmd, '--train-size', cdino_cfg.get('train_size'))
    _add_cli_arg(cmd, '--test-phase', cdino_cfg.get('test_phase', 'test'))
    _add_cli_arg(cmd, '--backbone', cdino_cfg.get('backbone'))
    _add_cli_arg(cmd, '--fpn', cdino_cfg.get('fpn'))
    _add_cli_arg(cmd, '--fpn-channels', cdino_cfg.get('fpn_channels'))
    _add_cli_arg(cmd, '--deform-groups', cdino_cfg.get('deform_groups'))
    _add_cli_arg(cmd, '--gamma-mode', cdino_cfg.get('gamma_mode'))
    _add_cli_arg(cmd, '--beta-mode', cdino_cfg.get('beta_mode'))
    _add_cli_arg(cmd, '--alpha', cdino_cfg.get('alpha'))
    _add_cli_arg(cmd, '--gamma', cdino_cfg.get('gamma'))
    _add_cli_arg(cmd, '--lr', cdino_cfg.get('lr'))
    _add_cli_arg(cmd, '--weight-decay', cdino_cfg.get('weight_decay'))

    print('$ ' + ' '.join(cmd), flush=True)
    return subprocess.run(cmd, cwd=project_root).returncode


def _run_changedino_big_infer(args, cfg, wrapper, infer_cfg, project_root):
    cdino_cfg = cfg.changedino

    out_dir = args.show_dir or args.work_dir or infer_cfg.get('out_base_dir')
    if not out_dir:
        out_dir = osp.join(project_root, 'results', 'changedino', 'big_tif')

    checkpoint_dir = infer_cfg.get('checkpoint_dir') or osp.dirname(
        args.checkpoint)
    weights = infer_cfg.get('weights') or []
    if args.checkpoint and args.checkpoint != '__auto__' and osp.isfile(
            args.checkpoint):
        weights = [args.checkpoint]

    cmd = [sys.executable, '-u', wrapper, 'big-infer']
    _add_cli_arg(cmd, '--changedino-root', cdino_cfg.get('changedino_root'))
    _add_cli_arg(cmd, '--tif1', infer_cfg.get('tif1'))
    _add_cli_arg(cmd, '--tif2', infer_cfg.get('tif2'))
    _add_cli_arg(cmd, '--checkpoint-dir', checkpoint_dir)
    _add_cli_arg(cmd, '--name', infer_cfg.get('name', cdino_cfg.get('name',
                                                                    '')))
    _add_cli_arg(cmd, '--out-base-dir', out_dir)
    _add_cli_arg(cmd, '--tile-size', infer_cfg.get('tile_size'))
    _add_cli_arg(cmd, '--overlap', infer_cfg.get('overlap'))
    _add_cli_arg(cmd, '--batch-size', infer_cfg.get('batch_size'))
    _add_cli_arg(cmd, '--gpu-ids', infer_cfg.get('gpu_ids', '0'))
    _add_cli_arg(cmd, '--binarize-mode', infer_cfg.get('binarize_mode'))
    _add_cli_arg(cmd, '--prob-threshold', infer_cfg.get('prob_threshold'))
    _add_cli_arg(cmd, '--logit-threshold', infer_cfg.get('logit_threshold'))
    _add_cli_arg(cmd, '--smooth-kernel-size',
                 infer_cfg.get('smooth_kernel_size'))
    _add_cli_arg(cmd, '--remove-holes-area',
                 infer_cfg.get('remove_holes_area'))
    _add_cli_arg(cmd, '--remove-objects-min-size',
                 infer_cfg.get('remove_objects_min_size'))
    _add_cli_arg(cmd, '--min-area-mu', infer_cfg.get('min_area_mu'))
    _add_cli_arg(cmd, '--epochs', infer_cfg.get('epochs', ''))
    _add_cli_arg(cmd, '--memmap-dir', infer_cfg.get('memmap_dir'))
    _add_cli_arg(cmd, '--align-script', infer_cfg.get('align_script'))
    if infer_cfg.get('no_amp', True):
        cmd.append('--no-amp')
    if infer_cfg.get('smooth_overlap', False):
        cmd.append('--smooth-overlap')
    if infer_cfg.get('fill_all_holes', False):
        cmd.append('--fill-all-holes')
    if infer_cfg.get('strict_crs', False):
        cmd.append('--strict-crs')
    if weights:
        cmd.append('--weights')
        cmd.extend([str(w) for w in weights])
    if infer_cfg.get('suppress_green_growth', False):
        cmd.append('--suppress-green-growth')
        _add_cli_arg(cmd, '--exg-delta-thr', infer_cfg.get('exg_delta_thr'))
        _add_cli_arg(cmd, '--exg2-min-thr', infer_cfg.get('exg2_min_thr'))
    if infer_cfg.get('no_auto_align_if_size_mismatch', False):
        cmd.append('--no-auto-align-if-size-mismatch')

    print('$ ' + ' '.join(cmd), flush=True)
    return subprocess.run(cmd, cwd=project_root).returncode


def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Turn on visualization
        visualization_hook['draw'] = True
        if args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = args.wait_time
        if args.show_dir:
            visulizer = cfg.visualizer
            visulizer['save_dir'] = args.show_dir
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')

    return cfg


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
        raise SystemExit(_run_change3d_test(args, cfg))
    if cfg.get('changedino', None) is not None:
        raise SystemExit(_run_changedino_test(args, cfg))

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.tta_model.module = cfg.model
        cfg.model = cfg.tta_model

    # build the runner from config
    runner = Runner.from_cfg(cfg)

    # start testing
    runner.test()


if __name__ == '__main__':
    main()
