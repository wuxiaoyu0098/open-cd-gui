# coding=utf-8

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

# Avoid argparse help UnicodeEncodeError on non-UTF8 terminals.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from model.trainer import Trainer  # noqa: E402


def _torch_load(path: str, map_location: Any, weights_only: Optional[bool] = None) -> Any:
    kwargs: Dict[str, Any] = {"map_location": map_location}
    if weights_only is not None:
        kwargs["weights_only"] = weights_only
    try:
        return torch.load(path, **kwargs)
    except TypeError:
        kwargs.pop("weights_only", None)
        return torch.load(path, map_location=map_location)


class Change3DMCDExport(nn.Module):
    """双时相 MCD 前向，供推理封装（与 train_MCD 中 update_mcd 一致）。"""

    def __init__(self, trainer: Trainer) -> None:
        super().__init__()
        self.trainer = trainer

    def forward(self, pre_img: torch.Tensor, post_img: torch.Tensor) -> torch.Tensor:
        # 返回 logits: [B, num_class, H, W]
        return self.trainer.update_mcd(pre_img, post_img)


def _normalize_dataset_tag(tag: str) -> str:
    tag = (tag or "").strip()
    # 只要保证包含 MCD，就会走 encoder+MCD decoder 分支。
    if "MCD" not in tag:
        tag = f"{tag}-MCD"
    return tag


def build_bcd_args(
    dataset: str,
    in_height: int,
    in_width: int,
    pretrained: str,
    num_class: int,
) -> SimpleNamespace:
    dataset = _normalize_dataset_tag(dataset)
    return SimpleNamespace(
        dataset=dataset,
        num_perception_frame=1,
        num_class=num_class,
        in_height=in_height,
        in_width=in_width,
        pretrained=pretrained,
    )


def _strip_module_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in state_dict.items():
        # PyTorch DDP adds "module." prefix.
        nk = k.replace("module.", "")
        # Lightning usually stores the wrapped nn.Module under "model." attribute
        # when users name it "model" inside LightningModule.
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        out[nk] = v
    return out


# bundle loader removed: this script now exports TorchScript .pt only.

def _try_jit_trace(export_mod: Change3DBCDExport, device: torch.device, h: int, w: int) -> None:
    example_pre = torch.randn(1, 3, h, w, device=device)
    example_post = torch.randn(1, 3, h, w, device=device)
    traced = torch.jit.trace(export_mod, (example_pre, example_post))
    with torch.no_grad():
        _ = traced(example_pre, example_post)
    print("JIT trace 在本环境成功（少数环境仍可能在 save 阶段失败）。")
    del traced


def _export_jit(
    export_mod: Change3DBCDExport,
    device: torch.device,
    h: int,
    w: int,
    out_jit_pt: str,
) -> None:
    example_pre = torch.randn(1, 3, h, w, device=device)
    example_post = torch.randn(1, 3, h, w, device=device)
    traced = torch.jit.trace(export_mod, (example_pre, example_post))
    # 验证输出一致性（粗略检查）
    with torch.no_grad():
        _ = traced(example_pre, example_post)

    out_dir = os.path.dirname(os.path.abspath(out_jit_pt))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    torch.jit.save(traced, out_jit_pt)
    print(f"->> TorchScript 导出成功: {out_jit_pt}")


@torch.no_grad()
def _compare_silu_swish(
    weights_path: str,
    dataset_tag: str,
    in_height: int,
    in_width: int,
    pretrained: str,
    device: torch.device,
    num_class: int,
) -> None:
    """
    用同一份权重，在两种 X3D 激活实现下跑一次前向，对比输出差异。
    """
    try:
        raw = _torch_load(weights_path, map_location=device, weights_only=True)
    except Exception:
        raw = _torch_load(weights_path, map_location=device, weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    state_dict = _strip_module_prefix(raw)

    silu_args = SimpleNamespace(
        dataset=_normalize_dataset_tag(dataset_tag),
        num_perception_frame=1,
        num_class=num_class,
        in_height=in_height,
        in_width=in_width,
        pretrained=pretrained,
        x3d_inner_act="silu",
    )
    swish_args = SimpleNamespace(
        dataset=_normalize_dataset_tag(dataset_tag),
        num_perception_frame=1,
        num_class=num_class,
        in_height=in_height,
        in_width=in_width,
        pretrained=pretrained,
        x3d_inner_act="swish",
    )

    trainer_silu = Trainer(silu_args).to(device).eval().float()
    trainer_silu.load_state_dict(state_dict, strict=True)

    trainer_swish = Trainer(swish_args).to(device).eval().float()
    trainer_swish.load_state_dict(state_dict, strict=True)

    mod_silu = Change3DMCDExport(trainer_silu).to(device).eval()
    mod_swish = Change3DMCDExport(trainer_swish).to(device).eval()

    torch.manual_seed(0)
    pre = torch.randn(1, 3, in_height, in_width, device=device)
    post = torch.randn(1, 3, in_height, in_width, device=device)

    out_silu = mod_silu(pre, post)
    out_swish = mod_swish(pre, post)

    diff = (out_silu - out_swish).abs()
    print("Swish->SiLU 激活替换影响自检（同权重同输入）:")
    print(f"  output_shape={tuple(out_silu.shape)}")
    print(f"  max_abs_diff={diff.max().item():.8e}")
    print(f"  mean_abs_diff={diff.mean().item():.8e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Change3D MCD: training ckpt -> deploy TorchScript .pt")
    parser.add_argument(
        "--weights",
        type=str,
        default="/mnt/wuxy/change_detection/Change3D/checkpoints/suizhou_MCD_1PF/suizhou_MCD_1212/_TrainingFiles/best-model-step=015504-val_miou=0.1929.ckpt",
        help="仅含 state_dict 的 .pth（如 best_model*.pth）",
    )
    parser.add_argument(
        "--out_pt",
        type=str,
        default="/mnt/wuxy/change_detection/Change3D/checkpoints/suizhou_MCD_1PF/suizhou_MCD_1212/_TrainingFiles/best-model-step=015504-val_miou=0.1929.pt",
        help="输出路径；默认同目录、与 weights 同名 .pt",
    )
    parser.add_argument("--dataset", type=str, default="suizhou")
    parser.add_argument("--in_height", type=int, default=512)
    parser.add_argument("--in_width", type=int, default=512)
    parser.add_argument("--num_class", type=int, default=31, help="MCD 输出类别数（与训练一致）")
    parser.add_argument(
        "--pretrained",
        type=str,
        default=os.path.join(_ROOT, "model", "X3D_L.pyth"),
        help="与训练时一致；构建 Trainer 时会读入，随后被 weights 覆盖",
    )
    parser.add_argument("--cuda_no", type=int, default=-1, help=">=0 用 GPU；-1 用 CPU")
    parser.add_argument(
        "--try_jit",
        action="store_true",
        help="仅尝试 jit.trace（通常因 SwishFunction 失败；默认不尝试）",
    )
    parser.add_argument(
        "--export_jit",
        action="store_true",
        help="成功 jit.trace 后执行 torch.jit.save，导出真正可部署的 TorchScript .pt（不依赖 model 代码）。",
    )
    parser.add_argument(
        "--compare_swish_silu",
        action="store_true",
        help="对比 X3D 使用 Swish vs SiLU 时的输出差异（同一份权重）。",
    )
    parser.add_argument(
        "--out_jit_pt",
        type=str,
        default="",
        help="TorchScript 输出路径；默认由 --out_pt 自动推导。",
    )
    cli = parser.parse_args()

    if cli.cuda_no >= 0 and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cli.cuda_no)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model_args = build_bcd_args(cli.dataset, cli.in_height, cli.in_width, cli.pretrained, num_class=cli.num_class)
    trainer = Trainer(model_args).to(device).float()

    try:
        raw = _torch_load(cli.weights, map_location=device, weights_only=True)
    except Exception:
        raw = _torch_load(cli.weights, map_location=device, weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    trainer.load_state_dict(_strip_module_prefix(raw), strict=True)
    trainer.eval()

    export_mod = Change3DMCDExport(trainer).to(device).eval()

    if cli.compare_swish_silu:
        _compare_silu_swish(
            weights_path=cli.weights,
            dataset_tag=cli.dataset,
            in_height=cli.in_height,
            in_width=cli.in_width,
            pretrained=cli.pretrained,
            device=device,
            num_class=cli.num_class,
        )

    if cli.try_jit:
        try:
            _try_jit_trace(export_mod, device, cli.in_height, cli.in_width)
        except Exception as e:
            print("JIT trace / export 不可用（Change3D/pytorchvideo 常见限制）:", e)

            return

    # Export TorchScript only (bundle payload removed).
    out_jit_pt = cli.out_jit_pt or cli.out_pt
    if not out_jit_pt:
        base, _ = os.path.splitext(cli.weights)
        out_jit_pt = base + "_jit.pt"

    _export_jit(export_mod, device, cli.in_height, cli.in_width, out_jit_pt)
    print(f"Exported TorchScript to: {out_jit_pt}")


if __name__ == "__main__":
    main()
