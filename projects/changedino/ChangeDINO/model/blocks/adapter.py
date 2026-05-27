import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import math

REPO_DIR = "dinov3"
DINO_NAME = "dinov3_vitl16"
MODEL_TO_NUM_LAYERS = {
    "VITS": 12,
    "VITSP": 12,
    "VITB": 12,
    "VITL": 24,
    "VITHP": 32,
    "VIT7B": 40,
}


class DINOV3Wrapper(nn.Module):
    def __init__(
        self,
        weights_path="dinov3/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
        extract_ids=[5, 11, 17, 23],
        device="cuda",
    ):
        super().__init__()
        self.device = device
        self.model = torch.hub.load(
            REPO_DIR,
            DINO_NAME,
            source="local",
            weights=weights_path,
        )
        self.model = self.model.eval().to(device)
        self.n_layers = MODEL_TO_NUM_LAYERS[
            re.sub(r"\d+", "", DINO_NAME.split("_")[-1]).upper()
        ]
        self.patch_size = int(re.findall(r"\d+", DINO_NAME.split("_")[-1])[-1])
        self.extract_ids = extract_ids

        # freeze the backbone
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x):
        # 动态计算合适的尺寸，确保是 patch_size 的倍数
        B, C, H, W = x.shape
        # 计算最接近的 patch_size 倍数
        target_h = ((H + self.patch_size - 1) // self.patch_size) * self.patch_size
        target_w = ((W + self.patch_size - 1) // self.patch_size) * self.patch_size
        # 如果尺寸已经是 patch_size 的倍数，则不插值
        if target_h != H or target_w != W:
            x = F.interpolate(
                x, size=(target_h, target_w), mode="bilinear", align_corners=True, antialias=True
            )
        with torch.no_grad():
            with torch.autocast(device_type=self.device, dtype=torch.float32):
                feats = self.model.get_intermediate_layers(
                    x, n=range(self.n_layers), reshape=True, norm=True
                )
                feats_ = []
                for i in range(len(self.extract_ids)):
                    feats_.append(feats[self.extract_ids[i]])  # [B, N, C]
        return feats_


class SepAdapterBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, r: int = 64, act=nn.SiLU):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_dim, r, kernel_size=1, bias=False),
            nn.BatchNorm2d(r),
            act(inplace=True),
        )
        self.dw = nn.Sequential(
            nn.Conv2d(
                r, r, kernel_size=3, padding=1, groups=r, bias=False
            ),  # depthwise
            nn.BatchNorm2d(r),
            act(inplace=True),
        )
        self.proj = nn.Conv2d(r, out_dim, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.reduce(x)
        x = self.dw(x)
        x = self.proj(x)
        return x


class DenseAdapterLite(nn.Module):
    def __init__(
        self,
        in_dim=1024,
        out_dim=256,
        sizes=(64, 32, 16, 8),
        bottleneck=64,
        share=False,
    ):
        super().__init__()
        self.sizes = list(sizes)  # 保留作为默认值，但实际使用时会动态调整
        if share:
            self.blocks = nn.ModuleList(
                [SepAdapterBlock(in_dim, out_dim, r=bottleneck)]
            )
        else:
            self.blocks = nn.ModuleList(
                [SepAdapterBlock(in_dim, out_dim, r=bottleneck) for _ in self.sizes]
            )
        self.share = share

    def forward(self, feats, target_sizes=None):
        """
        feats: list of 4 tensors, each [B, N, C] or [B, C, H, W] (from DINO)
        target_sizes: list of 4 tuples (H, W) for target spatial sizes, if None use self.sizes
        return: list of 4 tensors, each [B, out_dim, H_i, W_i]
        """
        outs = []
        for i, x in enumerate(feats):
            # 检查输入格式：可能是 [B, N, C] 或 [B, C, H, W]
            if len(x.shape) == 3:
                # [B, N, C] 格式，需要 reshape 到 [B, C, H, W]
                B, N, C = x.shape
                # 计算空间尺寸（DINO 输出通常是正方形，因为输入会被调整）
                # 如果 N 是完全平方数，则 H=W=sqrt(N)，否则尝试推断
                sqrt_N = int(math.sqrt(N))
                if sqrt_N * sqrt_N == N:
                    H = W = sqrt_N
                else:
                    # 如果不是完全平方数，尝试找到最接近的因子
                    # 这里假设是正方形，使用最接近的整数
                    H = W = sqrt_N
                x = x.reshape(B, C, H, W)
            elif len(x.shape) == 4:
                # 已经是 [B, C, H, W] 格式，直接使用
                pass
            else:
                raise ValueError(f"Unexpected tensor shape: {x.shape}, expected 3D [B, N, C] or 4D [B, C, H, W]")
            
            # 使用目标尺寸或默认尺寸
            if target_sizes is not None:
                target_h, target_w = target_sizes[i]
            else:
                target_h = target_w = self.sizes[i]
            
            x = F.interpolate(
                x,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            block = self.blocks[0] if self.share else self.blocks[i]
            outs.append(block(x))
        return outs
