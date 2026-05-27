import os
import sys
import time
import numpy as np
from argparse import ArgumentParser

import cv2
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import lightning as L
from lightning import Trainer as LTrainer
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
    ModelSummary,
)
from lightning.pytorch.loggers import CSVLogger

sys.path.insert(0, ".")

import data.dataset as RSDataset
import data.transforms as RSTransforms

from model.trainer import Trainer
from model.utils import CrossEntropyLoss2d, Evaluator


class ProgressPrintCallback(Callback):
    """Print a few training/validation lines to stdout (rank 0 only)."""

    def __init__(self, train_every_n_steps: int = 50):
        self.train_every_n_steps = max(1, int(train_every_n_steps))
        self._epoch_t0: float | None = None

    def on_train_epoch_start(self, trainer, pl_module):
        if getattr(trainer, "global_rank", 0) != 0:
            return
        self._epoch_t0 = time.time()
        total = getattr(trainer, "num_training_batches", 0)
        print(f"[PROGRESS] epoch={trainer.current_epoch} step=0/{total} loss=nan", flush=True)

    def on_train_epoch_end(self, trainer, pl_module):
        if getattr(trainer, "global_rank", 0) != 0:
            return
        if self._epoch_t0 is None:
            return
        dt = time.time() - self._epoch_t0
        step = int(getattr(trainer, "global_step", 0))
        print(
            f"[time] epoch={trainer.current_epoch} global_step={step} "
            f"epoch_seconds={dt:.2f} epoch_minutes={dt/60.0:.2f}",
            flush=True,
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if getattr(trainer, "global_rank", 0) != 0:
            return
        total = int(getattr(trainer, "num_training_batches", 0) or 0)
        step = int(batch_idx) + 1
        if step % self.train_every_n_steps != 0 and step != total:
            return
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        if loss is None:
            return
        if torch.is_tensor(loss):
            loss = float(loss.detach())
        else:
            loss = float(loss)
        print(
            f"[PROGRESS] epoch={trainer.current_epoch} "
            f"step={step}/{total} loss={loss:.4f}",
            flush=True,
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        if getattr(trainer, "global_rank", 0) != 0:
            return
        m = trainer.callback_metrics
        miou = m.get("val_miou")
        if miou is None:
            return
        v = miou.item() if torch.is_tensor(miou) else float(miou)
        vl = m.get("val_loss")
        extra = ""
        if vl is not None:
            extra = f" val_loss={float(vl.item() if torch.is_tensor(vl) else vl):.6f}"
        print(f"[val] epoch={trainer.current_epoch} val_miou={v:.6f}{extra}", flush=True)


def _map_invalid_labels(labels: torch.Tensor, num_class: int, ignore_index: int):
    invalid = (labels != ignore_index) & ((labels < 0) | (labels >= num_class))
    if invalid.any():
        labels = labels.clone()
        labels[invalid] = ignore_index
    return labels, invalid


def _compute_class_weight_median_freq_nonzero(
    label_dir: str,
    num_class: int,
    ignore_index: int,
    bg_weight: float,
    max_weight: float,
    min_weight: float = 0.1,
):
    """
    Compute class weights for CE using median-frequency balancing on NON-ZERO pixels.

    - f_c is computed within pixels where label != 0 and != ignore_index.
    - w_c = clip(median(f_k)/f_c, min_weight, max_weight) for c>=1 with count>0.
    - missing classes -> weight 1.0 (no effect).
    - class 0 -> bg_weight.
    """
    exts = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}
    names = [n for n in os.listdir(label_dir) if os.path.splitext(n)[1].lower() in exts]
    names.sort()
    if not names:
        raise RuntimeError(f"No label files found under: {label_dir}")

    counts = np.zeros((int(num_class),), dtype=np.int64)
    valid_total = 0
    for n in names:
        p = os.path.join(label_dir, n)
        lab = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if lab is None:
            continue
        if lab.ndim == 3:
            lab = lab[:, :, 0]
        lab = lab.astype(np.int64, copy=False)
        valid = (lab != int(ignore_index)) & (lab >= 0) & (lab < int(num_class))
        if not np.any(valid):
            continue
        vals, cnts = np.unique(lab[valid], return_counts=True)
        counts[vals] += cnts.astype(np.int64)
        valid_total += int(valid.sum())

    fg_counts = counts.copy()
    fg_counts[0] = 0
    fg_total = int(fg_counts.sum())

    weights = np.ones((int(num_class),), dtype=np.float32)
    weights[0] = float(bg_weight)
    if fg_total <= 0:
        return weights, counts, valid_total

    fg_freq = fg_counts.astype(np.float64) / float(fg_total)
    present = (fg_counts > 0)
    present[0] = False
    present_freq = fg_freq[present]
    if present_freq.size == 0:
        return weights, counts, valid_total

    median_freq = float(np.median(present_freq))
    for c in range(1, int(num_class)):
        if fg_counts[c] <= 0:
            weights[c] = 1.0
            continue
        wc = median_freq / float(fg_freq[c])
        wc = max(float(min_weight), min(float(max_weight), float(wc)))
        weights[c] = float(wc)

    return weights, counts, valid_total


def _multiclass_dice_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_class: int,
    ignore_index: int,
    include_background: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Multi-class soft Dice loss for segmentation.
    - logits: [B,C,H,W]
    - labels: [B,H,W] with ignore_index masked out
    """
    probs = torch.softmax(logits.float(), dim=1)  # [B,C,H,W]
    valid = (labels != int(ignore_index)).float()  # [B,H,W]

    safe_labels = labels.clone()
    safe_labels[safe_labels == int(ignore_index)] = 0
    one_hot = F.one_hot(safe_labels.long(), num_classes=int(num_class)).permute(0, 3, 1, 2).float()

    valid = valid.unsqueeze(1)  # [B,1,H,W]
    probs = probs * valid
    one_hot = one_hot * valid

    # per-class dice over (B,H,W)
    dims = (0, 2, 3)
    inter = (probs * one_hot).sum(dim=dims)
    denom = probs.sum(dim=dims) + one_hot.sum(dim=dims)
    dice = (2.0 * inter + eps) / (denom + eps)  # [C]

    start_c = 0 if include_background else 1
    if start_c >= int(num_class):
        return logits.sum() * 0.0

    dice_sel = dice[start_c:]
    return 1.0 - dice_sel.mean()


class MCDLightningModule(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(vars(args))
        self.model = Trainer(args).float()
        self.ce_loss = CrossEntropyLoss2d(ignore_index=args.ignore_index)
        self.val_evaluator = None
        self._ce_weight: torch.Tensor | None = None
        self.warned_invalid_train = False
        self.warned_invalid_val = False
        self._vis_cache_mcd = []
        self._vis_seen = 0
        self._vis_rng = None

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        pre_img = img[:, 0:3].float()
        post_img = img[:, 3:6].float()
        return self.model.update_mcd(pre_img, post_img)

    def on_fit_start(self):
        # Prepare CE class weights once per run.
        mode = str(getattr(self.args, "class_weight_mode", "none")).lower()
        manual = getattr(self.args, "class_weight", None)

        w_np = None
        if manual is not None and len(manual) > 0:
            if len(manual) != int(self.args.num_class):
                raise ValueError(f"--class_weight length must be num_class={self.args.num_class}, got {len(manual)}")
            w_np = np.array([float(x) for x in manual], dtype=np.float32)
        elif mode in ("median_freq_nonzero", "median_freq", "median"):
            rank = getattr(self.trainer, "global_rank", 0)
            num_class = int(self.args.num_class)
            ignore_index = int(self.args.ignore_index)
            bg_w = float(getattr(self.args, "class_weight_bg", 0.2))
            max_w = float(getattr(self.args, "class_weight_max", 10.0))
            min_w = float(getattr(self.args, "class_weight_min", 0.1))

            cache_ok = int(getattr(self.args, "class_weight_cache", 1)) == 1
            ckpt_dir = getattr(self.args, "checkpoint_dir", None) or getattr(self.args, "work_dirs", ".")
            cache_path = os.path.join(str(ckpt_dir), "class_weight_median_freq_nonzero.npy")

            if rank == 0:
                if cache_ok and os.path.isfile(cache_path):
                    w_np = np.load(cache_path).astype(np.float32)
                    print(f"[class_weight] loaded {cache_path}", flush=True)
                else:
                    label_dir = os.path.join(str(self.args.file_root), "train", str(self.args.label_dir))
                    w_np, counts, valid_total = _compute_class_weight_median_freq_nonzero(
                        label_dir=label_dir,
                        num_class=num_class,
                        ignore_index=ignore_index,
                        bg_weight=bg_w,
                        max_weight=max_w,
                        min_weight=min_w,
                    )
                    if cache_ok:
                        os.makedirs(str(ckpt_dir), exist_ok=True)
                        np.save(cache_path, w_np)
                    top = sorted([(i, float(w_np[i])) for i in range(num_class)], key=lambda x: -x[1])[:8]
                    print(
                        f"[class_weight] mode=median_freq_nonzero bg={bg_w} clip=[{min_w},{max_w}] "
                        f"valid_pixels={valid_total} saved={cache_ok}",
                        flush=True,
                    )
                    print(f"[class_weight] top={top}", flush=True)
            else:
                w_np = np.empty((int(self.args.num_class),), dtype=np.float32)

        if w_np is None:
            self._ce_weight = None
            return

        w_t = torch.tensor(w_np, device=self.device, dtype=torch.float32)
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(w_t, src=0)
        self._ce_weight = w_t

    def _vis_dir(self) -> str:
        ckpt_dir = getattr(self.args, "checkpoint_dir", None)
        if not ckpt_dir:
            ckpt_dir = os.path.join(getattr(self.args, "work_dirs", "./exp"), self.args.exp_name + "_TrainingFiles")
        return os.path.join(ckpt_dir, "vis")

    def _denorm_rgb_uint8(self, x_chw: torch.Tensor, mean3, std3) -> np.ndarray:
        """
        x_chw: float tensor [3,H,W], normalized by (x/255 - mean)/std (see transforms.py)
        return: uint8 BGR image [H,W,3] for cv2
        """
        mean = np.array(mean3, dtype=np.float32).reshape(3, 1, 1)
        std = np.array(std3, dtype=np.float32).reshape(3, 1, 1)
        x = x_chw.detach().float().cpu().numpy()
        x = (x * std + mean) * 255.0
        x = np.clip(x, 0.0, 255.0).astype(np.uint8)
        x = np.transpose(x, (1, 2, 0))  # HWC RGB
        x = x[:, :, ::-1]  # to BGR for cv2
        return x

    def _label_to_color_bgr(self, lab_hw: np.ndarray, num_class: int, ignore_index: int) -> np.ndarray:
        """
        lab_hw: int64 [H,W]
        return: uint8 BGR [H,W,3]
        """
        seed = int(getattr(self.args, "vis_color_seed", 123))
        rng = np.random.RandomState(seed)
        # class 0 -> black; others random but deterministic
        colors_rgb = np.zeros((num_class, 3), dtype=np.uint8)
        if num_class > 1:
            colors_rgb[1:] = rng.randint(0, 256, size=(num_class - 1, 3), dtype=np.uint8)
        ignore_rgb = np.array([192, 192, 192], dtype=np.uint8)  # light gray

        h, w = lab_hw.shape
        out_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        valid = (lab_hw != ignore_index) & (lab_hw >= 0) & (lab_hw < num_class)
        if valid.any():
            out_rgb[valid] = colors_rgb[lab_hw[valid]]
        out_rgb[lab_hw == ignore_index] = ignore_rgb
        out_bgr = out_rgb[:, :, ::-1]
        return out_bgr

    def _concat_with_gaps_bgr(self, imgs_bgr, gap: int, gap_color_bgr):
        """Concat BGR panels horizontally, with vertical gaps between panels."""
        assert len(imgs_bgr) >= 1
        if gap <= 0:
            return np.concatenate(imgs_bgr, axis=1)
        h = imgs_bgr[0].shape[0]
        gap_img = np.zeros((h, gap, 3), dtype=np.uint8)
        gap_img[:] = np.array(gap_color_bgr, dtype=np.uint8).reshape(1, 1, 3)
        parts = []
        for j, im in enumerate(imgs_bgr):
            if j > 0:
                parts.append(gap_img)
            parts.append(im)
        return np.concatenate(parts, axis=1)

    def _reservoir_update(self, cache: list, item, k: int) -> None:
        """
        Reservoir sampling update: keep k items uniformly from a stream.
        Uses self._vis_seen as the stream index (incremented outside).
        """
        if k <= 0:
            return
        if len(cache) < k:
            cache.append(item)
            return
        # pick random index in [0, self._vis_seen-1]
        j = int(self._vis_rng.randint(0, self._vis_seen))
        if j < k:
            cache[j] = item

    def training_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            imgs, labels, _names = batch
        else:
            imgs, labels = batch
        logits = self(imgs)
        labels = labels.long()
        labels, invalid = _map_invalid_labels(labels, self.args.num_class, self.args.ignore_index)
        if invalid.any() and (not self.warned_invalid_train) and getattr(self.trainer, "global_rank", 0) == 0:
            bad_vals = torch.unique(labels[invalid]).detach().cpu().tolist()
            print(
                f"[MCD] Found invalid label ids {bad_vals} (num_class={self.args.num_class}). "
                f"Mapping them to ignore_index={self.args.ignore_index}.",
                flush=True,
            )
            self.warned_invalid_train = True
        # CE + Dice for multi-class overlap optimization.
        ce_loss = F.cross_entropy(
            logits,
            labels,
            reduction="mean",
            ignore_index=int(self.args.ignore_index),
            weight=self._ce_weight,
        )
        dice_loss = _multiclass_dice_loss(
            logits=logits,
            labels=labels,
            num_class=int(self.args.num_class),
            ignore_index=int(self.args.ignore_index),
            include_background=bool(int(getattr(self.args, "dice_include_bg", 0))),
            eps=float(getattr(self.args, "dice_eps", 1e-6)),
        )
        dice_w = float(getattr(self.args, "dice_weight", 1.0))
        loss = ce_loss + dice_w * dice_loss
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log("train_ce_loss", ce_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("train_dice_loss", dice_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        return loss

    def on_validation_epoch_start(self):
        self.val_evaluator = Evaluator(num_class=self.args.num_class)
        self._vis_cache_mcd = []
        self._vis_seen = 0
        # epoch-different but reproducible sampling
        seed = int(getattr(self.args, "val_vis_seed", 2026))
        epoch = int(getattr(self.trainer, "current_epoch", 0))
        self._vis_rng = np.random.RandomState(seed + epoch * 10007)

    def validation_step(self, batch, batch_idx):
        names = None
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            imgs, labels, names = batch
        else:
            imgs, labels = batch
        logits = self(imgs)
        labels = labels.long()
        labels, invalid = _map_invalid_labels(labels, self.args.num_class, self.args.ignore_index)
        if invalid.any() and (not self.warned_invalid_val) and getattr(self.trainer, "global_rank", 0) == 0:
            bad_vals = torch.unique(labels[invalid]).detach().cpu().tolist()
            print(
                f"[MCD] Found invalid label ids {bad_vals} during val (num_class={self.args.num_class}). "
                f"Mapping them to ignore_index={self.args.ignore_index}.",
                flush=True,
            )
            self.warned_invalid_val = True

        ce_loss = F.cross_entropy(
            logits,
            labels,
            reduction="mean",
            ignore_index=int(self.args.ignore_index),
            weight=self._ce_weight,
        )
        dice_loss = _multiclass_dice_loss(
            logits=logits,
            labels=labels,
            num_class=int(self.args.num_class),
            ignore_index=int(self.args.ignore_index),
            include_background=bool(int(getattr(self.args, "dice_include_bg", 0))),
            eps=float(getattr(self.args, "dice_eps", 1e-6)),
        )
        dice_w = float(getattr(self.args, "dice_weight", 1.0))
        loss = ce_loss + dice_w * dice_loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("val_ce_loss", ce_loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)
        self.log("val_dice_loss", dice_loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)

        pred = torch.argmax(logits, dim=1).detach().cpu().numpy()
        gt = labels.detach().cpu().numpy()
        # Evaluator automatically ignores labels outside [0, num_class), so ignore_index=255 is excluded.
        self.val_evaluator.add_batch(gt, pred)

        # ---- optional visualization (rank0 only, random per-epoch) ----
        vis_num = int(getattr(self.args, "val_vis_num", 0))
        if vis_num > 0 and getattr(self.trainer, "global_rank", 0) == 0:
            b = imgs.size(0)
            ignore_index = int(self.args.ignore_index)
            num_class = int(self.args.num_class)
            nm = getattr(self.args, "normalize_mean", [0.5] * 6)
            ns = getattr(self.args, "normalize_std", [0.5] * 6)
            # allow passing 3 or 6 values; MCD default is 6 (pre+post)
            nm = list(nm)
            ns = list(ns)
            pre_mean3 = (nm[:3] if len(nm) >= 3 else [0.5, 0.5, 0.5])
            pre_std3 = (ns[:3] if len(ns) >= 3 else [0.5, 0.5, 0.5])
            post_mean3 = (nm[3:6] if len(nm) >= 6 else pre_mean3)
            post_std3 = (ns[3:6] if len(ns) >= 6 else pre_std3)
            gap = int(getattr(self.args, "val_vis_gap", 8))
            gap_color = getattr(self.args, "val_vis_gap_color_bgr", [32, 32, 32])

            for i in range(b):
                self._vis_seen += 1

                # Build panels only when they might be selected (small k; cheap enough here).
                pre = self._denorm_rgb_uint8(imgs[i, 0:3], mean3=pre_mean3, std3=pre_std3)
                post = self._denorm_rgb_uint8(imgs[i, 3:6], mean3=post_mean3, std3=post_std3)

                # MCD: pre | post | pred(class) | gt(class)
                pred_bgr = self._label_to_color_bgr(pred[i].astype(np.int64), num_class=num_class, ignore_index=ignore_index)
                gt_bgr = self._label_to_color_bgr(gt[i].astype(np.int64), num_class=num_class, ignore_index=ignore_index)
                row_mcd = self._concat_with_gaps_bgr([pre, post, pred_bgr, gt_bgr], gap=gap, gap_color_bgr=gap_color)

                # Name tokens (avoid depending on fixed batch order across epochs)
                raw_name = None
                if names is not None:
                    try:
                        raw_name = names[i]
                    except Exception:
                        raw_name = None
                if isinstance(raw_name, bytes):
                    raw_name = raw_name.decode("utf-8", errors="ignore")
                if raw_name is not None and not isinstance(raw_name, str):
                    raw_name = str(raw_name)
                token = (int(batch_idx), int(i), int(self._vis_seen), raw_name)
                self._reservoir_update(self._vis_cache_mcd, (row_mcd, token), k=vis_num)

    def on_validation_epoch_end(self):
        # NOTE: In DDP, each rank only sees a shard of the val dataloader.
        # We must aggregate the confusion matrix across ranks to compute
        # full-validation metrics (mIoU/pixacc/precision/recall/F1).
        rank = getattr(self.trainer, "global_rank", 0)
        miou = 0.0
        pixacc = 0.0
        prec_macro = 0.0
        rec_macro = 0.0
        f1_macro = 0.0

        cm_local = self.val_evaluator.confusion_matrix.astype(np.int64)
        cm_t = torch.from_numpy(cm_local).to(self.device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(cm_t, op=dist.ReduceOp.SUM)
        cm = cm_t.detach().cpu().numpy().astype(np.float64)

        # Per-class IoU from the aggregated confusion matrix.
        # IoU_c = TP_c / (TP_c + FP_c + FN_c)
        diag = np.diag(cm).astype(np.float64)
        denom = cm.sum(axis=1) + cm.sum(axis=0) - diag + 1e-7
        iou_per_class = diag / denom

        if rank == 0:
            # Print all class IoUs (class id -> IoU).
            # This helps verify which classes contribute to val_miou.
            pairs = [f"{i}:{iou_per_class[i]:.6f}" for i in range(len(iou_per_class))]
            print(f"[val] per_class_iou {', '.join(pairs)}", flush=True)

        # IMPORTANT:
        # - val_miou excludes class 0
        # - and now only averages over non-zero IoU values (to match your request).
        iou_1_to_30 = iou_per_class[1:]
        valid_mask = np.isfinite(iou_1_to_30) & (iou_1_to_30 != 0.0)
        used_count = int(valid_mask.sum())
        if used_count > 0:
            miou = float(iou_1_to_30[valid_mask].mean())
            if rank == 0:
                used_classes = np.where(valid_mask)[0] + 1  # shift back to [1..30]
                used_values = iou_1_to_30[valid_mask]
                pairs = [f"{int(c)}:{float(v):.10f}" for c, v in zip(used_classes, used_values)]
                print(
                    f"[val] val_miou_used_classes(1-30 nonzero)={used_count} "
                    f"details={', '.join(pairs)}",
                    flush=True,
                )
        else:
            miou = 0.0

        # overall pixel accuracy (from aggregated CM)
        total = float(cm.sum())
        pixacc = float(diag.sum() / max(1.0, total))

        # multi-class macro precision/recall（仅统计 1..num_class-1）
        num_class = cm.shape[0]
        if num_class > 1:
            tp = np.diag(cm)
            fn = cm.sum(axis=1) - tp
            fp = cm.sum(axis=0) - tp
            eps = 1e-7
            prec_per_class = tp / (tp + fp + eps)
            rec_per_class = tp / (tp + fn + eps)
            # 排除 0 类（通常为背景）
            if num_class > 1:
                prec_macro = float(np.nanmean(prec_per_class[1:]))
                rec_macro = float(np.nanmean(rec_per_class[1:]))
            else:
                prec_macro = float(np.nanmean(prec_per_class))
                rec_macro = float(np.nanmean(rec_per_class))

            if prec_macro + rec_macro > 0:
                f1_macro = float(2 * prec_macro * rec_macro / (prec_macro + rec_macro + 1e-7))

        if rank == 0:
            print(
                f"[val] epoch={self.trainer.current_epoch} "
                f"pixacc={pixacc:.6f} precision_macro={prec_macro:.6f} "
                f"recall_macro={rec_macro:.6f} f1_macro={f1_macro:.6f}",
                flush=True,
            )

        # Log on all ranks (values are identical because CM has been all-reduced).
        self.log("val_miou", float(miou), on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
        self.log("val_pixacc", float(pixacc), on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
        self.log("val_precision", float(prec_macro), on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
        self.log("val_recall", float(rec_macro), on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
        self.log("val_f1", float(f1_macro), on_epoch=True, prog_bar=False, logger=True, sync_dist=False)

        # Write visualization images at the end of val epoch (rank0 only).
        vis_num = int(getattr(self.args, "val_vis_num", 0))
        if rank == 0 and vis_num > 0 and len(self._vis_cache_mcd) > 0:
            vis_dir = self._vis_dir()
            os.makedirs(vis_dir, exist_ok=True)
            epoch = int(getattr(self.trainer, "current_epoch", 0))
            step = int(getattr(self.trainer, "global_step", 0))
            def _stem(raw_name: str | None) -> str:
                if not raw_name:
                    return "unknown"
                base = os.path.basename(raw_name)
                stem, _ = os.path.splitext(base)
                # keep filename-safe chars
                stem = "".join([c if (c.isalnum() or c in ("-", "_")) else "_" for c in stem])
                return stem[:80] if len(stem) > 80 else stem

            for row_bgr, token in self._vis_cache_mcd:
                batch_idx, in_batch_i, seen, raw_name = token
                out_name = f"epoch={epoch:03d}-step={step:06d}-seen{seen:06d}-{_stem(raw_name)}.png"
                cv2.imwrite(os.path.join(vis_dir, out_name), row_bgr)

            print(f"[val] saved vis={len(self._vis_cache_mcd)} to {vis_dir}", flush=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.args.lr,
            betas=(0.9, 0.99),
            eps=1e-8,
            weight_decay=1e-4,
        )

        # Match BCD: optional poly LR with warmup, stepped every optimizer step.
        if getattr(self.args, "lr_mode", "poly") != "poly":
            return optimizer

        total_steps = int(getattr(self.args, "max_steps", 0) or 0)
        if total_steps <= 0:
            return optimizer

        warmup_steps = int(getattr(self.args, "warmup_steps", 200))
        power = 0.9

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return 0.9 * (step + 1) / max(1, warmup_steps) + 0.1
            progress = step / max(1, total_steps)
            return float(max(0.0, (1.0 - progress) ** power))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


class MCDDataModule(L.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_ds = None
        self.val_ds = None
        self.train_transform = None
        self.val_transform = None

    def setup(self, stage=None):
        if self.train_ds is not None and self.val_ds is not None:
            return
        self.train_transform, self.val_transform = RSTransforms.MCDTransforms.get_transform_pipelines(self.args)
        self.train_ds = RSDataset.MCDDataset(
            file_root=self.args.file_root,
            split="train",
            transform=self.train_transform,
            im1_dir=self.args.im1_dir,
            im2_dir=self.args.im2_dir,
            label_dir=self.args.label_dir,
        )
        self.val_ds = RSDataset.MCDDataset(
            file_root=self.args.file_root,
            split="val",
            transform=self.val_transform,
            im1_dir=self.args.im1_dir,
            im2_dir=self.args.im2_dir,
            label_dir=self.args.label_dir,
        )
        # Return raw filename for val visualization naming (no effect on training).
        try:
            setattr(self.val_ds, "return_name", True)
        except Exception:
            pass

    def train_dataloader(self):
        sampler = None
        if dist.is_available() and dist.is_initialized():
            sampler = DistributedSampler(
                self.train_ds,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=True,
                drop_last=False,
            )
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.args.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.args.num_workers,
            pin_memory=True,
            persistent_workers=bool(int(self.args.num_workers) > 0),
            prefetch_factor=(2 if int(self.args.num_workers) > 0 else None),
            drop_last=False,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
            persistent_workers=bool(int(self.args.num_workers) > 0),
            prefetch_factor=(2 if int(self.args.num_workers) > 0 else None),
            drop_last=False,
        )


def lightning_train(args):
    if not hasattr(args, "work_dirs") or args.work_dirs is None:
        args.work_dirs = getattr(args, "save_dir", "./exp")
    if not hasattr(args, "exp_name") or args.exp_name is None:
        args.exp_name = "Default"
    if not hasattr(args, "resume_path"):
        args.resume_path = getattr(args, "resume", None)

    # Make sure Trainer goes into MCD branch
    if "MCD" not in args.dataset:
        args.dataset = f"{args.dataset}-MCD"

    if getattr(args, "max_steps", -1) != -1:
        args.max_epochs = None
        args.check_val_every_n_epoch = None

    ckpt_dir = os.path.join(args.work_dirs, args.exp_name + "_TrainingFiles")
    args.checkpoint_dir = ckpt_dir
    checkpoint_callback = ModelCheckpoint(
        monitor="val_miou",
        mode="max",
        save_top_k=10,
        dirpath=ckpt_dir,
        filename="best-model-{step:06d}-{val_miou:.4f}",
        save_last=True,
        verbose=True,
    )
    early_stop_callback = EarlyStopping(
        monitor="val_miou",
        patience=args.early_stop,
        mode="max",
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    csv_logger = CSVLogger(
        save_dir=ckpt_dir,
        name=args.exp_name,
        flush_logs_every_n_steps=getattr(args, "csv_flush_every", 1),
    )

    progress_print = None
    if getattr(args, "train_log_interval", 0) > 0:
        progress_print = ProgressPrintCallback(train_every_n_steps=args.train_log_interval)

    callbacks = [
        early_stop_callback,
        checkpoint_callback,
        lr_monitor,
        ModelSummary(max_depth=3),
    ]
    if progress_print is not None:
        callbacks.insert(0, progress_print)

    trainer = LTrainer(
        strategy=args.strategy,
        devices=args.devices,
        accelerator=args.accelerator,
        precision=args.precision,
        # DDP: Lightning 只在 global rank 0 渲染进度条，不会因多卡重复刷屏
        enable_progress_bar=False,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        logger=csv_logger,
        log_every_n_steps=2,
        gradient_clip_val=1.0,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        val_check_interval=args.val_check_interval,
        callbacks=callbacks,
        num_sanity_val_steps=0,
    )

    print(f"[config] dataset_tag={args.dataset}", flush=True)
    print(f"[config] dataset_root={args.file_root}", flush=True)
    print(f"[config] checkpoint_dir={ckpt_dir}", flush=True)

    module = MCDLightningModule(args)
    dm = MCDDataModule(args)
    trainer.fit(module, datamodule=dm, ckpt_path=args.resume_path if args.resume_path else None)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--dataset', default="MCD", help='Dataset name (will include "MCD" to select model branch).')
    parser.add_argument('--file_root', default='/mnt/wuxy/change_detection/datasets/your_dataset_root/image_512_1', help='path to the dataset directory')

    parser.add_argument('--im1_dir', default="A", help='subdir name for time-1 images (fallback: t1)')
    parser.add_argument('--im2_dir', default="B", help='subdir name for time-2 images (fallback: t2)')
    parser.add_argument('--label_dir', default="label", help='subdir name for labels')

    parser.add_argument('--in_height', type=int, default=512, help='Height of RGB input.')
    parser.add_argument('--in_width', type=int, default=512, help='Width of RGB input.')

    parser.add_argument('--num_perception_frame', type=int, default=1, help='Keep 1 for MCD.')
    parser.add_argument('--num_class', type=int, default=31, help='Number of change-type classes (include 0=unchanged).')
    # Many masks use 255 as "void/ignore". Keep 0 as valid class (unchanged).
    parser.add_argument('--ignore_index', type=int, default=255, help='Ignore index for labels (common: 255; do NOT use 0).')

    parser.add_argument('--max_steps', type=int, default=80000, help='Maximum number of iterations.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size.')
    # Windows uses 'spawn' and needs picklable dataset/transform objects.
    # Our transform pipeline uses closures, so safest default is 0.
    parser.add_argument('--num_workers', type=int, default=8, help='Number of worker threads (Windows: prefer 0).')

    parser.add_argument('--lr', type=float, default=2e-4, help='Initial learning rate.')
    parser.add_argument('--lr_mode', default='poly', help='Learning rate policy: "step" or "poly".')
    parser.add_argument('--step_loss', type=int, default=100, help='Decrease learning rate after how many epochs.')

    parser.add_argument('--pretrained', default='model/X3D_L.pyth', type=str, help='Path to pretrained weights.')
    parser.add_argument('--save_dir', default='./exp', help='Directory to save experiment results.')
    parser.add_argument('--resume', default=None, help='Resume training from a checkpoint.')
    parser.add_argument('--log_file', default='train_val_log.txt', help='Log file to store training/val stats.')
    parser.add_argument('--dice_weight', type=float, default=1.0, help='Total loss = CE + dice_weight * DiceLoss')
    parser.add_argument('--dice_include_bg', type=int, default=0, help='Include class 0 in Dice (0/1).')
    parser.add_argument('--dice_eps', type=float, default=1e-6, help='Numerical epsilon for Dice.')

    # ---- class weighting (for CE only) ----
    parser.add_argument(
        '--class_weight_mode',
        type=str,
        default='median_freq_nonzero',
        help='CE class weight mode: none | median_freq_nonzero (median-freq on non-zero pixels, clipped).',
    )
    parser.add_argument(
        '--class_weight',
        type=float,
        nargs='+',
        default=None,
        help='Manual CE class weights (length must equal num_class, include class 0). Overrides class_weight_mode.',
    )
    parser.add_argument('--class_weight_bg', type=float, default=0.2, help='Weight for class 0 when using auto mode.')
    parser.add_argument('--class_weight_max', type=float, default=10.0, help='Max clip value for auto CE weights.')
    parser.add_argument('--class_weight_min', type=float, default=0.1, help='Min clip value for auto CE weights (classes >=1).')
    parser.add_argument('--class_weight_cache', type=int, default=1, help='Cache auto class weights to checkpoint dir (0/1).')

    # ---- single-GPU by default; set --devices > 1 and a DDP strategy manually for multi-GPU ----
    parser.add_argument('--accelerator', type=str, default='gpu', help='Lightning accelerator')
    parser.add_argument('--devices', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--strategy', type=str, default='auto', help='Lightning distributed strategy')
    parser.add_argument('--precision', type=int, default=16, help='Use AMP when set to 16; otherwise 32')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID number')
    parser.add_argument('--seed', type=int, default=16, help='Random seed')

    # ---- Lightning trainer controls ----
    parser.add_argument('--exp_name', type=str, default='', help='Experiment name')
    parser.add_argument('--work_dirs', type=str, default=None, help='Working directory for saving results')
    parser.add_argument('--max_epochs', type=int, default=120, help='Number of epochs (-1 for unlimited)')
    parser.add_argument('--early_stop', type=int, default=80, help='Early stopping patience')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=20, help='Validation interval (epochs)')
    parser.add_argument('--val_check_interval', type=float, default=1.0, help='Validation interval. 1.0 validates every epoch.')
    parser.add_argument('--warmup_steps', type=int, default=200, help='Warmup steps for poly LR')
    parser.add_argument('--train_log_interval', type=int, default=50, help='Print train loss every N steps; 0 disables')
    parser.add_argument('--csv_flush_every', type=int, default=1, help='CSVLogger flush interval')

    # ---- validation visualization ----
    parser.add_argument('--val_vis_num', type=int, default=8, help='Save N visualization images during each val epoch (rank0 only). 0 disables.')
    parser.add_argument('--vis_color_seed', type=int, default=123, help='Random seed for class-color palette in visualization.')
    parser.add_argument('--val_vis_gap', type=int, default=8, help='Gap width (px) between pre/post/pred/gt panels in val visualization.')
    parser.add_argument(
        '--val_vis_gap_color_bgr',
        type=int,
        nargs=3,
        default=[32, 32, 32],
        help='Gap color in BGR (3 ints 0-255), e.g. --val_vis_gap_color_bgr 255 255 255 for white.',
    )
    parser.add_argument('--val_vis_seed', type=int, default=2026, help='Sampling seed for per-epoch random val visualizations.')

    args = parser.parse_args()
    # Seed (rank-aware inside Lightning).
    L.seed_everything(getattr(args, "seed", 16), workers=True)
    torch.backends.cudnn.benchmark = True
    cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    lightning_train(args)
