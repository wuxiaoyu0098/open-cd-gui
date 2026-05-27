# Copyright (c) Duowang Zhu.
# All rights reserved.

import os
import sys
import time
import subprocess
import numpy as np
import cv2
from os.path import join as osp
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import lightning as L
from lightning import Trainer as LTrainer
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
    ModelSummary,
    RichProgressBar,
)
from lightning.pytorch.loggers import CSVLogger

# Insert current path for local module imports
sys.path.insert(0, '.')

import data.dataset as RSDataset
import data.transforms as RSTransforms
from utils.metric_tool import ConfuseMatrixMeter, cm2score

from model.trainer import Trainer
from ckpt_to_pt_3D_BCD import export_bcd_ckpt_to_pt


class ProgressPrintCallback(Callback):
    """
    Print training/validation lines to stdout (rank 0 only) so nohup log files
    capture the training process without relying on TTY progress bars.
    """

    def __init__(self, train_every_n_steps: int = 50):
        self.train_every_n_steps = max(1, int(train_every_n_steps))

    def on_train_epoch_start(self, trainer, pl_module):
        if getattr(trainer, "global_rank", 0) != 0:
            return
        total = getattr(trainer, "num_training_batches", 0)
        print(f"[PROGRESS] epoch={trainer.current_epoch} step=0/{total} loss=nan", flush=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if getattr(trainer, "global_rank", 0) != 0:
            return
        total = int(getattr(trainer, "num_training_batches", 0) or 0)
        step = int(batch_idx) + 1
        if step % self.train_every_n_steps != 0 and step != total:
            return
        loss = None
        if isinstance(outputs, dict):
            loss = outputs.get("loss")
            if loss is None:
                for v in outputs.values():
                    if torch.is_tensor(v) and v.numel() == 1:
                        loss = v
                        break
        elif outputs is not None:
            loss = outputs
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
        val_iou = m.get("val_iou")
        if val_iou is None:
            return
        vi = val_iou.item() if torch.is_tensor(val_iou) else float(val_iou)
        val_loss = m.get("val_loss")
        extra = ""
        if val_loss is not None:
            vl = val_loss.item() if torch.is_tensor(val_loss) else float(val_loss)
            extra = f" val_loss={vl:.6f}"
        print(
            f"[val] epoch={trainer.current_epoch} val_iou={vi:.6f}{extra}",
            flush=True,
        )


class ExportPtAfterValidationCallback(Callback):
    def __init__(self, args):
        self.args = args
        self._converted = set()

    def _convert_one(self, ckpt_path: str):
        if not ckpt_path or ckpt_path in self._converted:
            return
        if not os.path.isfile(ckpt_path):
            return
        out_pt = os.path.splitext(ckpt_path)[0] + ".pt"
        try:
            export_bcd_ckpt_to_pt(
                weights=ckpt_path,
                out_pt=out_pt,
                dataset=self.args.dataset,
                in_height=self.args.in_height,
                in_width=self.args.in_width,
                pretrained=self.args.pretrained,
                cuda_no=getattr(self.args, "gpu_id", -1),
            )
            if os.path.isfile(out_pt):
                os.remove(ckpt_path)
                self._converted.add(ckpt_path)
                print(f"[CKPT] exported pt and removed ckpt: {out_pt}",
                      flush=True)
        except Exception as exc:
            print(f"[CKPT] pt export failed for {ckpt_path}: {exc}",
                  flush=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        if getattr(trainer, "global_rank", 0) != 0:
            return
        ckpt_cb = trainer.checkpoint_callback
        if ckpt_cb is None:
            return
        self._convert_one(getattr(ckpt_cb, "best_model_path", ""))
        self._convert_one(getattr(ckpt_cb, "last_model_path", ""))

from model.utils import (
    adjust_learning_rate,
    BCEDiceLoss,
    load_checkpoint,
    setup_logger
)


def create_data_loaders(args, train_transform, val_transform, distributed: bool, rank: int, world_size: int):
    """
    Creates data loaders for training, validation, and testing.
    
    Args:
        args: Command line arguments.
        train_transform: Transform pipeline for training data.
        val_transform: Transform pipeline for validation and testing data.
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, max_batches).
    """
    # Training data
    train_data = RSDataset.BCDDataset(
        file_root=args.file_root,
        split="train",
        transform=train_transform,
        pre_dir=args.pre_dir,
        post_dir=args.post_dir,
        label_dir=args.label_dir,
    )

    train_sampler = None
    if distributed:
        # Ensure each rank sees a different shard of data
        train_sampler = DistributedSampler(
            train_data,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    # Validation data
    val_data = RSDataset.BCDDataset(
        file_root=args.file_root,
        split="val",
        transform=val_transform,
        pre_dir=args.pre_dir,
        post_dir=args.post_dir,
        label_dir=args.label_dir,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Test data
    test_data = RSDataset.BCDDataset(
        file_root=args.file_root,
        split="test",
        transform=val_transform,
        pre_dir=args.pre_dir,
        post_dir=args.post_dir,
        label_dir=args.label_dir,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    max_batches = len(train_loader)
    print(f"For each epoch, we have {max_batches} batches.")
    
    return train_loader, val_loader, test_loader, max_batches


def _binary_confusion_from_preds(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute a 2x2 confusion matrix where rows=GT and cols=Pred.
    pred/target shapes can be [B, H, W] or [B, 1, H, W].
    """
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)
    pred = pred.to(torch.int64)
    target = target.to(torch.int64)
    num_classes = 2
    idx = num_classes * target.reshape(-1) + pred.reshape(-1)
    hist = torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return hist


class BCDLightningModule(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(vars(args))
        self.model = Trainer(args).float()
        self.val_confusion = None
        self.test_confusion = None
        self.test_loader = None
        self._val_vis_saved = 0
        self._val_seen = 0
        self._val_vis_pick = set()
        self._val_local_to_global = []
        self._val_vis_dir = None
        if int(getattr(self.args, "val_vis_count", 0)) > 0:
            work_dirs = getattr(self.args, "work_dirs", None) or getattr(self.args, "save_dir", "./checkpoints/")
            exp_name = getattr(self.args, "exp_name", "Default")
            self._val_vis_dir = os.path.join(work_dirs, f"{exp_name}_TrainingFiles", "vis")
            os.makedirs(self._val_vis_dir, exist_ok=True)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # img: [B, 6, H, W] => pre [B,3,H,W], post [B,3,H,W]
        pre_img = img[:, 0:3].float()
        post_img = img[:, 3:6].float()
        return self.model.update_bcd(pre_img, post_img)

    def _ensure_dataloaders(self):
        """
        Build only train/val dataloaders.
        NOTE: test split can contain broken symlinks; building it during sanity check
        will prevent training from starting. We'll build test_loader lazily in test_dataloader().
        """
        if hasattr(self, "train_loader") and hasattr(self, "val_loader"):
            return

        train_transform, val_transform = RSTransforms.BCDTransforms.get_transform_pipelines(self.args)

        train_dataset = RSDataset.BCDDataset(
            file_root=self.args.file_root,
            split="train",
            transform=train_transform,
            pre_dir=self.args.pre_dir,
            post_dir=self.args.post_dir,
            label_dir=self.args.label_dir,
        )
        val_dataset = RSDataset.BCDDataset(
            file_root=self.args.file_root,
            split="val",
            transform=val_transform,
            pre_dir=self.args.pre_dir,
            post_dir=self.args.post_dir,
            label_dir=self.args.label_dir,
        )

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def train_dataloader(self):
        self._ensure_dataloaders()
        return self.train_loader

    def val_dataloader(self):
        self._ensure_dataloaders()
        return self.val_loader

    def test_dataloader(self):
        if self.test_loader is not None:
            return self.test_loader

        # Lazily build test loader to avoid failing during sanity check.
        _, val_transform = RSTransforms.BCDTransforms.get_transform_pipelines(self.args)
        test_dataset = RSDataset.BCDDataset(
            file_root=self.args.file_root,
            split="test",
            transform=val_transform,
            pre_dir=self.args.pre_dir,
            post_dir=self.args.post_dir,
            label_dir=self.args.label_dir,
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        return self.test_loader

    def training_step(self, batch, batch_idx):
        img, target = batch
        output = self(img)
        loss = BCEDiceLoss(output, target.float())

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def on_validation_epoch_start(self):
        device = self.device
        self.val_confusion = torch.zeros((2, 2), device=device, dtype=torch.float64)
        self._val_vis_saved = 0
        self._val_seen = 0
        self._val_vis_pick = set()
        self._val_local_to_global = []
        vis_limit = int(getattr(self.args, "val_vis_count", 0))
        if vis_limit > 0:
            try:
                val_loader = self.val_dataloader()
                sampler = getattr(val_loader, "sampler", None)
                dataset_total = len(val_loader.dataset)
                if sampler is not None:
                    # Local validation iteration order -> global dataset index.
                    self._val_local_to_global = [int(x) for x in list(iter(sampler))]
                else:
                    self._val_local_to_global = list(range(dataset_total))
            except Exception:
                dataset_total = 0
                self._val_local_to_global = []
            if dataset_total > 0:
                k = min(vis_limit, dataset_total)
                picks = []
                # Sample from full validation set on rank0 then broadcast to all ranks.
                if not (dist.is_available() and dist.is_initialized()) or int(getattr(self, "global_rank", 0)) == 0:
                    seed = int(getattr(self.args, "seed", 16)) + int(self.current_epoch)
                    rng = np.random.default_rng(seed)
                    picks = [int(x) for x in rng.choice(dataset_total, size=k, replace=False).tolist()]
                if dist.is_available() and dist.is_initialized():
                    obj = [picks]
                    dist.broadcast_object_list(obj, src=0)
                    picks = [int(x) for x in obj[0]]
                self._val_vis_pick = set(picks)

    def _save_val_visuals(self, img: torch.Tensor, pred: torch.Tensor, target: torch.Tensor):
        vis_limit = int(getattr(self.args, "val_vis_count", 0))
        if vis_limit <= 0:
            return
        if self._val_vis_dir is None:
            return

        bsz = int(img.shape[0])
        for i in range(bsz):
            local_idx = self._val_seen + i
            if local_idx < len(self._val_local_to_global):
                global_idx = int(self._val_local_to_global[local_idx])
            else:
                global_idx = int(local_idx)
            if global_idx not in self._val_vis_pick:
                continue

            # img is normalized to roughly [-1, 1], split into t1/t2 RGB
            t1 = img[i, 0:3].detach().float().cpu().permute(1, 2, 0).numpy()
            t2 = img[i, 3:6].detach().float().cpu().permute(1, 2, 0).numpy()
            t1 = np.clip((t1 * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
            t2 = np.clip((t2 * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)

            p = pred[i].detach().float().cpu().squeeze().numpy()
            g = target[i].detach().float().cpu().squeeze().numpy()
            p_img = (np.clip(p, 0, 1) * 255.0).astype(np.uint8)
            g_img = (np.clip(g, 0, 1) * 255.0).astype(np.uint8)
            p_rgb = np.stack([p_img, p_img, p_img], axis=-1)
            g_rgb = np.stack([g_img, g_img, g_img], axis=-1)

            panel = np.hstack([t1, t2, p_rgb, g_rgb])  # 最后一张是 GT
            rank = int(getattr(self, "global_rank", 0))
            out_name = (
                f"epoch={int(self.current_epoch):03d}-"
                f"step={int(self.global_step):06d}-"
                f"rank{rank:02d}-gidx{int(global_idx):06d}-val_{int(global_idx):05d}.png"
            )
            out_path = os.path.join(self._val_vis_dir, out_name)
            cv2.imwrite(out_path, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
            self._val_vis_saved += 1
        self._val_seen += bsz

    def validation_step(self, batch, batch_idx):
        img, target = batch
        output = self(img)
        loss = BCEDiceLoss(output, target.float())
        pred = (output > 0.5).long()
        self._save_val_visuals(img, pred, target)
        hist = _binary_confusion_from_preds(pred, target)
        self.val_confusion += hist.to(torch.float64)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

    def on_validation_epoch_end(self):
        conf = self.val_confusion
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(conf, op=dist.ReduceOp.SUM)

        scores = cm2score(conf.detach().cpu().numpy())
        # Mirror keys used in CSDNet `main.py`
        self.log_dict(
            {
                "val_oa": float(scores["OA"]),
                "val_iou": float(scores["IoU"]),
                "val_f1": float(scores["F1"]),
                "val_recall": float(scores["recall"]),
                "val_precision": float(scores["precision"]),
            },
            prog_bar=True,
            on_epoch=True,
            logger=True,
        )

    def on_test_epoch_start(self):
        device = self.device
        self.test_confusion = torch.zeros((2, 2), device=device, dtype=torch.float64)

    def test_step(self, batch, batch_idx):
        img, target = batch
        output = self(img)
        pred = (output > 0.5).long()
        hist = _binary_confusion_from_preds(pred, target)
        self.test_confusion += hist.to(torch.float64)

    def on_test_epoch_end(self):
        conf = self.test_confusion
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(conf, op=dist.ReduceOp.SUM)

        scores = cm2score(conf.detach().cpu().numpy())
        self.log_dict(
            {
                "test_oa": float(scores["OA"]),
                "test_iou": float(scores["IoU"]),
                "test_f1": float(scores["F1"]),
                "test_recall": float(scores["recall"]),
                "test_precision": float(scores["precision"]),
            },
            prog_bar=True,
            on_epoch=True,
            logger=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.args.lr,
            betas=(0.9, 0.99),
            eps=1e-8,
            weight_decay=1e-4,
        )

        # Keep LR schedule simple but deterministic in multi-GPU.
        # - If lr_mode != 'poly', just use constant lr.
        if getattr(self.args, "lr_mode", "poly") != "poly":
            return optimizer

        total_steps = int(getattr(self.args, "max_steps", 0) or 0)
        if total_steps <= 0:
            return optimizer

        warmup_steps = int(getattr(self.args, "warmup_steps", 200))
        power = 0.9

        def lr_lambda(step: int) -> float:
            # step starts from 0
            if step < warmup_steps:
                return 0.9 * (step + 1) / max(1, warmup_steps) + 0.1
            progress = step / max(1, total_steps)
            factor = max(0.0, (1.0 - progress) ** power)
            return float(factor)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def lightning_train(args):
    """
    Use Lightning Trainer to run multi-GPU training (DDP) the same way as:
    `CSDNet-master/main.py`
    """
    # Map old args to Lightning-style dirs/names
    if not hasattr(args, "work_dirs") or args.work_dirs is None:
        args.work_dirs = getattr(args, "save_dir", "./exp")
    if not hasattr(args, "exp_name") or args.exp_name is None:
        args.exp_name = "Default"
    if not hasattr(args, "resume_path"):
        args.resume_path = getattr(args, "resume", None)

    if getattr(args, "max_steps", -1) != -1:
        args.max_epochs = None
        args.check_val_every_n_epoch = None

    ckpt_dir = os.path.join(args.work_dirs, args.exp_name + "_TrainingFiles")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_iou",
        mode="max",
        save_top_k=5,
        dirpath=ckpt_dir,
        filename="best-model-{step:06d}-{val_iou:.4f}",
        save_last=True,
        verbose=True,
    )
    early_stop_callback = EarlyStopping(
        monitor="val_iou",
        patience=args.early_stop,
        mode="max",
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Default flush_logs_every_n_steps=100 defers writing metrics.csv until ~step 99;
    # use 1 so metrics.csv exists soon after training starts (easier to monitor nohup runs).
    csv_logger = CSVLogger(
        save_dir=ckpt_dir,
        name=args.exp_name,
        flush_logs_every_n_steps=getattr(args, "csv_flush_every", 1),
    )

    progress_print = None
    if getattr(args, "train_log_interval", 0) > 0:
        progress_print = ProgressPrintCallback(train_every_n_steps=args.train_log_interval)

    callback_list = [
        early_stop_callback,
        checkpoint_callback,
        ExportPtAfterValidationCallback(args),
        lr_monitor,
        ModelSummary(max_depth=3),
    ]
    # Keep Lightning/Rich progress disabled for GUI and Windows GBK consoles.
    # ProgressPrintCallback prints plain ASCII lines that the OpenCD GUI parses.
    if progress_print is not None:
        callback_list.insert(0, progress_print)

    trainer = LTrainer(
        strategy=args.strategy,
        devices=args.devices,
        accelerator=args.accelerator,
        precision=args.precision,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        logger=csv_logger,
        log_every_n_steps=2,
        gradient_clip_val=1.0,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        val_check_interval=args.val_check_interval,
        callbacks=callback_list,
    )

    print(f"[config] dataset_tag={args.dataset}", flush=True)
    print(f"[config] dataset_root={args.file_root}", flush=True)
    print(f"[config] checkpoint_dir={ckpt_dir}", flush=True)

    module = BCDLightningModule(args)
    trainer.fit(module, ckpt_path=args.resume_path if args.resume_path else None)
    if not getattr(args, "skip_test", 1):
        trainer.test(module)


@torch.no_grad()
def val(args, val_loader, model, epoch, device):
    """
    Validates the model on the validation set.
    
    Args:
        args: Command line arguments.
        val_loader (DataLoader): DataLoader for validation data.
        model (nn.Module): The model to validate.
        epoch (int): Current epoch index.
        
    Returns:
        tuple: (average_loss, scores).
    """
    model_core = model.module if hasattr(model, 'module') else model
    model.eval()
    eval_meter = ConfuseMatrixMeter(n_class=2)
    epoch_loss = []
    total_batches = len(val_loader)
    
    print(f"Validation on {total_batches} batches")
    
    for iter_idx, batched_inputs in enumerate(val_loader):
        img, target = batched_inputs
        
        # Simplified data preparation
        pre_img = img[:, 0:3].to(device, non_blocking=True).float()
        post_img = img[:, 3:6].to(device, non_blocking=True).float()
        target = target.to(device, non_blocking=True).float()

        start_time = time.time()

        # Forward pass
        output = model_core.update_bcd(pre_img, post_img)
        loss = BCEDiceLoss(output, target)

        # Binarize predictions
        pred = torch.where(
            output > 0.5,
            torch.ones_like(output),
            torch.zeros_like(output)
        ).long()

        time_taken = time.time() - start_time
        epoch_loss.append(loss.data.item())

        # Update evaluation metrics
        f1 = eval_meter.update_cm(
            pr=pred.cpu().numpy(),
            gt=target.cpu().numpy()
        )
        
        if iter_idx % 5 == 0:
            print(
                f"\r[{iter_idx}/{total_batches}] "
                f"F1: {f1:.3f} loss: {loss.data.item():.3f} "
                f"time: {time_taken:.3f}",
                end=''
            )

    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)
    scores = eval_meter.get_scores()

    return average_epoch_loss_val, scores


def train(args, train_loader, model, optimizer, epoch, max_batches, 
          cur_iter=0, lr_factor=1., device=None, use_amp: bool = False, scaler=None):
    """
    Trains the model for one epoch.
    
    Args:
        args: Command line arguments.
        train_loader (DataLoader): DataLoader for training data.
        model (nn.Module): Model to train.
        optimizer: Optimizer instance.
        epoch (int): Current epoch index.
        max_batches (int): Number of batches per epoch.
        cur_iter (int): Current iteration count.
        lr_factor (float): Learning rate adjustment factor.
        
    Returns:
        tuple: (average_loss, scores, current_lr).
    """
    model_core = model.module if hasattr(model, 'module') else model
    model.train()
    eval_meter = ConfuseMatrixMeter(n_class=2)
    epoch_loss = []

    for iter_idx, batched_inputs in enumerate(train_loader):
        img, target = batched_inputs

        # Simplified data preparation
        pre_img = img[:, 0:3].to(device, non_blocking=True).float()
        post_img = img[:, 3:6].to(device, non_blocking=True).float()
        target = target.to(device, non_blocking=True).float()

        start_time = time.time()

        # Adjust learning rate
        lr = adjust_learning_rate(
            args,
            optimizer,
            epoch,
            iter_idx + cur_iter,
            max_batches,
            lr_factor=lr_factor
        )

        # Forward pass (optional AMP)
        if use_amp:
            with torch.cuda.amp.autocast(enabled=True):
                output = model_core.update_bcd(pre_img, post_img)
                loss = BCEDiceLoss(output, target)
        else:
            output = model_core.update_bcd(pre_img, post_img)
            loss = BCEDiceLoss(output, target)

        # Binarize predictions
        pred = torch.where(
            output > 0.5,
            torch.ones_like(output),
            torch.zeros_like(output)
        ).long()

        # Backpropagation
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Record loss
        epoch_loss.append(loss.data.item())
        time_taken = time.time() - start_time
        res_time = (max_batches * args.max_epochs - iter_idx - cur_iter) * time_taken / 3600

        # Update metrics
        with torch.no_grad():
            f1 = eval_meter.update_cm(
                pr=pred.cpu().numpy(),
                gt=target.cpu().numpy()
            )

        if (iter_idx + 1) % 5 == 0:
            print(
                f"[epoch {epoch}] [iter {iter_idx + 1}/{len(train_loader)} {res_time:.2f}h] "
                f"[lr {optimizer.param_groups[0]['lr']:.6f}] "
                f"[bn_loss {loss.data.item():.4f}] "
            )

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
    scores = eval_meter.get_scores()

    return average_epoch_loss_train, scores, lr


def trainValidate(args):
    """
    Main training and validation routine.
    
    Args:
        args: Command line arguments.
    """
    # DDP setup: torchrun provides LOCAL_RANK/RANK/WORLD_SIZE.
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    is_rank0 = rank == 0

    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=args.ddp_backend, init_method="env://")
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(args.gpu_id)
    
    # Enable CUDA optimizations and fix random seed
    torch.backends.cudnn.benchmark = True
    cudnn.benchmark = True
    seed = getattr(args, "seed", 16)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed + rank)

    # Initialize model
    model = Trainer(args).to(device).float()

    # Create experiment save directory
    save_path = osp(
        args.save_dir,
        f"{args.dataset}_iter_{args.max_steps}_lr_{args.lr}"
    )
    os.makedirs(save_path, exist_ok=True)

    # Data transformations
    train_transform, val_transform = RSTransforms.BCDTransforms.get_transform_pipelines(args)

    # Data loaders
    train_loader, val_loader, test_loader, max_batches = create_data_loaders(
        args, train_transform, val_transform, distributed=distributed, rank=rank, world_size=world_size
    )

    # Compute maximum epochs
    args.max_epochs = int(np.ceil(args.max_steps / max_batches))
    
    # Load checkpoint if needed
    start_epoch, cur_iter = load_checkpoint(args, model, save_path, max_batches)
    
    # Set up logger (rank0 only)
    logger = setup_logger(args, save_path) if is_rank0 else None

    # Wrap with DDP (after loading checkpoint, before optimizer)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        args.lr,
        (0.9, 0.99),
        eps=1e-08,
        weight_decay=1e-4
    )

    use_amp = (args.precision == 16 and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    # Track best IoU score (binary change: class 1)
    max_IoU_val = 0

    # Main training loop
    for epoch in range(start_epoch, args.max_epochs):
        torch.cuda.empty_cache()

        if distributed and hasattr(train_loader, "sampler") and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        # Train one epoch
        loss_train, score_tr, lr = train(
            args,
            train_loader,
            model,
            optimizer,
            epoch,
            max_batches,
            cur_iter,
            device=device,
            use_amp=use_amp,
            scaler=scaler,
        )
        cur_iter += len(train_loader)

        # Skip validation for the first epoch
        if epoch == 0:
            continue
        
        # Validation (rank0 only; avoids needing to all-reduce metrics)
        if is_rank0:
            torch.cuda.empty_cache()
            loss_val, score_val = val(args, test_loader, model, epoch, device)

            # Log validation results
            logger.write(
                "\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" % (
                    epoch,
                    score_val['Kappa'],
                    score_val['IoU'],
                    score_val['F1'],
                    score_val['recall'],
                    score_val['precision']
                )
            )
            logger.flush()

            # Save checkpoint
            model_core = model.module if hasattr(model, 'module') else model
            torch.save({
                'epoch': epoch + 1,
                'arch': str(model),
                'state_dict': model_core.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss_train': loss_train,
                'loss_val': loss_val,
                'F_train': score_tr['F1'],
                'F_val': score_val['F1'],
                'lr': lr
            }, osp(save_path, 'checkpoint.pth.tar'))

            # Save the best model
            model_file_name = osp(save_path, 'best_model.pth')
            if epoch % 1 == 0 and max_IoU_val <= score_val['IoU']:
                max_IoU_val = score_val['IoU']
                stamped_name = osp(save_path, f"best_model_class1_iou_{max_IoU_val:.4f}.pth")
                torch.save(model_core.state_dict(), model_file_name)
                torch.save(model_core.state_dict(), stamped_name)

            # Print summary
            print(f"\nEpoch {epoch}: Details")
            print(
                f"\nEpoch No. {epoch}:\tTrain Loss = {loss_train:.4f}\t"
                f"Val Loss = {loss_val:.4f}\tF1(tr) = {score_tr['F1']:.4f}\t"
                f"F1(val) = {score_val['F1']:.4f}"
            )
        else:
            # Keep variables defined (not used for non-rank0 saving/logging)
            loss_val, score_val = 0.0, {'Kappa': 0, 'IoU': 0, 'F1': 0, 'recall': 0, 'precision': 0}
    
    # Test with the best model
    if is_rank0:
        model_file_name = osp(save_path, 'best_model.pth')
        model_core = model.module if hasattr(model, 'module') else model
        state_dict = torch.load(model_file_name, map_location=device)
        model_core.load_state_dict(state_dict)

        loss_test, score_test = val(args, test_loader, model, 0, device)
        print(
            f"\nTest:\t Kappa (te) = {score_test['Kappa']:.4f}\t "
            f"IoU (te) = {score_test['IoU']:.4f}\t"
            f"F1 (te) = {score_test['F1']:.4f}\t "
            f"R (te) = {score_test['recall']:.4f}\t"
            f"P (te) = {score_test['precision']:.4f}"
        )

        logger.write(
            "\n%s\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" % (
                'Test',
                score_test['Kappa'],
                score_test['IoU'],
                score_test['F1'],
                score_test['recall'],
                score_test['precision']
            )
        )
        logger.flush()
        logger.close()

    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--dataset',
        default='LEVIR-CD',
        help='Tag for logging; must contain substring "CD" for the binary branch (or any name - '
             'we auto-append "-CD" if missing).',
    )
    parser.add_argument(
        '--file_root',
        default='/mnt/wuxy/change_detection/datasets/your_dataset_root/image_512_1',
        help='Dataset root containing train/val/test subfolders',
    )
    parser.add_argument(
        '--pre_dir',
        default='A',
        help='Pre-change image subfolder per split (LEVIR: t1; use A if your data uses A/B)',
    )
    parser.add_argument(
        '--post_dir',
        default='B',
        help='Post-change image subfolder per split',
    )
    parser.add_argument(
        '--label_dir',
        default='label',
        help='Binary mask subfolder per split',
    )
    parser.add_argument(
        '--in_height',
        type=int,
        default=512,
        help='Height of RGB image'
    )
    parser.add_argument(
        '--in_width',
        type=int,
        default=512,
        help='Width of RGB image'
    )
    parser.add_argument(
        '--num_perception_frame',
        type=int,
        default=1,
        help='Number of perception frames'
    )
    parser.add_argument(
        '--num_class',
        type=int,
        default=1,
        help='Number of classes'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=10,
        help='Max number of iterations'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=0,
        help='Number of parallel threads'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=2e-4,
        help='Initial learning rate'
    )
    parser.add_argument(
        '--lr_mode',
        default='poly',
        help='Learning rate policy: step or poly'
    )
    parser.add_argument(
        '--step_loss',
        type=int,
        default=100,
        help='Decrease learning rate after how many epochs'
    )
    parser.add_argument(
        '--pretrained',
        default='model/X3D_L.pyth',
        type=str,
        help='Path to pretrained weight'
    )
    parser.add_argument(
        '--save_dir',
        default='./checkpoints/',
        help='Directory to save the experiment results'
    )
    parser.add_argument(
        '--resume',
        default=None,
        help='Checkpoint to resume training'
    )
    parser.add_argument(
        '--log_file',
        default='train_val_log.txt',
        help='File that stores the training and validation logs'
    )
    parser.add_argument(
        '--gpu_id',
        default=0,
        type=int,
        help='GPU ID number'
    )

    # ---- DDP / multi-GPU options ----
    parser.add_argument('--accelerator', type=str, default='gpu', help='Lightning accelerator')
    parser.add_argument('--devices', type=str, default=1, help='Number of GPUs')
    parser.add_argument(
        '--strategy',
        type=str,
        default='auto',
        help='Strategy for distributed training',
    )
    parser.add_argument('--precision', type=int, default=16, help='Use AMP when set to 16; otherwise 32')
    parser.add_argument('--seed', type=int, default=16, help='Random seed')

    # ---- Lightning trainer controls (mirror CSDNet main.py style) ----
    parser.add_argument('--exp_name', type=str, default='Default', help='Experiment name')
    parser.add_argument('--work_dirs', type=str, default=None, help='Working directory for saving results')
    parser.add_argument('--max_epochs', type=int, default=120, help='Number of epochs (-1 for unlimited)')
    parser.add_argument('--early_stop', type=int, default=80, help='Early stopping patience')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=20, help='Validation interval (epochs)')
    parser.add_argument('--val_check_interval', type=float, default=1.0, help='Validation interval. 1.0 validates every epoch.')
    parser.add_argument('--warmup_steps', type=int, default=200, help='Warmup steps for poly LR')
    parser.add_argument('--skip_test', type=int, default=1, help='Skip final trainer.test() (default 1 to avoid broken test symlinks)')
    parser.add_argument(
        '--train_log_interval',
        type=int,
        default=50,
        help='Print train loss every N global steps to stdout (for nohup logs); 0 disables',
    )
    parser.add_argument(
        '--csv_flush_every',
        type=int,
        default=1,
        help='CSVLogger: flush metrics.csv to disk every N logged steps (default 1; Lightning default 100 delays file creation)',
    )
    parser.add_argument(
        '--val_vis_count',
        type=int,
        default=16,
        help='Number of validation samples to visualize each val epoch (0 disables).',
    )

    args = parser.parse_args()

    args.dataset = args.dataset.strip()
    # Trainer.__init__ uses ('CD' in dataset and 'MCD' not in dataset) for binary (sigmoid) decoder.
    if 'MCD' in args.dataset:
        raise ValueError(
            f'--dataset={args.dataset!r} selects the multi-class branch; use scripts/train_MCD.py instead.'
        )
    if 'CD' not in args.dataset:
        args.dataset = f'{args.dataset}-CD'

    # Dataset default folder names:
    # - Some datasets use A/B/label
    # - Some datasets use t1/t2/label
    # For LEVIR (and similar), pick the one that actually exists in `file_root/train`.
    if 'LEVIR' in args.dataset:
        # Only auto-adjust when user didn't override (defaults are A/B)
        pre_dir = getattr(args, "pre_dir", None)
        post_dir = getattr(args, "post_dir", None)
        if pre_dir in (None, "A", "a") and post_dir in (None, "B", "b"):
            train_root = os.path.join(args.file_root, "train")
            if os.path.exists(os.path.join(train_root, "t1")) and os.path.exists(os.path.join(train_root, "t2")):
                args.pre_dir = "t1"
                args.post_dir = "t2"
            else:
                args.pre_dir = "A"
                args.post_dir = "B"
    lightning_train(args)
