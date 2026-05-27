from PIL import Image  # Import before torch to avoid Windows DLL load issues.
import torchvision.transforms.functional as TF  # noqa: F401

import torch
import torch.nn.functional as F
from option import Options
from data.cd_dataset import DataLoader
from model.create_ChangeDINO import create_model
import torch.optim as optim
from tqdm import tqdm
import math
from util.metric_tool import ConfuseMatrixMeter
import os
import json
import numpy as np
import random
from datetime import datetime
import time
import sys
from util.util import make_numpy_grid, de_norm
import matplotlib.pyplot as plt

import lightning.pytorch as pl
from lightning import Trainer
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    RichProgressBar,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import CSVLogger
import torch.distributed as dist


def setup_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False 
    torch.backends.cudnn.benchmark = True  
    torch.backends.cudnn.enabled = True  


class ChangeDINOLightningModule(pl.LightningModule):
    """PyTorch Lightning Module for ChangeDINO training with DDP support"""
    
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.save_hyperparameters(ignore=['opt'])  # Save hyperparameters except opt object
        
        # Create model
        self.model_wrapper = create_model(opt)
        self.model = self.model_wrapper.model  # The actual model
        self.focal = self.model_wrapper.focal
        self.dice = self.model_wrapper.dice
        
        # Training parameters
        self.alpha = 0.5
        self.running_metric = ConfuseMatrixMeter(n_class=2)
        
        # Resume info
        if self.model_wrapper.resume_info is not None:
            self.start_epoch = self.model_wrapper.resume_info["epoch"] + 1
            self.previous_best = self.model_wrapper.resume_info["previous_best"]
            print(f"Resume training from epoch {self.start_epoch}, previous best IoU: {self.previous_best*100:.3f}%")
        else:
            self.start_epoch = 1
            self.previous_best = 0.0
        
        # Logging paths
        self.log_path = os.path.join(self.model_wrapper.save_dir, "record.txt")
        self.vis_path = os.path.join(self.model_wrapper.save_dir, opt.vis_path)
        os.makedirs(self.vis_path, exist_ok=True)
        
        # Initialize log file
        if not os.path.exists(self.log_path):
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write("# Record of training/validation metrics\n")
                f.write(
                    "# name: %s | backbone: %s\n"
                    % (opt.name, getattr(opt, "backbone", "NA"))
                )
                f.write("# time,epoch,train_loss,train_focal,train_dice,lr,")
                f.write("val_metrics(json)\n")
        
        # Track metrics for logging
        self.train_losses = []
        self.train_focal_losses = []
        self.train_dice_losses = []
        self.current_epoch_metrics = {}
        
        # For rescheduling optimizer at epoch 0.9
        self.rescheduled = False
    
    def forward(self, img1, img2, label):
        """Forward pass"""
        return self.model_wrapper(img1, img2, label)
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        img1 = batch["img1"]
        img2 = batch["img2"]
        label = batch["cd_label"]
        
        # Forward pass
        pred, focal, dice = self.model_wrapper(img1, img2, label)
        
        # Compute loss
        loss = focal * self.alpha + dice
        
        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_focal", focal, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train_dice", dice, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        
        # Store for epoch-end logging
        self.train_losses.append(loss.item())
        self.train_focal_losses.append(focal.item())
        self.train_dice_losses.append(dice.item())
        
        # Plot result on last batch of epoch
        if batch_idx == self.trainer.num_training_batches - 1:
            self._plot_cd_result(img1, img2, pred, label, self.current_epoch, "train")
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        try:
            img1 = batch["img1"]
            img2 = batch["img2"]
            label = batch["cd_label"]
            
            # Ensure data has correct shape
            # DataLoader should automatically batch data, but handle edge cases
            if len(img1.shape) == 3:  # (C, H, W) - missing batch dimension
                img1 = img1.unsqueeze(0)
                img2 = img2.unsqueeze(0)
                if len(label.shape) == 2:  # (H, W)
                    label = label.unsqueeze(0)
            
            # Ensure label has correct shape (B, H, W)
            if len(label.shape) == 4:  # (B, 1, H, W)
                label = label.squeeze(1)
            elif len(label.shape) == 2:  # (H, W) - missing batch dimension
                label = label.unsqueeze(0)
            
            # Move to device (Lightning handles this automatically, but ensure consistency)
            device = next(self.model.parameters()).device
            img1 = img1.to(device)
            img2 = img2.to(device)
            
            # Inference - model expects (B, C, H, W)
            val_pred = self.model_wrapper.inference(img1, img2)
            val_target = label.detach()
            
            # Resize if needed
            target_size = val_target.shape[-2:]
            if val_pred.shape[-2:] != target_size:
                val_pred = F.interpolate(
                    val_pred, size=target_size, mode="bilinear", align_corners=False
                )
            
            val_pred = torch.argmax(val_pred.detach(), dim=1)
            
            # Process label
            val_target_np = val_target.cpu().numpy()
            if val_target_np.max() > 1:
                val_target_np = (val_target_np > 0).astype(np.uint8)
            
            # Update confusion matrix
            _ = self.running_metric.update_cm(
                pr=val_pred.cpu().numpy(), gt=val_target_np
            )
            
            # Plot result on last batch of epoch
            if hasattr(self, 'trainer') and self.trainer is not None:
                num_val_batches = self.trainer.num_val_batches
                if isinstance(num_val_batches, list) and len(num_val_batches) > 0:
                    if batch_idx == num_val_batches[0] - 1:
                        self._plot_cd_result(img1, img2, val_pred, label, self.current_epoch, "val")
        except Exception as e:
            print(f"Error in validation_step at batch {batch_idx}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch"""
        # Get scores from confusion matrix
        val_scores = self.running_metric.get_scores()
        
        # Log validation metrics
        for k, v in val_scores.items():
            self.log(f"val_{k}", v, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Store for file logging
        self.current_epoch_metrics = val_scores
        
        # Print message (include epoch)
        # Note: Lightning's current_epoch is 0-based; also show 1-based for readability.
        epoch0 = getattr(self.trainer, "current_epoch", 0) if getattr(self, "trainer", None) is not None else 0
        epoch1 = epoch0 + 1
        message = "(phase: %s) epoch=%d(1based=%d) " % (self.opt.phase, epoch0, epoch1)
        for k, v in val_scores.items():
            message += "%s: %.3f " % (k, v * 100)
        # Gate printing to rank0
        if getattr(self, "global_rank", 0) == 0:
            print(message, flush=True)
        
        # Clear confusion matrix for next epoch
        self.running_metric.clear()
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch"""
        # Get average metrics
        n = len(self.train_losses)
        if n > 0:
            avg_loss = sum(self.train_losses) / n
            avg_focal = sum(self.train_focal_losses) / n
            avg_dice = sum(self.train_dice_losses) / n
            try:
                current_lr = self.optimizers().param_groups[0]["lr"]
            except:
                current_lr = self.model_wrapper.optimizer.param_groups[0]["lr"]
            
            # Store for file logging
            self.current_epoch_metrics.update({
                "train_loss": avg_loss,
                "train_focal": avg_focal,
                "train_dice": avg_dice,
                "lr": current_lr
            })
            
            # Clear for next epoch
            self.train_losses.clear()
            self.train_focal_losses.clear()
            self.train_dice_losses.clear()
        
        # CSD-style stability: do NOT replace optimizer/scheduler objects mid-training in DDP.
        # The previous implementation mutated `trainer.optimizers` / `trainer.lr_schedulers`, which can hang in DDP.
        # If you need LR decay, implement it via configure_optimizers() scheduler or manually adjust param_group['lr'].
        # We keep the flag for backward compatibility but intentionally disable hot-swapping.
        if not self.rescheduled and self.trainer is not None and self.trainer.current_epoch >= int(self.opt.num_epochs * 0.9):
            if getattr(self, "global_rank", 0) == 0:
                print("[WARN] rescheduler disabled for DDP stability (use scheduler in configure_optimizers instead).", flush=True)
            self.rescheduled = True
    
    def on_train_epoch_start(self):
        """Called at the start of training epoch"""
        # Lightning's `current_epoch` is managed by Trainer and cannot be assigned to.
        # Keep a separate field if we need an offset epoch index for logging/checkpoint naming.
        if hasattr(self, "trainer") and self.trainer is not None:
            self.epoch_for_log = self.trainer.current_epoch + self.start_epoch
        else:
            self.epoch_for_log = self.start_epoch
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = self.model_wrapper.optimizer
        scheduler = {
            'scheduler': self.model_wrapper.schedular,
            'interval': 'epoch',
            'frequency': 1,
        }
        return [optimizer], [scheduler]
    
    def _rescheduler(self):
        """Reschedule optimizer at 90% of training"""
        # Deprecated: kept for historical reference only.
        # Do not call this in Lightning/DDP.
        return
    
    def _plot_cd_result(self, x1, x2, pred, target, epoch, stage):
        """Plot change detection results"""
        # Lightning automatically handles this, but we check anyway
        if hasattr(self, 'trainer') and self.trainer is not None:
            if self.trainer.global_rank != 0:
                return
        elif hasattr(self, 'global_rank') and self.global_rank != 0:
            return
            
        if len(pred.shape) == 4:
            pred = torch.argmax(pred, dim=1)
        vis_input = make_numpy_grid(de_norm(x1[0:8]))
        vis_input2 = make_numpy_grid(de_norm(x2[0:8]))
        vis_pred = make_numpy_grid(pred[0:8].unsqueeze(1).repeat(1, 3, 1, 1))
        vis_gt = make_numpy_grid(target[0:8].unsqueeze(1).repeat(1, 3, 1, 1))
        vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
        vis = np.clip(vis, a_min=0.0, a_max=1.0)
        file_name = os.path.join(self.vis_path, f"{stage}_{epoch}.jpg")
        plt.imsave(file_name, vis)
    
    def _append_log_line(self, epoch, train_stats, val_scores):
        """Append log line to record.txt"""
        # Lightning automatically handles this, but we check anyway
        if hasattr(self, 'trainer') and self.trainer is not None:
            if self.trainer.global_rank != 0:
                return
        elif hasattr(self, 'global_rank') and self.global_rank != 0:
            return
            
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = (
            f"{ts},{epoch},"
            f"{train_stats.get('loss', float('nan')):.6f},"
            f"{train_stats.get('focal', float('nan')):.6f},"
            f"{train_stats.get('dice', float('nan')):.6f},"
            f"{train_stats.get('lr', float('nan')):.8f},"
            + json.dumps(val_scores, ensure_ascii=False)
            + "\n"
        )
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line)
    
    def on_train_end(self):
        """Called at the end of training.

        NOTE:
        - We only want *one* .pth per epoch (saved after validation).
        - Saving again here easily causes duplicate epoch-0 .pth files.
        """
        return


class ChangeDINODataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for ChangeDINO"""
    
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.train_dataset = None
        self.val_dataset = None
    
    def setup(self, stage=None):
        """Setup datasets"""
        # Training dataset
        train_opt = type(self.opt)(**vars(self.opt))  # Copy opt
        train_opt.phase = "train"
        train_loader = DataLoader(train_opt)
        self.train_dataset = train_loader.dataset
        
        # Validation dataset
        val_opt = type(self.opt)(**vars(self.opt))  # Copy opt
        val_opt.phase = "val"
        if hasattr(self.opt, "val_batch_size") and self.opt.val_batch_size is not None:
            val_opt.batch_size = self.opt.val_batch_size
        val_loader = DataLoader(val_opt)
        self.val_dataset = val_loader.dataset

        # Print dataset paths & sizes once (rank0 only), so you can confirm the correct split is used.
        try:
            rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        except Exception:
            rank = 0

        if rank == 0:
            td = self.train_dataset
            vd = self.val_dataset

            train_a = getattr(td, "dir1", None)
            train_b = getattr(td, "dir2", None)
            train_label = getattr(td, "dir_label", None)
            train_n = int(getattr(td, "dataset_size", len(td)))

            val_a = getattr(vd, "dir1", None)
            val_b = getattr(vd, "dir2", None)
            val_label = getattr(vd, "dir_label", None)
            val_n = int(getattr(vd, "dataset_size", len(vd)))

            print(
                "[DATA] TRAIN="
                f"{self.opt.dataroot}/{self.opt.dataset} "
                f"A={train_a} B={train_b} label={train_label} count={train_n}",
                flush=True,
            )
            print(
                "[DATA] VAL="
                f"{self.opt.dataroot}/{self.opt.dataset} "
                f"A={val_a} B={val_b} label={val_label} count={val_n}",
                flush=True,
            )
    
    def train_dataloader(self):
        """Return training dataloader with automatic DDP support"""
        from torch.utils.data import DataLoader as TorchDataLoader
        return TorchDataLoader(
            self.train_dataset,
            batch_size=self.opt.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=int(self.opt.num_workers),
        )
    
    def val_dataloader(self):
        """Return validation dataloader"""
        from torch.utils.data import DataLoader as TorchDataLoader
        # Get val_batch_size, handling None case
        val_batch_size = getattr(self.opt, "val_batch_size", None)
        if val_batch_size is None or val_batch_size <= 0:
            val_batch_size = self.opt.batch_size
        # Ensure batch_size is at least 1
        val_batch_size = max(1, int(val_batch_size))
        return TorchDataLoader(
            self.val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=int(self.opt.num_workers),
            collate_fn=None,  # Use default collate_fn to ensure proper batching
        )


if __name__ == "__main__":
    opt = Options().parse()

    # Lightning 的纯 "ddp" 对“有些参数在某些 step 不参与 loss”的模型会直接报错：
    # RuntimeError: parameters were not used in producing the loss...
    # CSDNet 里一般不会遇到这个，但 ChangeDINO 这套代码会，因此在不改启动方式的前提下做一次兼容兜底。
    if str(opt.strategy).strip() == "ddp":
        # 只在 rank0 提示即可
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        if rank == 0:
            print(
                "[WARN] strategy='ddp' triggered unused-parameters error in this model. "
                "Switching to 'ddp_find_unused_parameters_true'. "
                "You can also pass: --strategy ddp_find_unused_parameters_true",
                flush=True,
            )
        opt.strategy = "ddp_find_unused_parameters_true"
    
    # Parse gpu_ids if it is a string
    if isinstance(opt.gpu_ids, str):
        opt.gpu_ids = [int(x) for x in opt.gpu_ids.split(",") if x.strip() != ""]
    
    # Setup seed
    setup_seed(seed=1)
    
    # Create data module first (needed for model initialization in some cases)
    data_module = ChangeDINODataModule(opt)
    data_module.setup()  # Setup datasets
    
    # Create Lightning module
    model = ChangeDINOLightningModule(opt)
    
    # Setup callbacks
    callbacks = []
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    # Per-epoch wall-time (rank0 only)
    class EpochTimeCallback(pl.Callback):
        def __init__(self):
            super().__init__()
            self._t0_train = None
            self._t0_epoch = None

        def on_train_epoch_start(self, trainer, pl_module):
            if trainer.global_rank != 0:
                return
            t = time.perf_counter()
            self._t0_train = t
            self._t0_epoch = t

        def on_train_epoch_end(self, trainer, pl_module):
            if trainer.global_rank != 0:
                return
            if self._t0_train is None:
                return
            dt = time.perf_counter() - self._t0_train
            print(f"[TIME] epoch={trainer.current_epoch} train={dt:.2f}s", flush=True)

        def on_validation_epoch_end(self, trainer, pl_module):
            if trainer.global_rank != 0:
                return
            if self._t0_epoch is None:
                return
            dt = time.perf_counter() - self._t0_epoch
            print(f"[TIME] epoch={trainer.current_epoch} total(train+val)={dt:.2f}s", flush=True)

    callbacks.append(EpochTimeCallback())
    
    # 在重定向到 log 文件（nohup > xxx.log）时，tqdm/rich 的动态进度条通常不会“逐行写入”；
    # 这里加一个纯文本进度打印，确保你在 log 里能持续看到训练在跑。
    class PrintProgressCallback(pl.Callback):
        def __init__(self, every_n_steps: int = 10):
            super().__init__()
            self.every_n_steps = max(1, int(every_n_steps))

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if trainer.global_rank != 0:
                return
            if (batch_idx + 1) % self.every_n_steps != 0 and (batch_idx + 1) != trainer.num_training_batches:
                return

            # 尽量从 lightning metrics 里拿到 loss（可能叫 train_loss / train_loss_step）
            metrics = trainer.callback_metrics
            loss_val = None
            for k in ("train_loss", "train_loss_step"):
                if k in metrics:
                    try:
                        loss_val = float(metrics[k])
                    except Exception:
                        pass
                    break

            msg = (
                f"[PROGRESS] epoch={trainer.current_epoch} "
                f"step={batch_idx + 1}/{trainer.num_training_batches}"
            )
            if loss_val is not None:
                msg += f" loss={loss_val:.4f}"
            print(msg, flush=True)

        def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
            if trainer.global_rank != 0:
                return
            num_val = trainer.num_val_batches[dataloader_idx] if isinstance(trainer.num_val_batches, list) else trainer.num_val_batches
            if num_val is None:
                return
            if (batch_idx + 1) % self.every_n_steps != 0 and (batch_idx + 1) != num_val:
                return
            print(
                f"[PROGRESS][val] epoch={trainer.current_epoch} step={batch_idx + 1}/{num_val}",
                flush=True,
            )

    every = int(os.environ.get("PRINT_EVERY_N_STEPS", "10"))
    callbacks.append(PrintProgressCallback(every_n_steps=every))

    # Save ONLY Lightning checkpoints (.ckpt). Disable any custom .pt exporting.
    ckpt_dir = os.path.join(opt.checkpoint_dir, opt.name, "TrainingFiles")
    use_lightning_ckpt = True
    checkpoint_callback = ModelCheckpoint(
        monitor="val_iou_1",
        mode="max",
        save_top_k=5,
        save_last=True,
        dirpath=ckpt_dir,
        filename="best-model-{epoch:03d}-{step:06d}-{val_iou_1:.4f}",
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

    class ExportPtCheckpointCallback(pl.Callback):
        """Export checkpoints with configurable content.

        save_mode:
        - "state_dict": torch.save({"network": state_dict, ...})  (default)
        - "full_model": torch.save(model)  (pickle; requires Python source to load)
        - "torchscript": torch.jit.save(scripted_model)  (deployable; torch.jit.load)
        """
        def __init__(self, out_dir: str, save_mode: str = "state_dict"):
            super().__init__()
            self.out_dir = out_dir
            self.save_mode = save_mode
            self.best_score = None
            os.makedirs(self.out_dir, exist_ok=True)

        @staticmethod
        def _to_float(v):
            if v is None:
                return None
            if isinstance(v, torch.Tensor):
                return float(v.detach().cpu().item())
            return float(v)

        @staticmethod
        def _get_model_to_save(pl_module):
            model_obj = pl_module.model_wrapper.model
            return model_obj.module if hasattr(model_obj, "module") else model_obj

        def _export_torchscript_cpu(self, pl_module, model_to_save: torch.nn.Module) -> "torch.jit.ScriptModule":
            # Export a CPU TorchScript module without permanently mutating the training model.
            import copy

            m = model_to_save
            was_training = m.training
            m.eval()
            try:
                try:
                    m_cpu = copy.deepcopy(m).cpu()
                except Exception:
                    # Fallback: temporary move (will be restored).
                    dev = next(m.parameters()).device
                    m_cpu = m.to("cpu")
                with torch.no_grad():
                    # Prefer script (best portability), but fall back to trace for models
                    # that contain non-scriptable code paths (e.g. SyncBatchNorm + torch.distributed).
                    try:
                        scripted = torch.jit.script(m_cpu)
                        return scripted
                    except Exception as e:
                        print(f"[CKPT-EXPORT][torchscript] script failed, fallback to trace: {type(e).__name__}: {e}", flush=True)

                    class _InferWrapper(torch.nn.Module):
                        def __init__(self, inner: torch.nn.Module):
                            super().__init__()
                            self.inner = inner

                        def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
                            if hasattr(self.inner, "_forward"):
                                return self.inner._forward(x1, x2)
                            out = self.inner(x1, x2)
                            return out[0] if isinstance(out, (tuple, list)) else out

                    wrapper = _InferWrapper(m_cpu).eval()
                    ts = int(getattr(getattr(pl_module, "opt", None), "train_size", 512) or 512)
                    x1 = torch.randn(1, 3, ts, ts, device="cpu")
                    x2 = torch.randn(1, 3, ts, ts, device="cpu")
                    traced = torch.jit.trace(wrapper, (x1, x2), strict=False)
                    traced = torch.jit.freeze(traced)
                    return traced
            finally:
                # Best-effort restore mode/device if we moved the live module.
                try:
                    if "dev" in locals():
                        m.to(dev)
                except Exception:
                    pass
                try:
                    if was_training:
                        m.train()
                except Exception:
                    pass

        def _build_payload(self, pl_module, score: float):
            model_to_save = self._get_model_to_save(pl_module)
            if self.save_mode == "torchscript":
                return self._export_torchscript_cpu(pl_module, model_to_save)
            if self.save_mode == "full_model":
                return model_to_save
            return {
                "network": model_to_save.state_dict(),
                "epoch": int(pl_module.trainer.current_epoch),
                "meta": {
                    "val_iou_1": score,
                    "save_mode": self.save_mode,
                },
            }

        def _save_pt(self, pl_module, score: float, out_path: str):
            payload = self._build_payload(pl_module, score)
            if self.save_mode == "torchscript":
                torch.jit.save(payload, out_path)
            else:
                torch.save(payload, out_path)
            print(f"[CKPT-EXPORT] saved ({self.save_mode}): {out_path}", flush=True)

        def on_validation_epoch_end(self, trainer, pl_module):
            if trainer.global_rank != 0:
                return
            if trainer.sanity_checking:
                return

            score = self._to_float(trainer.callback_metrics.get("val_iou_1"))
            if score is None:
                return

            if self.save_mode == "torchscript":
                ext = ".pt"
            elif self.save_mode == "full_model":
                ext = ".ckpt"
            else:
                ext = ".pt"

            # Save rolling latest checkpoint
            last_path = os.path.join(self.out_dir, f"last{ext}")
            self._save_pt(pl_module, score, last_path)

            # Save new best .pt
            if (self.best_score is None) or (score > self.best_score):
                self.best_score = score
                best_name = (
                    f"best-model-epoch={trainer.current_epoch:03d}"
                    f"-step={trainer.global_step:06d}"
                    f"-val_iou_1={score:.4f}{ext}"
                )
                best_path = os.path.join(self.out_dir, best_name)
                self._save_pt(pl_module, score, best_path)

    # NOTE: disabled. We only save .ckpt via Lightning ModelCheckpoint.
    # callbacks.append(ExportPtCheckpointCallback(ckpt_dir, save_mode=opt.save_ckpt_mode))

    # Progress bar:
    # RichProgressBar can crash in some DDP / redirected-output environments (seen: IndexError in rich live stack).
    # Use TQDMProgressBar for stability.
    if sys.stdout.isatty():
        callbacks.append(TQDMProgressBar())
    callbacks.append(ModelSummary(max_depth=3))

    # Print one-line checkpoint path per epoch (rank0 only)
    class PrintCheckpointPathCallback(pl.Callback):
        def on_validation_epoch_end(self, trainer, pl_module):
            if trainer.global_rank != 0:
                return
            if not use_lightning_ckpt:
                return
            cb = trainer.checkpoint_callback
            if cb is None:
                return
            last_path = getattr(cb, "last_model_path", "")
            best_path = getattr(cb, "best_model_path", "")
            if last_path:
                print(f"[CKPT] epoch={trainer.current_epoch} last: {last_path}", flush=True)
            if best_path:
                print(f"[CKPT] epoch={trainer.current_epoch} best: {best_path}", flush=True)

    callbacks.append(PrintCheckpointPathCallback())
    
    # ===== 创建 Trainer：形式对齐 CSDNet main.py 的多卡配置 =====
    # CSDNet: Trainer(strategy=args.strategy, devices=args.devices, accelerator=args.accelerator, precision=args.precision, ...)
    # 默认跑“全量 epoch”（limit_train_batches=1.0 表示 100% batches）。
    # 如需快速调试，可设置环境变量：
    #   MAX_TRAIN_ITERS=30  -> 每个 epoch 只跑 30 个 train batch
    #   MAX_VAL_ITERS=15    -> 每次验证只跑 15 个 val batch
    max_train_iters = os.environ.get("MAX_TRAIN_ITERS", "").strip()
    max_val_iters = os.environ.get("MAX_VAL_ITERS", "").strip()
    limit_train_batches = int(max_train_iters) if max_train_iters else 1.0
    limit_val_batches = int(max_val_iters) if max_val_iters else 1.0

    trainer = Trainer(
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        max_epochs=opt.num_epochs,
        accelerator=opt.accelerator,
        devices=opt.devices,
        strategy=opt.strategy,
        precision=opt.precision,
        callbacks=callbacks,
        logger=CSVLogger(save_dir=os.path.join(opt.checkpoint_dir, opt.name), name="logs"),
        enable_progress_bar=sys.stdout.isatty(),
        enable_model_summary=True,
        # 你当前 limit_train_batches=30，原来 50 会导致一个 epoch 内几乎看不到 lightning 日志
        log_every_n_steps=1,
        val_check_interval=1.0,  # Validate every epoch
        num_sanity_val_steps=0,  # Skip sanity check
        sync_batchnorm=True if str(opt.strategy).startswith("ddp") else False,
        # Always enable Lightning checkpointing (.ckpt only).
        enable_checkpointing=True,
    )
    
    # Start training
    trainer.fit(model, data_module)
    
    print("Done!")
