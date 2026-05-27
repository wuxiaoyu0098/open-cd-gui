from .ChangeDINO import ChangeModel
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import os
import torch.optim as optim
from .loss.focal import FocalLoss
from .loss.dice import DICELoss


def get_model(backbone_name="mobilenetv2", fpn_channels=128, n_layers=[1, 1, 1], **kwargs):
    model = ChangeModel(backbone_name, fpn_channels, n_layers=n_layers, **kwargs)
    # print(model)
    return model


class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        # choose device safely: use first gpu_id when provided, otherwise CPU
        use_cuda = torch.cuda.is_available() and len(opt.gpu_ids) > 0
        self.device = torch.device(f"cuda:{opt.gpu_ids[0]}" if use_cuda else "cpu")
        self.opt = opt
        self.base_lr = opt.lr
        self.save_dir = os.path.join(opt.checkpoint_dir, opt.name)
        os.makedirs(self.save_dir, exist_ok=True)

        self.model = get_model(
            backbone_name=opt.backbone,
            fpn_name=opt.fpn,
            fpn_channels=opt.fpn_channels,
            deform_groups=opt.deform_groups,
            gamma_mode=opt.gamma_mode,
            beta_mode=opt.beta_mode,
            n_layers=opt.n_layers,
            extract_ids=opt.extract_ids,
        )
        # 先把模型移动到目标设备，再创建优化器，避免部分参数/状态停留在 CPU
        self.model.to(self.device)

        self.focal = FocalLoss(alpha=opt.alpha, gamma=opt.gamma)
        self.dice = DICELoss()
        

        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay
        )
        self.schedular = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, opt.num_epochs, eta_min=1e-7
        )
        self.resume_info = None
        if opt.load_pretrain:
            if getattr(opt, 'resume', False):
                # 断点续训：加载完整的训练状态
                self.resume_info = self.load_resume_ckpt(self.model, self.optimizer, self.schedular, opt.name, opt.backbone)
            else:
                # 仅加载模型权重（用于测试）
                self.load_ckpt(self.model, self.optimizer, opt.name, opt.backbone)
        # 打印任意一个参数的均值，确认加载前后变化（也可以更严格）
        p0 = next(self.model.parameters()).detach().float().mean().item()
        print(f"[CKPT] after load, first param mean = {p0:.6f}")

        # 确保模型在目标设备
        self.model.to(self.device)

        print("---------- Networks initialized -------------")

    def forward(self, x1, x2, label):
        final_pred, preds = self.model(x1, x2)
        # 保证预测与标签空间尺寸一致（移除硬编码 256x256）
        target_size = label.shape[-2:]
        if final_pred.shape[-2:] != target_size:
            final_pred = F.interpolate(final_pred, size=target_size, mode="bilinear", align_corners=False)
        resized_preds = []
        for p in preds:
            if p.shape[-2:] != target_size:
                p = F.interpolate(p, size=target_size, mode="bilinear", align_corners=False)
            resized_preds.append(p)

        label = label.long()
        focal = self.focal(final_pred, label)
        dice = self.dice(final_pred, label)
        for p in resized_preds:
            focal += self.focal(p, label)
            dice += 0.5 * self.dice(p, label)

        return final_pred, focal, dice

    @torch.inference_mode()
    def inference(self, x1, x2):
        pred = self.model._forward(x1, x2)
        return pred

    def load_ckpt(self, network, optimizer, name, backbone):
        save_filename = "%s_%s_best.pth" % (name, backbone)
        save_path = os.path.join(self.save_dir, save_filename)
        print(f"[CKPT] will load: {save_path}")

        if not os.path.isfile(save_path):
            print("%s not exists yet!" % save_path)
            raise ("%s must exist!" % save_filename)
        else:
            # weights_only=False 因为 checkpoint 可能包含 numpy scalar 等元数据
            checkpoint = torch.load(
                save_path, map_location=self.device, weights_only=False
            )
            state_dict = self._extract_network_state_dict(checkpoint, network)
            if isinstance(checkpoint, dict):
                print(f"[CKPT] keys: {list(checkpoint.keys())}")
                for k in ["epoch", "iter", "best", "metric", "acc", "miou"]:
                    if k in checkpoint:
                        print(f"[CKPT] {k} = {checkpoint[k]}")

            network.load_state_dict(state_dict, strict=False)
            print("load pre-trained")

    @staticmethod
    def _extract_network_state_dict(checkpoint, network):
        if not isinstance(checkpoint, dict):
            raise ValueError("Unsupported checkpoint format.")

        if "network" in checkpoint:
            return checkpoint["network"]
        if "state_dict" in checkpoint:
            raw_state = checkpoint["state_dict"]
        else:
            raw_state = checkpoint

        target_keys = set(network.state_dict().keys())
        prefixes = [
            "model.",
            "model.model.",
            "model_wrapper.model.",
            "model_wrapper.model.model.",
            "",
        ]

        def score_prefix(prefix):
            score = 0
            for key in raw_state.keys():
                new_key = key[len(prefix):] if prefix and key.startswith(prefix) else key
                if new_key in target_keys:
                    score += 1
            return score

        best_prefix = max(prefixes, key=score_prefix)
        converted = {}
        for key, value in raw_state.items():
            new_key = key[len(best_prefix):] if best_prefix and key.startswith(best_prefix) else key
            if new_key in target_keys:
                converted[new_key] = value
        print(f"[CKPT] converted state keys: {len(converted)}/{len(target_keys)}")
        return converted
    
    def load_resume_ckpt(self, network, optimizer, scheduler, name, backbone):
        """加载完整的训练状态用于断点续训"""
        save_filename = "%s_%s_best.pth" % (name, backbone)
        save_path = os.path.join(self.save_dir, save_filename)
        if not os.path.isfile(save_path):
            print("%s not exists yet!" % save_path)
            raise ("%s must exist!" % save_filename)
        else:
            # weights_only=False 因为 checkpoint 可能包含 numpy scalar 等元数据
            checkpoint = torch.load(
                save_path, map_location=self.device, weights_only=False
            )
            network.load_state_dict(checkpoint["network"], strict=False)
            
            # 兼容旧的checkpoint格式：如果存在optimizer和scheduler，则加载
            if "optimizer" in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                    print("Loaded optimizer state from checkpoint")
                except Exception as e:
                    print(f"Warning: Failed to load optimizer state: {e}")
            
            if "scheduler" in checkpoint and scheduler is not None:
                try:
                    scheduler.load_state_dict(checkpoint["scheduler"])
                    print("Loaded scheduler state from checkpoint")
                except Exception as e:
                    print(f"Warning: Failed to load scheduler state: {e}")
            
            # 尝试从checkpoint中读取epoch和previous_best
            epoch = checkpoint.get("epoch", None)
            previous_best = checkpoint.get("previous_best", None)
            
            # 如果checkpoint中没有这些信息（旧格式），尝试从record.txt读取
            if epoch is None or previous_best is None:
                log_path = os.path.join(self.save_dir, "record.txt")
                if os.path.exists(log_path):
                    epoch, previous_best = self._read_training_info_from_log(log_path)
                    if epoch is not None:
                        print(f"Read training info from record.txt: epoch={epoch}, best IoU={previous_best*100:.3f}%")
            
            # 如果仍然没有找到，使用默认值
            resume_info = {
                "epoch": epoch if epoch is not None else 0,
                "previous_best": previous_best if previous_best is not None else 0.0,
            }
            
            print(f"Resume training from epoch {resume_info['epoch']}, previous best IoU: {resume_info['previous_best']*100:.3f}%")
            return resume_info
    
    def _read_training_info_from_log(self, log_path):
        """从record.txt文件中读取最后一个epoch和最佳IoU值"""
        import json
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            # 跳过注释行，找到最后一行数据
            last_epoch = None
            last_best_iou = None
            
            for line in lines:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                
                # 解析CSV格式：time,epoch,train_loss,train_focal,train_dice,lr,val_metrics(json)
                parts = line.split(",", 6)  # 最多分割6次，保留最后一个部分（JSON）
                if len(parts) >= 7:
                    try:
                        epoch = int(parts[1])
                        val_metrics_str = parts[6]
                        val_metrics = json.loads(val_metrics_str)
                        iou_1 = val_metrics.get("iou_1", 0.0)
                        
                        last_epoch = epoch
                        # 记录最大的iou_1作为best
                        if last_best_iou is None or iou_1 > last_best_iou:
                            last_best_iou = iou_1
                    except (ValueError, json.JSONDecodeError) as e:
                        continue
            
            return last_epoch, last_best_iou
        except Exception as e:
            print(f"Warning: Failed to read training info from log: {e}")
            return None, None

    def save_ckpt(self, network, optimizer, model_name, backbone, scheduler=None, epoch=None, previous_best=None,
                  iou_value=None, is_best=False):
        # 如果提供了epoch和iou_value，使用新的命名格式
        if epoch is not None and iou_value is not None:
            # IoU值格式化为4位小数，例如 0.7456 -> 07456 (去掉小数点，保留4位)
            iou_str = f"{iou_value * 10000:.0f}".zfill(5)  # 例如 0.7456 -> 07456
            save_filename = "%s_%s_e%03d_iou%s.pth" % (model_name, backbone, epoch, iou_str)
        elif is_best:
            # 保留best文件的命名
            save_filename = "%s_%s_best.pth" % (model_name, backbone)
            # 如果best文件已存在，先删除
            save_path = os.path.join(self.save_dir, save_filename)
            if os.path.exists(save_path):
                os.remove(save_path)
        else:
            # 默认命名
            save_filename = "%s_%s_best.pth" % (model_name, backbone)

        save_path = os.path.join(self.save_dir, save_filename)

        checkpoint_dict = {
            "network": network.cpu().state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        # 保存额外的训练状态信息用于断点续训
        if scheduler is not None:
            checkpoint_dict["scheduler"] = scheduler.state_dict()
        if epoch is not None:
            checkpoint_dict["epoch"] = epoch
        if previous_best is not None:
            checkpoint_dict["previous_best"] = previous_best
        if iou_value is not None:
            checkpoint_dict["iou_value"] = iou_value

        torch.save(checkpoint_dict, save_path)
        if torch.cuda.is_available():
            network.cuda()

    
    def save(self, model_name, backbone, scheduler=None, epoch=None, previous_best=None, iou_value=None, is_best=False):
        self.save_ckpt(self.model, self.optimizer, model_name, backbone, scheduler, epoch, previous_best, iou_value,
                       is_best)

    def name(self):
        return self.opt.name


def create_model(opt):
    model = Model(opt)
    print("model [%s] was created" % model.name())

    return model.to(model.device)
