from PIL import Image  # Import before torch to avoid Windows DLL load issues.
import torchvision.transforms.functional as TF  # noqa: F401

import torch
import torch.nn.functional as F
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm

from util.metric_tool import ConfuseMatrixMeter
from util.util import make_numpy_grid, de_norm
from option import Options
from data.cd_dataset import DataLoader
from model.create_ChangeDINO import create_model

if __name__ == "__main__":
    opt = Options().parse()
    opt.phase = getattr(opt, "test_phase", "test")
    test_loader = DataLoader(opt)
    test_data = test_loader.load_data()
    test_size = len(test_loader)
    print("#testing images = %d" % test_size)

    opt.load_pretrain = True
    model = create_model(opt)
    device = model.device

    tbar = tqdm(test_data, ncols=80, disable=not sys.stdout.isatty())
    total_iters = test_size
    running_metric = ConfuseMatrixMeter(n_class=2)
    running_metric.clear()

    # 使用 result_dir 保存结果
    result_save_dir = os.path.join(opt.result_dir, opt.name)
    os.makedirs(result_save_dir, exist_ok=True)
    
    # 预测结果保存路径
    test_save_path = os.path.join(result_save_dir, "pred")
    if opt.save_test and not os.path.exists(test_save_path):
        os.makedirs(test_save_path, exist_ok=True)
    
    # 可视化结果保存路径
    vis_save_path = os.path.join(result_save_dir, "vis")
    if opt.save_test and not os.path.exists(vis_save_path):
        os.makedirs(vis_save_path, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        for i, _data in enumerate(tbar):
            if (i + 1) % 10 == 0 or (i + 1) == total_iters:
                print(f"[PROGRESS][test] step={i + 1}/{total_iters}", flush=True)
            img1 = _data["img1"].to(device)
            img2 = _data["img2"].to(device)
            val_pred = model.inference(img1, img2)
            # update metric
            val_target = _data["cd_label"].detach()
            # Resize prediction to match ground truth size
            target_size = val_target.shape[-2:]
            if val_pred.shape[-2:] != target_size:
                val_pred = F.interpolate(val_pred, size=target_size, mode="bilinear", align_corners=False)
            val_pred = torch.argmax(val_pred.detach(), dim=1)
            _ = running_metric.update_cm(
                pr=val_pred.cpu().detach().numpy(), gt=val_target.cpu().detach().numpy()
            )
            if opt.save_test:
                for j in range(val_pred.shape[0]):
                    # 保存单独的预测结果（二值图）
                    pred = Image.fromarray((val_pred[j].cpu().detach().numpy()*255).astype("uint8"))
                    pred.save(
                        os.path.join(test_save_path, _data["fname"][j])
                    )
                    
                    # 保存可视化对比图（Time 1, Time 2, Prediction, Ground Truth）
                    # 准备数据
                    x1_single = img1[j:j+1]  # [1, C, H, W]
                    x2_single = img2[j:j+1]
                    
                    # 使用与pred文件夹完全相同的处理方式：val_pred * 255
                    # pred文件夹保存的是 (val_pred * 255).astype("uint8")，即0或255
                    # vis中也使用相同的值，转换为[0,1]范围供matplotlib显示
                    pred_np = val_pred[j].cpu().detach().numpy() * 255.0  # [H, W], 值域[0, 255]
                    pred_single = torch.from_numpy(pred_np).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).float() / 255.0  # [1, 3, H, W], 值域[0,1]
                    
                    # Ground truth也使用相同的处理方式
                    gt_np = val_target[j].cpu().detach().numpy() * 255.0  # [H, W], 值域[0, 255]
                    gt_single = torch.from_numpy(gt_np).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).float() / 255.0  # [1, 3, H, W], 值域[0,1]
                    
                    # 反归一化原始图像
                    x1_vis = de_norm(x1_single.cpu())
                    x2_vis = de_norm(x2_single.cpu())
                    
                    # 转换为numpy并拼接
                    vis_input = make_numpy_grid(x1_vis)
                    vis_input2 = make_numpy_grid(x2_vis)
                    vis_pred = make_numpy_grid(pred_single.cpu())
                    vis_gt = make_numpy_grid(gt_single.cpu())
                    
                    # 使用 matplotlib 创建带标题的可视化
                    # 根据图片尺寸动态计算 figure 大小
                    img_height, img_width = vis_input.shape[:2]
                    fig_width = 10
                    fig_height = (img_height * 4) / img_width * fig_width  # 4张图片垂直排列
                    
                    fig, axes = plt.subplots(4, 1, figsize=(fig_width, fig_height))
                    titles = ["Time 1", "Time 2", "Prediction", "Ground Truth"]
                    images = [vis_input, vis_input2, vis_pred, vis_gt]
                    
                    for ax, img, title in zip(axes, images, titles):
                        ax.imshow(img)
                        ax.set_title(title, fontsize=16, fontweight='bold', pad=10)
                        ax.axis('off')
                    
                    plt.tight_layout()
                    
                    # 保存可视化结果
                    vis_filename = os.path.join(vis_save_path, _data["fname"][j].replace(".png", "_vis.jpg"))
                    plt.savefig(vis_filename, dpi=150, bbox_inches='tight')
                    plt.close()  # 关闭图形以释放内存
        val_scores = running_metric.get_scores()
        message = "(phase: %s) " % (opt.phase)
        for k, v in val_scores.items():
            message += "%s: %.4f " % (k, v * 100)
        print(message)
        
        # 保存评估指标到文件
        metrics_file = os.path.join(result_save_dir, "test_metrics.json")
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump({k: float(v) for k, v in val_scores.items()}, f, indent=2, ensure_ascii=False)
        print(f"评估指标已保存到: {metrics_file}")
