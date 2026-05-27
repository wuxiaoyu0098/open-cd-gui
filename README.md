# OpenCD GUI 使用说明

这个项目现在在 OpenCD 基础上集成了一个图形界面，并加入了 Change3D、ChangeDINO 的训练和大图推理入口。

## 1. 启动 GUI

先进入 OpenCD 环境：

```powershell
conda activate opencd
cd D:\Projects\change_detection\open-cd-main
python tools\opencd_gui.py
```

界面主要有三个页签：

- `模型训练`
- `模型推理` 切片推理
- `大图推理` tif推理

三个页签都有自己的模型选择，不会互相联动。

## 2. 数据集格式

训练数据集目录需要是：

```text
dataset_root/
  train/
    A/
    B/
    label/
  val/
    A/
    B/
    label/
  test/
    A/
    B/
    label/
```

同一个 split 里，`A`、`B`、`label` 的文件名必须一一对应。

## 3. 模型训练

在 `模型训练` 页选择模型预设：

- `ChangeFormer-B0`
- `ChangeFormer-B1`
- `Changer-R18`
- `Change3D-BCD`
- `Change3D-MCD`
- `ChangeDINO-BCD`

常用参数：

- `训练 Epoch / 迭代次数`：Change3D 使用 Epoch，OpenCD 原生模型使用 iter。
- `Batch Size`：显存够可以调大。
- `Num Workers`：Windows 下如果报多进程错误，可以设为 0。
- `数据集目录`：选择包含 `train/val/test` 的根目录。
- `输出目录`：训练日志和权重保存位置。

Change3D 训练日志会显示：

```text
[PROGRESS] epoch=0 step=10/75 loss=...
```

GUI 进度条显示当前 epoch，例如：

```text
Epoch 1 / 120
```

## 4. 小图推理

在 `模型推理` 页选择模型预设，填写：

- `权重文件`
- `测试文件夹`
- `可视化输出目录`

测试文件夹格式：

```text
test/
  A/
  B/
  label/
```

说明：

- ChangeFormer / Changer 支持小图测试。
- ChangeDINO 已接入小图测试。
- Change3D 当前主要用于训练和大图推理，暂未接入小图测试。

## 5. 大图推理

在 `大图推理` 页选择模型预设，填写：

- `权重文件`
- `第一时相 TIF`
- `第二时相 TIF`
- `输出目录`

常用参数：

- `Tile Size`：滑窗大小，常用 512。
- `Overlap`：滑窗重叠，常用 128。
- `Batch Size`：显存够可以调大。
- `阈值`：二值化阈值。
- `孔洞面积`：后处理中填充小孔洞的面积阈值。
- `最小目标`：过滤小斑块的面积阈值。

输出的矢量文件统一命名为：

```text
change.shp
```

同时会生成 Shapefile 配套文件：

```text
change.shx
change.dbf
change.prj
change.cpg
```

## 6. Change3D 说明

Change3D 已集成：

- BCD 训练
- MCD 训练
- BCD 大图推理
- MCD 大图推理

Change3D BCD 训练保存权重后，会自动把 `.ckpt` 转成大图推理使用的 `.pt`，并只保留 `.pt`。

如果需要手动转换 BCD 权重：

```powershell
python projects\change3d\Change3D\ckpt_to_pt_3D_BCD.py ^
  --weights path\to\model.ckpt ^
  --out_pt path\to\model.pt ^
  --dataset Change3D-CD ^
  --in_height 512 ^
  --in_width 512 ^
  --pretrained projects\change3d\Change3D\model\X3D_L.pyth ^
  --cuda_no 0
```

Change3D 大图推理会直接调用：

```text
projects/change3d/Change3D/scripts/predict_BCD.py
projects/change3d/Change3D/scripts/predict_MCD.py
```

## 7. ChangeDINO 说明

ChangeDINO 已集成：

- BCD 训练
- 小图测试
- 大图推理

ChangeDINO 大图推理最终调用：

```text
projects/changedino/ChangeDINO/2.predict_big_pt_flow_v3.py
```

`--no_amp` 表示关闭 AMP 混合精度，稳定优先。如果想加速，可以不使用 `--no_amp`。


