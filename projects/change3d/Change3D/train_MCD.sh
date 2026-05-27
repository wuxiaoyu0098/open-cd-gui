#!/bin/bash

# 指定使用 GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 训练吞吐常用加速开关（按需调整）
# export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
# export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

# Quick validation switches:
# - MAX_STEPS: total training steps
# - VAL_CHECK_INTERVAL: run validation every N training steps
#   Example (full): MAX_STEPS=50000 VAL_CHECK_INTERVAL=1000 bash train_BCD.sh
#   Example (fast): MAX_STEPS=8 VAL_CHECK_INTERVAL=4 bash train_BCD.sh
: "${MAX_STEPS:=50000}"
: "${VAL_CHECK_INTERVAL:=204}"

# 避免 CUDA 内存碎片（可选，但经常有帮助）
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"


nohup /mnt/wuxy/conda_envs/Change3D/bin/python scripts/train_MCD_v1.py \
  --batch_size 6 \
  --devices 4 \
  --precision 16 \
  --strategy ddp_find_unused_parameters_true \
  --num_workers 16 \
  --dataset suizhou_MCD \
  --file_root /mnt/wuxy/change_detection/datasets/all_elements/suizhou_MCD/0429_1 \
  --save_dir /mnt/wuxy/change_detection/Change3D/checkpoints/suizhou_MCD_1PF/suizhou_MCD_1212 \
  --max_steps "${MAX_STEPS}" \
  --val_check_interval "${VAL_CHECK_INTERVAL}" \
  > /mnt/wuxy/change_detection/Change3D/bash_files/train/suizhou_MCD_1PF/0429_suizhou_MCD_1212.log 2>&1 &

