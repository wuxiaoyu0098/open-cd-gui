#!/bin/bash

# 指定使用 GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Quick validation switches:
# - MAX_STEPS: total training steps
# - VAL_CHECK_INTERVAL: run validation every N training steps
#   Example (full): MAX_STEPS=50000 VAL_CHECK_INTERVAL=1000 bash train_BCD.sh
#   Example (fast): MAX_STEPS=8 VAL_CHECK_INTERVAL=4 bash train_BCD.sh
: "${MAX_STEPS:=50000}"
: "${VAL_CHECK_INTERVAL:=132}"

nohup /mnt/wuxy/conda_envs/Change3D/bin/python scripts/train_BCD.py --batch_size 8 --devices 4 --precision 16 --strategy ddp_find_unused_parameters_true  --dataset suizhou  --file_root /mnt/wuxy/change_detection/datasets/all_elements/suizhou_MCD/xiuzheng_0423/datasets --save_dir /mnt/wuxy/change_detection/Change3D/checkpoints/BCD/new_0424 --max_steps "${MAX_STEPS}" --val_check_interval "${VAL_CHECK_INTERVAL}" > /mnt/wuxy/change_detection/Change3D/bash_files/train/BCD/0424_suizhou_BCD_new.log 2>&1 &

