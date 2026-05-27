

export CUDA_VISIBLE_DEVICES=0,1,2,3

# Quick validation switches:
# - MAX_TRAIN_ITERS / MAX_VAL_ITERS: run only first N train/val batches each epoch (fast smoke test).
#   Example (fast):  MAX_TRAIN_ITERS=10 MAX_VAL_ITERS=5  bash trainval.sh
#   Example (full):  MAX_TRAIN_ITERS=  MAX_VAL_ITERS=    bash trainval.sh
# : "${MAX_TRAIN_ITERS:=5}"
# : "${MAX_VAL_ITERS:=2}"

nohup /root/anaconda3/envs/changedino/bin/python3 /mnt/wuxy/change_detection/ChangeDINO_dynamic_DDP/trainval.py \
  --accelerator gpu --devices 4 --strategy ddp_find_unused_parameters_true --precision 16  \
  > /mnt/wuxy/change_detection/ChangeDINO_dynamic_DDP/bash_logs/train/suizhou_0408.log 2>&1 &