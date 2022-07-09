#!/bin/bash
#
# 3. Train Model with nonlinear classifier (densenet)
#    experiment name: pd_conservation003
#    
#    we run this experiment again because for some reason,
#    when executed with default batch size (32),
#    the accuracy did not change at all during training
python training.py \
    --dataset pcam \
    --max_time 00:12:00:00 \
    --save_checkpoint_every 1 \
    --run_name pd_conservation003 \
    --classifier NonLinear \
    --classifier_depth 3 \
    --classifier_classes 2 \
    --batch_size 128

# 4. Train Model with linear classifier
#    experiment name: pl_disentanglement
#    we test whether the encoding can learn
#    a linearly separable encoding
#
#    we run this experiment again with varying
#    batch sizes to see if this has an influence
#    on the noisyness of curves
python training.py \
    --dataset pcam \
    --max_time 00:04:00:00 \
    --save_checkpoint_every 1 \
    --run_name pl_dis64_003 \
    --classifier Linear \
    --classifier_classes 2 \
    --batch_size 64
#
python training.py \
    --dataset pcam \
    --max_time 00:04:00:00 \
    --save_checkpoint_every 1 \
    --run_name pl_dis128_003 \
    --classifier Linear \
    --classifier_classes 2 \
    --batch_size 128
#
