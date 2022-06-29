#!/bin/bash
#
######################
# RUN PCAM EXPERIMENTS
######################
#
# 1. Train Model without classifier
#    experiment name: pn_reconstruction
#    we test the quality of image reconstruction when
#    this is the only optimization goal
python training.py \
    --dataset pcam \
    --max_time 00:12:00:00 \
    --save_checkpoint_every 1 \
    --run_name pn_reconstruction
#
#
# 2. Train Model with nonlinear classifier (resnet)
#    experiment name: pr_conservation
#    we test whether the encoding preserves the
#    information encoded in the labels
python training.py \
    --dataset pcam \
    --max_time 00:12:00:00 \
    --save_checkpoint_every 1 \
    --run_name pr_conservation \
    --classifier Resnet \
    --classifier_depth 3 \
    --classifier_classes 2
#
#
# 3. Train Model with nonlinear classifier (densenet)
#    experiment name: pd_conservation
#    see 2.
python training.py \
    --dataset pcam \
    --max_time 00:12:00:00 \
    --save_checkpoint_every 1 \
    --run_name pd_conservation \
    --classifier NonLinear \
    --classifier_depth 3 \
    --classifier_classes 2
#
# 4. Train Model with linear classifier
#    experiment name: pl_disentanglement
#    we test whether the encoding can learn
#    a linearly separable encoding
python training.py \
    --dataset pcam \
    --max_time 00:12:00:00 \
    --save_checkpoint_every 1 \
    --run_name pl_disentanglement \
    --classifier Linear \
    --classifier_classes 2
#
#
#
# 5. Future: measure linear separability with svm
#
#
#
#######################
# RUN MNIST EXPERIMENTS
#######################
#
# 1. Train Model without classifier
#    experiment name: mn_reconstruction
python training.py \
    --dataset mnist \
    --max_time 00:06:00:00 \
    --save_checkpoint_every 8 \
    --run_name mn_reconstruction
#
#
#
# 2. Train Model with nonlinear classifier (resnet)
#    experiment name: mr_conservation
python training.py \
    --dataset mnist \
    --max_time 00:06:00:00 \
    --save_checkpoint_every 8 \
    --run_name mr_conservation \
    --classifier Resnet \
    --classifier_depth 3 \
    --classifier_classes 10
#
#
# 3. Train Model with nonlinear classifier (densenet)
#    experiment name: md_conservation
python training.py \
    --dataset mnist \
    --max_time 00:06:00:00 \
    --save_checkpoint_every 8 \
    --run_name md_conservation \
    --classifier NonLinear \
    --classifier_depth 3 \
    --classifier_classes 10
#
# 4. Train Model with linear classifier
#    experiment name: ml_disentanglement
python training.py \
    --dataset mnist \
    --max_time 00:06:00:00 \
    --save_checkpoint_every 8 \
    --run_name ml_disentanglement \
    --classifier Linear \
    --classifier_classes 10
#
#
#
# 5. Future: measure linear separability with svm
#
#
