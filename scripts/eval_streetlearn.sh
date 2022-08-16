#!/bin/bash
export INTERIORNET_STREETLEARN_PATH=/home/cnris/vl/ExtremeRotation_code

# TRAINED
# CKPT=output/streetlearn/checkpoints/120000.pth

# PRETRAINED
CKPT=pretrained_models/streetlearn.pth

EXPNAME=streetlearn

nice -n 19 python test_streetlearn_interiornet.py --exp ${EXPNAME} --transformer_depth 6 \
        --fusion_transformer --ckpt $CKPT \
        --datapath=$INTERIORNET_STREETLEARN_PATH --dataset streetlearn
