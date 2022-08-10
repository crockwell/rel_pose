#!/bin/bash
export INTERIORNET_STREETLEARN_PATH=/home/cnris/vl/ExtremeRotation_code

# TRAINED
# CKPT=output/streetlearn_t/checkpoints/120000.pth

# PRETRAINED
CKPT=pretrained_models/streetlearn_t.pth

EXPNAME=streetlearn_t
POOL_SIZE=60
WEIGHTS=120000.pth

nice -n 19 python test_streetlearn_interiornet.py --exp ${EXPNAME} --transformer_depth 6 \
        --fusion_transformer --ckpt $CKPT \
        --datapath=$INTERIORNET_STREETLEARN_PATH --dataset streetlearn --streetlearn_interiornet_type T 
