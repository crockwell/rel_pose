#!/bin/bash
sh scripts/export_paths.sh

# TRAINED
# CKPT=output/streetlearn/checkpoints/120000.pth

# PRETRAINED
CKPT=pretrained_models/streetlearn.pth

EXPNAME=streetlearn
POOL_SIZE=60
WEIGHTS=120000.pth

nice -n 19 python test_streetlearn_interiornet.py --exp ${EXPNAME} --transformer_depth 6 \
        --fusion_transformer --ckpt $CKPT \
        --datapath=$INTERIORNET_STREETLEARN_PATH --dataset streetlearn
