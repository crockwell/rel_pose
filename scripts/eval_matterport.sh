#!/bin/bash
sh scripts/export_paths.sh

# TRAINED
# CKPT=output/matterport/checkpoints/120000.pth

# PRETRAINED
CKPT=pretrained_models/matterport.pth

EXPNAME=matterport
POOL_SIZE=60
WEIGHTS=120000.pth

nice -n 19 python test_matterport.py --exp ${EXPNAME} --transformer_depth 6 \
        --fusion_transformer --ckpt $CKPT \
        --datapath=$MATTERPORT_PATH 
        
