#!/bin/bash
export MATTERPORT_PATH=matterport

# TRAINED
# CKPT=output/matterport/checkpoints/120000.pth

# PRETRAINED
CKPT=pretrained_models/matterport.pth

EXPNAME=matterport

nice -n 19 python test_matterport.py --exp ${EXPNAME} --transformer_depth 6 \
        --fusion_transformer --ckpt $CKPT \
        --datapath=$MATTERPORT_PATH 
        
