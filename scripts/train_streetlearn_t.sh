#!/bin/bash
export INTERIORNET_STREETLEARN_PATH=/home/cnris/vl/ExtremeRotation_code

EXPNAME=streetlearn_t
POOL_SIZE=60
WEIGHTS=120000.pth

nice -n 19 python train.py --name ${EXPNAME} --gpus=10 --batch=6 \
        --lr=5e-4 --fusion_transformer --transformer_depth 6 \
        --w_tr 10 --w_rot 10 --steps 120000 \
        --datapath=$INTERIORNET_STREETLEARN_PATH --dataset streetlearn --streetlearn_interiornet_type T 
