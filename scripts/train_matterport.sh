#!/bin/bash
export MATTERPORT_PATH=matterport

EXPNAME=matterport

nice -n 19 python train.py --name ${EXPNAME} --gpus=10 --batch=6 \
        --lr=5e-4 --fusion_transformer --transformer_depth 6 \
        --w_tr 10 --w_rot 10 --steps 120000 \
        --datapath=$MATTERPORT_PATH --dataset matterport 
