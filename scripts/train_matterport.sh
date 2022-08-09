#!/bin/bash
sh scripts/export_paths.sh

EXPNAME=matterport
POOL_SIZE=60
WEIGHTS=120000.pth

nice -n 19 python train_cnn.py --name ${EXPNAME} --gpus=10 --batch=6 \
        --lr=5e-4 --fusion_transformer --transformer_depth 6 \
        --w_tr 10 --w_rot 10 --steps 120000 \
        --datapath=$MATTERPORT_PATH --dataset matterport 
