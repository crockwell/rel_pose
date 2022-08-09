import cv2
from tqdm import tqdm
import numpy as np
import torch
import lietorch
import os
import glob
import time
import yaml
import argparse

import torch 
import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

from geom import losses

from model import ViTEss
from collections import OrderedDict
import pickle

from lietorch import SE3

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--datapath")
    parser.add_argument("--weights")
    parser.add_argument("--image_size", default=[384,512])
    parser.add_argument("--img1")
    parser.add_argument("--img2")
    parser.add_argument("--ckpt")

    # model
    parser.add_argument('--no_pos_encoding', action='store_true')
    parser.add_argument('--noess', action='store_true')
    parser.add_argument('--cross_features', action='store_true')
    parser.add_argument('--use_single_softmax', action='store_true')  
    parser.add_argument('--l1_pos_encoding', action='store_true')
    parser.add_argument('--fc_hidden_size', type=int, default=512)
    parser.add_argument('--pool_size', type=int, default=60)
    parser.add_argument('--transformer_depth', type=int, default=6)

    args = parser.parse_args()
    args.fusion_transformer = True
    torch.multiprocessing.set_start_method('spawn')

    print('predicting pose on %s using model %s' % (args.img, args.ckpt))

    # assume same intrinsics as training
    intrinsics = np.stack([np.array([[517.97, 517.97, 320, 240], [517.97, 517.97, 320, 240]])]).astype(np.float32)
    intrinsics = torch.from_numpy(intrinsics).cuda()

    if args.ckpt is not None:
        ckpt = args.ckpt

    model = ViTEss(args)
    state_dict = OrderedDict([
        (k.replace("module.", ""), v) for (k, v) in torch.load(ckpt)['model'].items()])
    model.load_state_dict(state_dict)
    model = model.cuda().eval()
    
    images = [cv2.imread(args.img1), cv2.imread(args.img1)]
    images = np.stack(images).astype(np.float32)
    images = torch.from_numpy(images).float()
    images = images.permute(0, 3, 1, 2)
    images = F.interpolate(images, size=[384,512])
    images = images.unsqueeze(0).cuda()

    base_pose = np.array([0,0,0,0,0,0,1])
    Gs = SE3(base_pose)

    N=2
    graph = OrderedDict()
    for ll in range(N):
        graph[ll] = [j for j in range(N) if ll!=j and abs(ll-j) <= 2]
                
    with torch.no_grad():
        poses_est = model(images, Gs, intrinsics=intrinsics)

    preds = poses_est[0][0][1].data.cpu().numpy()    
    pr_copy = np.copy(preds)

    if "matterport" in ckpt:
        DEPTH_SCALE = 5
        preds[3] = pr_copy[6] # swap 3 & 6, we used W last in Matterport; want W first in quat
        preds[6] = pr_copy[3]
        preds[:3] = preds[:3] * DEPTH_SCALE # undo scale change we made during training

    print("predicted quaternion, in format x,y,z,qw,qx,qy,qz", preds)