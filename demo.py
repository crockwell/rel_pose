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

from src.model import ViTEss
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

    print('predicting pose on %s and %s using model %s' % (args.img1, args.img2, args.ckpt))

    # assume same intrinsics as training
    if "matterport" in args.ckpt:
        intrinsics = np.stack([np.array([[517.97, 517.97, 320, 240], [517.97, 517.97, 320, 240]])]).astype(np.float32)
    else:
        intrinsics = np.stack([np.array([[128,128,128,128], [128,128,128,128]])]).astype(np.float32)
    
    intrinsics = torch.from_numpy(intrinsics).cuda()

    model = ViTEss(args)
    state_dict = OrderedDict([
        (k.replace("module.", ""), v) for (k, v) in torch.load(args.ckpt)['model'].items()])
    model.load_state_dict(state_dict)
    model = model.cuda().eval()
    
    images = [cv2.imread(args.img1), cv2.imread(args.img2)]
    images = np.stack(images).astype(np.float32)
    images = torch.from_numpy(images).float()
    images = images.permute(0, 3, 1, 2)

    if "matterport" in args.ckpt:
        # following matterport data preprocessing we used during training
        images = F.interpolate(images, size=[384,512])
    else:
        # images are already correct size
        pass
    images = images.unsqueeze(0).cuda()

    base_pose = np.array([0,0,0,0,0,0,1])
    poses = np.vstack([base_pose, base_pose]).astype(np.float32)
    poses = torch.from_numpy(poses).unsqueeze(0).cuda()
    Gs = SE3(poses)
                
    with torch.no_grad():
        poses_est = model(images, Gs, intrinsics=intrinsics)

    preds = poses_est[0][0][1].data.cpu().numpy()    
    pr_copy = np.copy(preds)

    if "matterport" in args.ckpt:
        DEPTH_SCALE = 5
        preds[:3] = preds[:3] * DEPTH_SCALE # undo scale change we made during training
        preds[3:] = np.array([pr_copy[4],pr_copy[5],pr_copy[3],pr_copy[6]]) # on Matterport we predict in format yzxw, want xyzw

    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=5)
    if "matterport" in args.ckpt:
        print("predicted R&t, as quaternion, in format x,y,z,qx,qy,qz,qw:")
        print(preds)
    else:
        print("predicted R, as quaternion in format qx,qy,qz,qw")
        print(preds[3:])


########## Expected outputs ##########

# Matterport: image pair 5ZKStnWn8Zo/0_11_11.png 
#                        5ZKStnWn8Zo/0_11_51.png
# pred: [ 2.17275  0.1722   -0.87071  0.00044  0.54702  0.10733  0.83021] 
# gt:   [ 2.73153  0.25285  -1.35598  0.00000  0.56102  0.10905  0.82059]


# InteriorNet-T: image pair HD1/3FO4K4086OLR/original_7_7/0000000028151666688/043.png
#                           HD1/3FO4K4086OLR/original_1_1/0000000028111667200/010.png
# pred: [ 0.62947 -0.11058 -0.12595  0.75873]
# gt:   [ 0.62734 -0.12698 -0.11345  0.75990]


# Streetlearn-T: image pair  Ruy-1-EbfKAhoIQ6cNa5cw/078.png
#                            dcz-r3Si40Ptxdf2KwPalA/005.png
# pred: [ 0.39714  0.25738  0.18597  0.86108]
# gt:   [ 0.39073  0.27050  0.19321  0.85838]
