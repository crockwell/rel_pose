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
    images = F.interpolate(images, size=[384,512])
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

    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=5)
    if "matterport" in args.ckpt:
        print("predicted R&t, as quaternion, in format x,y,z,qx,qy,qz,qw:")
        print(preds)
    else:
        print("predicted R, as quaternion in format qx,qy,qz,qw")
        print(preds[3:])


# Expected outputs:
# mp: [ 2.17275  0.1722  -0.87071  0.10733  0.00044  0.54702  0.83021] (image pair 301)
# ground truth: 

# in: [ 0.63364 -0.11078 -0.12625  0.75518] (image pair 245)
# ground truth: 

# sl: [0.40741 0.257   0.18473 0.85665] (image pair 138)
# ground truth: 