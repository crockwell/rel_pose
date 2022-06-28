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

from geom import projective_ops, losses

from model import ViTEss
from collections import OrderedDict
import pickle

from lietorch import SE3

DEPTH_SCALE = 5

def eval_camera(predictions):
    acc_threshold = {
        "tran": 1.0,
        "rot": 30,
    }  # threshold for translation and rotation error to say prediction is correct.

    pred_tran = np.vstack(predictions["camera"]["preds"]["tran"])
    pred_rot = np.vstack(predictions["camera"]["preds"]["rot"])

    gt_tran = np.vstack(predictions["camera"]["gts"]["tran"])
    gt_rot = np.vstack(predictions["camera"]["gts"]["rot"])

    top1_error = {
        "tran": np.linalg.norm(gt_tran - pred_tran, axis=1),
        "rot": 2 * np.arccos(np.clip(np.abs(np.sum(np.multiply(pred_rot, gt_rot), axis=1)), -1.0, 1.0)) * 180 / np.pi,
    }
    top1_accuracy = {
        "tran": (top1_error["tran"] < acc_threshold["tran"]).sum()
        / len(top1_error["tran"]),
        "rot": (top1_error["rot"] < acc_threshold["rot"]).sum()
        / len(top1_error["rot"]),
    }
    camera_metrics = {
        f"top1 T err < {acc_threshold['tran']}": top1_accuracy["tran"] * 100,
        f"top1 R err < {acc_threshold['rot']}": top1_accuracy["rot"] * 100,
        f"T mean err": np.mean(top1_error["tran"]),
        f"R mean err": np.mean(top1_error["rot"]),
        f"T median err": np.median(top1_error["tran"]),
        f"R median err": np.median(top1_error["rot"]),
    }
    
    gt_mags = {"tran": np.linalg.norm(gt_tran, axis=1), "rot": 2 * np.arccos(gt_rot[:,0]) * 180 / np.pi}

    tran_graph = np.stack([gt_mags['tran'], top1_error['tran']],axis=1)
    tran_graph_name = os.path.join('output', args.exp, output_folder, args.weights[:-4], 'tran_graph.csv')
    np.savetxt(tran_graph_name, tran_graph, delimiter=',', fmt='%1.5f')

    rot_graph = np.stack([gt_mags['rot'], top1_error['rot']],axis=1)
    rot_graph_name = os.path.join('output', args.exp, output_folder, args.weights[:-4], 'rot_graph.csv')
    np.savetxt(rot_graph_name, rot_graph, delimiter=',', fmt='%1.5f')
    
    return camera_metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--datapath")
    parser.add_argument("--weights")
    parser.add_argument("--image_size", default=[384,512])
    parser.add_argument("--exp")
    parser.add_argument("--checkpoint_dir")
    parser.add_argument('--gamma', type=float, default=0.9)    

    # model
    parser.add_argument('--no_pos_encoding', action='store_true')
    parser.add_argument('--noess', action='store_true')
    parser.add_argument('--cross_features', action='store_true')
    parser.add_argument('--use_single_softmax', action='store_true')  
    parser.add_argument('--l1_pos_encoding', action='store_true')
    parser.add_argument('--fusion_transformer', action="store_true", default=False)
    parser.add_argument('--fc_hidden_size', type=int, default=512)
    parser.add_argument('--pool_size', type=int, default=60)
    parser.add_argument('--transformer_depth', type=int, default=6)

    args = parser.parse_args()
    torch.multiprocessing.set_start_method('spawn')

    from data_readers.matterport import test_split, cur_path

    dset = test_split
    output_folder = 'matterport_test'

    print('performing evaluation on %s set using model %s' % (output_folder, args.checkpoint_dir))

    ate_list = []
    named_ates = []   
    geo_list_tr = []
    named_geos_tr = []
    geo_list_rot = []
    named_geos_rot = []
    rotation_mags_list = []
    rotation_mags_gt_list = []
    named_rotation_mags = []
    named_rotation_mags_gt = []


    kmeans_trans_path = '/home/cnris/vl/SparsePlanes/sparsePlane/models/kmeans_trans_32.pkl'
    kmeans_rots_path = '/home/cnris/vl/SparsePlanes/sparsePlane/models/kmeans_rots_32.pkl'
    assert os.path.exists(kmeans_trans_path)
    assert os.path.exists(kmeans_rots_path)
    with open(kmeans_trans_path, "rb") as f:
        kmeans_trans = pickle.load(f)
    with open(kmeans_rots_path, "rb") as f:
        kmeans_rots = pickle.load(f)

    checkpoint_dir = args.exp
    if args.checkpoint_dir is not None:
        checkpoint_dir = args.checkpoint_dir

    try:
        os.makedirs(os.path.join('output', args.exp, output_folder, args.weights[:-4]))
    except:
        pass

    all_geo_loss_rot, all_geo_loss_tr, all_rotation_mags, all_rotation_mags_gt = [], [], [], []

    model = ViTEss(args)
    state_dict = OrderedDict([
        (k.replace("module.", ""), v) for (k, v) in torch.load(os.path.join('output', args.checkpoint_dir,'checkpoints', args.weights))['model'].items()])
    model.load_state_dict(state_dict)
    model = model.cuda().eval()
    
    train_val = ''
    predictions = {'camera': {'preds': {'tran': [], 'rot': []}, 'gts': {'tran': [], 'rot': []}}}
    metrics = {'_geo_loss_tr': [], '_geo_loss_rot': []}

    for i in tqdm(range(len(dset['data']))):
        images = []
        for imgnum in ['0', '1']:
            img_name = os.path.join(cur_path, '/'.join(str(dset['data'][i][imgnum]['file_name']).split('/')[6:]))
            images.append(cv2.imread(img_name))

        images = np.stack(images).astype(np.float32)
        images = torch.from_numpy(images).float()
        images = images.permute(0, 3, 1, 2)
        images = F.interpolate(images, size=[384,512])
        images = images.unsqueeze(0).cuda()
        intrinsics = np.stack([np.array([[517.97, 517.97, 320, 240], [517.97, 517.97, 320, 240]])]).astype(np.float32)
        intrinsics = torch.from_numpy(intrinsics).cuda()

        base_pose = np.array([0,0,0,0,0,0,1])
        rel_pose = np.array(dset['data'][i]['rel_pose']['position'] + dset['data'][i]['rel_pose']['rotation'])
        print(rel_pose)
        import pdb; pdb.set_trace()
        og_rel_pose = np.copy(rel_pose)
        rel_pose[:3] /= DEPTH_SCALE
        cprp = np.copy(rel_pose)
        rel_pose[6] = cprp[3] # swap 3 & 6, we want W last.
        rel_pose[3] = cprp[6]
        if rel_pose[6] < 0:
            rel_pose[3:] *= -1
        poses = np.vstack([base_pose, rel_pose]).astype(np.float32)
        poses = torch.from_numpy(poses).unsqueeze(0).cuda()
        Ps = SE3(poses)

        Gs = SE3.IdentityLike(Ps)

        N=2
        graph = OrderedDict()
        for ll in range(N):
            graph[ll] = [j for j in range(N) if ll!=j and abs(ll-j) <= 2]
                    
        with torch.no_grad():
            poses_est = model(images, Gs, intrinsics=intrinsics)
            geo_loss_tr, geo_loss_rot, rotation_mag, rotation_mag_gt, geo_metrics = losses.geodesic_loss(Ps, poses_est, \
                    graph, do_scale=False, train_val=train_val, gamma=args.gamma)

        predictions['camera']['gts']['tran'].append(dset['data'][i]['rel_pose']['position'])
        gt_rotation = dset['data'][i]['rel_pose']['rotation']
        if gt_rotation[0] < 0: # normalize quaternions to have positive "W"
            gt_rotation[0] *= -1
            gt_rotation[1] *= -1
            gt_rotation[2] *= -1
            gt_rotation[3] *= -1
        predictions['camera']['gts']['rot'].append(gt_rotation)

        preds = poses_est[0][0][1].data.cpu().numpy()    
        pr_copy = np.copy(preds)
        preds[3] = pr_copy[6] # swap 3 & 6, we used W last; want W first in quat
        preds[6] = pr_copy[3]
        preds[:3] = preds[:3] * DEPTH_SCALE # undo scale change we made during training

        predictions['camera']['preds']['tran'].append(preds[:3])
        predictions['camera']['preds']['rot'].append(preds[3:])

        metrics['_geo_loss_tr'].append(geo_metrics['_geo_loss_tr'])
        metrics['_geo_loss_rot'].append(geo_metrics['_geo_loss_rot'])

    print('mean geo tr', np.mean(np.array(metrics['_geo_loss_tr'])))
    print('mean geo rot', np.mean(np.array(metrics['_geo_loss_rot'])))

    camera_metrics = eval_camera(predictions)
    for k in camera_metrics:
        print(k, camera_metrics[k])
    
    with open(os.path.join('output', args.exp, output_folder, args.weights[:-4], 'results.txt'), 'w') as f:
        print('mean geo tr', np.mean(np.array(metrics['_geo_loss_tr'])), file=f)
        print('mean geo rot', np.mean(np.array(metrics['_geo_loss_rot'])), file=f)
        for k in camera_metrics:
            print(k, camera_metrics[k], file=f)
