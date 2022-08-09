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
from scipy.spatial.transform import Rotation as R

from lietorch import SE3

def compute_angle_from_r_matrices(m):
    batch = m.shape[0]

    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()))
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda()) * -1)

    theta = torch.acos(cos)

    return theta

def compute_geodesic_distance_from_two_matrices(m1, m2):
    batch = m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3

    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()))
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda()) * -1)

    theta = torch.acos(cos)

    return theta

def compute_rotation_matrix_from_two_matrices(m1, m2):
        batch = m1.shape[0]
        m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3
        return m

def compute_rotation_matrix_from_viewpoint(rotation_x, rotation_y, batch):
    rotax = rotation_x.view(batch, 1).type(torch.FloatTensor)
    rotay = - rotation_y.view(batch, 1).type(torch.FloatTensor)

    c1 = torch.cos(rotax).view(batch, 1)  # batch*1
    s1 = torch.sin(rotax).view(batch, 1)  # batch*1
    c2 = torch.cos(rotay).view(batch, 1)  # batch*1
    s2 = torch.sin(rotay).view(batch, 1)  # batch*1

    # pitch --> yaw
    row1 = torch.cat((c2, s1 * s2, c1 * s2), 1).view(-1, 1, 3)  # batch*1*3
    row2 = torch.cat((torch.autograd.Variable(torch.zeros(s2.size())), c1, -s1), 1).view(-1, 1, 3)  # batch*1*3
    row3 = torch.cat((-s2, s1 * c2, c1 * c2), 1).view(-1, 1, 3)  # batch*1*3

    matrix = torch.cat((row1, row2, row3), 1)  # batch*3*3

    return matrix

def evaluation_metric_rotation(predict_rotation, gt_rotation, save_folder):
    geodesic_loss = compute_geodesic_distance_from_two_matrices(predict_rotation.view(-1, 3, 3),
                                                                gt_rotation.view(-1, 3, 3)) / np.pi * 180
    gt_distance = compute_angle_from_r_matrices(gt_rotation.view(-1, 3, 3))

    geodesic_loss_overlap_none = geodesic_loss[gt_distance.view(-1) > (np.pi / 2)]
    geodesic_loss_overlap_large = geodesic_loss[gt_distance.view(-1) < (np.pi / 4)]
    geodesic_loss_overlap_small = geodesic_loss[(gt_distance.view(-1) >= np.pi / 4) & (gt_distance.view(-1) < np.pi / 2)]

    all_rotation_err = geodesic_loss[gt_distance.view(-1) < (np.pi / 2)] 
    all_rotation_mags_gt = gt_distance[gt_distance.view(-1) < (np.pi / 2)] / np.pi * 180

    all_rotation_err = all_rotation_err.cpu().numpy().astype(np.float32)
    all_rotation_err_name = os.path.join(save_folder, 'all_rotation_err_degrees.csv')
    np.savetxt(all_rotation_err_name, all_rotation_err, delimiter=',', fmt='%1.5f')

    all_rotation_mags_gt = all_rotation_mags_gt.cpu().numpy().astype(np.float32)
    all_rotation_mags_gt_name = os.path.join(save_folder, 'all_gt_rot_degrees.csv')
    np.savetxt(all_rotation_mags_gt_name, all_rotation_mags_gt, delimiter=',', fmt='%1.5f')

    res_error = {
        "gt_angle": gt_distance / np.pi * 180,
        "rotation_geodesic_error_overlap_large": geodesic_loss_overlap_large,
        "rotation_geodesic_error_overlap_small": geodesic_loss_overlap_small,
        "rotation_geodesic_error_overlap_none": geodesic_loss_overlap_none,
        "rotation_geodesic_error": geodesic_loss,
    }
    return res_error

def eval_camera(predictions, save_folder):

    # convert pred & gt to quaternion
    pred, gt = np.copy(predictions['camera']['preds']['rot']), np.copy(predictions['camera']['gts']['rot'])

    r = R.from_quat(pred)
    r_pred = r.as_matrix()

    r = R.from_quat(gt)
    r_gt = r.as_matrix()

    res_error = evaluation_metric_rotation(torch.from_numpy(r_pred).cuda(), torch.from_numpy(r_gt).cuda(), save_folder)

    all_res = {}
    # mean, median, max, std, 10deg
    for k, v in res_error.items():
        v = v.view(-1).detach().cpu().numpy()
        if k == "gt_angle" or v.size == 0:
            continue
        mean = np.mean(v)
        median = np.median(v)
        error_max = np.max(v)
        std = np.std(v)
        count_10 = (v <= 10).sum(axis=0)
        percent_10 = np.true_divide(count_10, v.shape[0])
        all_res.update({k + '/mean': mean, k + '/median': median, k + '/max': error_max, k + '/std': std,
                        k + '/10deg': percent_10})
    print("Results:", all_res)
    return all_res

def compute_gt_rmat(rotation_x1, rotation_y1, rotation_x2, rotation_y2, batch_size):
    gt_mtx1 = compute_rotation_matrix_from_viewpoint(rotation_x1, rotation_y1, batch_size).view(batch_size, 3, 3)
    gt_mtx2 = compute_rotation_matrix_from_viewpoint(rotation_x2, rotation_y2, batch_size).view(batch_size, 3, 3)
    gt_rmat_matrix = compute_rotation_matrix_from_two_matrices(gt_mtx2, gt_mtx1).view(batch_size, 3, 3)
    return gt_rmat_matrix   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # data
    parser.add_argument("--datapath")
    parser.add_argument("--weights")
    parser.add_argument("--image_size", default=[384,512])
    parser.add_argument("--exp")
    parser.add_argument("--checkpoint_dir")
    parser.add_argument('--dataset', default='interiornet', choices=("interiornet", 'streetlearn'))
    parser.add_argument('--gamma', type=float, default=0.9)    
    parser.add_argument('--streetlearn_interiornet_type', default='', choices=('',"nooverlap","T",'nooverlapT'))

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

    if args.dataset == 'interiornet':
        if args.streetlearn_interiornet_type == 'T':
            from data_readers.interiornet import test_split_T, cur_path
            output_folder = 'interiornetT_test'
            dset = test_split_T
        else:
            from data_readers.interiornet import test_split, cur_path
            output_folder = 'interiornet_test'
            dset = test_split
    else:
        if args.streetlearn_interiornet_type == 'T':
            from data_readers.streetlearn import test_split_T, cur_path
            output_folder = 'streetlearnT_test'
            dset = test_split_T
            args.dataset = 'streetlearn_2016'
        else:
            from data_readers.streetlearn import test_split, cur_path
            output_folder = 'streetlearn_test'
            dset = test_split

    print('performing evaluation on %s set using model %s' % (output_folder, args.checkpoint_dir))

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

    sorted(dset.keys())

    for i, dset_i in tqdm(sorted(dset.items())):
        if args.dataset == 'interiornet':
            if i > 999:
                break
        base_pose = np.array([0,0,0,0,0,0,1])

        images = [cv2.imread(os.path.join(cur_path, 'data', args.dataset, dset[i]['img1']['path'])),
                    cv2.imread(os.path.join(cur_path, 'data', args.dataset, dset[i]['img2']['path']))]
        
        x1, y1 = dset[i]['img1']['x'], dset[i]['img1']['y']
        x2, y2 = dset[i]['img2']['x'], dset[i]['img2']['y']

        # compute rotation matrix
        gt_rmat = compute_gt_rmat(torch.tensor([[x1]]), torch.tensor([[y1]]), torch.tensor([[x2]]), torch.tensor([[y2]]), 1)

        # get quaternions from rotation matrix
        r = R.from_matrix(gt_rmat)
        rotation = r.as_quat()[0]
        
        rel_pose = np.concatenate([np.array([0,0,0]), rotation]) # translation is 0

        images = np.stack(images).astype(np.float32)
        images = torch.from_numpy(images).float()
        images = images.permute(0, 3, 1, 2)
        images = images.unsqueeze(0).cuda()

        intrinsics = np.stack([np.array([[128,128,128,128], [128,128,128,128]])]).astype(np.float32)
        intrinsics = torch.from_numpy(intrinsics).cuda()
        
        poses = np.vstack([base_pose, rel_pose]).astype(np.float32)
        poses = torch.from_numpy(poses).unsqueeze(0).cuda()
        Ps = SE3(poses)

        Gs = SE3.IdentityLike(Ps)

        # only 2 images so frame graph has no randomness
        N=2
        graph = OrderedDict()
        for ll in range(N):
            graph[ll] = [j for j in range(N) if ll!=j and abs(ll-j) <= 2]
                    
        with torch.no_grad():
            poses_est = model(images, Gs, intrinsics=intrinsics)
            geo_loss_tr, geo_loss_rot, rotation_mag, rotation_mag_gt, geo_metrics = losses.geodesic_loss(Ps, poses_est, \
                    graph, do_scale=False, train_val=train_val, gamma=args.gamma)
            preds = poses_est[0][0][1].data.cpu().numpy()

        predictions['camera']['gts']['tran'].append(np.array([0,0,0]))
        predictions['camera']['gts']['rot'].append(rotation)

        predictions['camera']['preds']['tran'].append(preds[:3])
        predictions['camera']['preds']['rot'].append(preds[3:])

        metrics['_geo_loss_tr'].append(geo_metrics['_geo_loss_tr'])
        metrics['_geo_loss_rot'].append(geo_metrics['_geo_loss_rot'])

    print('did this many:',len(predictions['camera']['gts']['rot']))

    print('mean geo tr', np.mean(np.array(metrics['_geo_loss_tr'])))
    print('mean geo rot', np.mean(np.array(metrics['_geo_loss_rot'])))

    full_output_folder = os.path.join('output', args.exp, output_folder, args.weights[:-4])
    camera_metrics = eval_camera(predictions, full_output_folder)
    

    with open(os.path.join(full_output_folder, 'results.txt'), 'w') as f:
        print('mean geo tr', np.mean(np.array(metrics['_geo_loss_tr'])), file=f)
        print('mean geo rot', np.mean(np.array(metrics['_geo_loss_rot'])), file=f)
        for k in camera_metrics:
            print(k, camera_metrics[k], file=f)
