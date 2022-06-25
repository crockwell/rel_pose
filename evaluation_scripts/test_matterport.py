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

    '''
    tran_logits = torch.stack(
        [p["camera"]["logits"]["tran"] for p in predictions]
    ).numpy()
    rot_logits = torch.stack(
        [p["camera"]["logits"]["rot"] for p in predictions]
    ).numpy()
    gt_tran_cls = torch.stack(
        [p["camera"]["gts"]["tran_cls"] for p in predictions]
    ).numpy()
    gt_rot_cls = torch.stack(
        [p["camera"]["gts"]["rot_cls"] for p in predictions]
    ).numpy()
    pred_tran = self.class2xyz(np.argmax(tran_logits, axis=1))
    pred_rot = self.class2quat(np.argmax(rot_logits, axis=1))
    '''
    pred_tran = np.vstack(predictions["camera"]["preds"]["tran"])#([p["camera"]["preds"]["tran"] for p in predictions])
    pred_rot = np.vstack(predictions["camera"]["preds"]["rot"])#([p["camera"]["preds"]["rot"] for p in predictions])

    gt_tran = np.vstack(predictions["camera"]["gts"]["tran"]) #([p["camera"]["gts"]["tran"] for p in predictions])
    gt_rot = np.vstack(predictions["camera"]["gts"]["rot"]) #([p["camera"]["gts"]["rot"] for p in predictions])

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
    parser.add_argument("--datapath", default="datasets/TartanAir")
    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=1000)
    parser.add_argument("--image_size", default=[384,512])
    parser.add_argument("--stereo", action="store_true")
    parser.add_argument("--disable_vis", action="store_true")
    parser.add_argument("--plot_curve", action="store_true")
    parser.add_argument("--id", type=int, default=-1)
    parser.add_argument("--exp", default="droidslam")
    parser.add_argument("--checkpoint_dir")
    parser.add_argument('--fusion_transformer', action="store_true", default=False)
    parser.add_argument('--post_conv', action="store_true", default=False)
    parser.add_argument('--post_conv_3D', action="store_true", default=False)
    parser.add_argument('--use_correlation_volume', action='store_true', default=False)
    parser.add_argument('--fc_activation', default='dropout')
    parser.add_argument('--use_full_transformer_output', action="store_true", default=False)
    parser.add_argument('--fc_hidden_size', type=int, default=512)
    parser.add_argument('--num_input_images', type=int, default=7)
    parser.add_argument('--feature_width', type=int, default=7)
    parser.add_argument('--n_frames', type=int, default=7)
    parser.add_argument('--normalized_coords', action="store_true", default=False)
    parser.add_argument('--normalize_quats', action="store_true", default=False)
    parser.add_argument('--prediction_pose_type', choices=("change", "absolute", 'classify'))
    parser.add_argument('--predict_normalized_pose', action="store_true", default=False)
    parser.add_argument('--normalize_in_dataloader', action="store_true", default=False)
    parser.add_argument('--eval_on_train_set', action="store_true", default=False)
    parser.add_argument('--eval_on_test_set', action="store_true", default=False)
    parser.add_argument('--eval_on_test_set_hard', action="store_true", default=False)
    parser.add_argument('--feature_resolution', type=int, default=7)
    parser.add_argument('--transformer_connectivity', default='cross_image', choices=("in_image","cross_image", "all", 'in_image_stacked', 'difference'))
    parser.add_argument('--cross_indices', nargs='+', type=int, help='indices for cross-attention, if using cross_image transformer connectivity')
    parser.add_argument('--fundamental_temp', type=float, default=1.0)
    parser.add_argument('--positional_encoding', nargs='+', type=int, help='indices for positional_encoding if using cross_image transformer connectivity')
    parser.add_argument('--outer_prod', nargs='+', type=int, help='indices for fundamental calc if using cross_image transformer connectivity')
    parser.add_argument('--pool_transformer_output', action="store_true", default=False)
    parser.add_argument('--use_amp', action="store_true", default=False)
    parser.add_argument('--weird_feats', action="store_true", default=False)
    parser.add_argument('--pool_size', type=int, default=12)
    parser.add_argument('--use_big_transformer', action="store_true", default=False)
    parser.add_argument('--use_droidslam_encoder', choices=("True", "False"))
    parser.add_argument('--transformer_depth', type=int, default=12)
    parser.add_argument('--use_medium_transformer', action="store_true", default=False)
    parser.add_argument('--pos_encoding_size', type=int, default=0, choices=(0,2,6,10,18,34,66))
    parser.add_argument('--no_pretrained_transformer', action='store_true')
    parser.add_argument('--use_procrustes', action='store_true')    
    parser.add_argument('--use_camera_encodings', action="store_true", default=False)
    parser.add_argument('--multiframe_predict_last_frame_only', action="store_true", default=False)    
    parser.add_argument('--gamma', type=float, default=0.9)    
    parser.add_argument('--squeeze_excite', nargs='+', type=int, help='indices for fundamental calc if using cross_image transformer connectivity')    
    parser.add_argument('--use_essential_units', action='store_true')
    parser.add_argument('--fund_resid', action='store_true')
    parser.add_argument('--squeeze_excite_big', action='store_true')
    parser.add_argument('--max_scale_aug', type=float, default=0.25)
    parser.add_argument('--cross_features', action='store_true')
    parser.add_argument('--epi_dist_scale', type=float, default=1.0)
    parser.add_argument('--epi_dist_sub', type=float, default=1.0)
    parser.add_argument('--first_head_only', action='store_true')
    parser.add_argument('--epipolar_loss_both_dirs', action='store_true')
    parser.add_argument('--attn_scale', type=float, default=1.0)
    parser.add_argument('--attn_shift', type=float, default=0.0)
    parser.add_argument('--use_sigmoid_attn', action='store_true')
    parser.add_argument('--use_vo_mlp', action='store_true')
    parser.add_argument('--use_test_bn', action='store_true')
    parser.add_argument('--pwc_big', action='store_true')
    parser.add_argument('--no_pretrained_resnet', action='store_true')
    parser.add_argument('--supervise_epi', action='store_true')  
    parser.add_argument('--use_single_softmax', action='store_true')  
    parser.add_argument('--use_hybrid_vit', action='store_true')
    parser.add_argument('--seperate_tf_qkv', action='store_true')
    parser.add_argument('--l1_pos_encoding', action='store_true')
    parser.add_argument('--use_medium_transformer_3head', action='store_true')

    parser.add_argument('--sparse_plane_baseline', action='store_true')     
    parser.add_argument('--cnn_attn_plus_feats', action='store_true')
    parser.add_argument('--use_pwc_encoder', action="store_true", default=False)
    parser.add_argument('--attn_one_way', action='store_true')
    
    parser.add_argument('--use_cnn_decoder', action='store_true')
    parser.add_argument('--use_positional_images', action='store_true')
    parser.add_argument('--cnn_decoder_use_essential', action='store_true')
    parser.add_argument('--cnn_decode_each_head', action='store_true')    
    parser.add_argument('--use_fixed_intrinsics', action='store_true')

    parser.add_argument('--optical_flow_input', default='', choices=('',"input","before_tf",'after_tf', "both"))
    parser.add_argument('--optical_flow_type', default='', choices=('',"ground_truth","tartanvo",'droid_slam'))
    parser.add_argument('--clustered_dim', type=int, default=0)
    parser.add_argument('--no_pos_encoding', action='store_true')
    parser.add_argument('--noess', action='store_true')

    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--filter_thresh", type=float, default=2.4)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--keyframe_thresh", type=float, default=3.5)
    parser.add_argument("--frontend_thresh", type=float, default=15)
    parser.add_argument("--frontend_window", type=int, default=20)
    parser.add_argument("--frontend_radius", type=int, default=1)
    parser.add_argument("--frontend_nms", type=int, default=1)

    parser.add_argument("--backend_thresh", type=float, default=20.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    args = parser.parse_args()
    torch.multiprocessing.set_start_method('spawn')

    from data_readers.matterport import val_split, test_split, cur_path#, train_split_for_eval

    output_folder = 'matterport_val'
    dset = val_split
    if args.eval_on_train_set: 
        output_folder = 'matterport_train'
        dset = train_split_for_eval
    elif args.eval_on_test_set_hard:
        dset = test_split
        output_folder = 'matterport_test'

    if args.use_droidslam_encoder == 'True':
        args.use_droidslam_encoder = True
    else:
        args.use_droidslam_encoder = False


    args.no_pretrained_transformer = True

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
    metrics = {'_geo_loss_tr': [], '_geo_loss_rot': [], '_class_loss_tr': [], '_class_loss_rot': []}

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
        Ps = SE3(poses)#.inv()

        Gs = SE3.IdentityLike(Ps)

        class_rot = kmeans_rots.predict([[og_rel_pose[3], og_rel_pose[4], og_rel_pose[5], og_rel_pose[6]]])
        class_rot = torch.from_numpy(class_rot).unsqueeze(0).cuda()

        class_tr = kmeans_trans.predict([[og_rel_pose[0], og_rel_pose[1], og_rel_pose[2]]])
        class_tr = torch.from_numpy(class_tr).unsqueeze(0).cuda()

        # only 2 images so frame graph has no randomness
        N=2
        graph = OrderedDict()
        for ll in range(N):
            graph[ll] = [j for j in range(N) if ll!=j and abs(ll-j) <= 2]
            
        intrinsics0 = intrinsics# / 8.0
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                if args.clustered_dim > 0:
                    clusters_est = model(images, Gs, intrinsics=intrinsics0)
                    poses_est = args.cluster_fcn.cluster_centers_(clusters_est) # just use argmax

                elif args.use_sigmoid_attn or args.supervise_epi:
                    poses_est, poses_est_mtx, attention_scores = model(images, Gs, intrinsics=intrinsics0)
                    if args.w_epi > 0:
                        epipolar_loss, epipolar_metrics = losses.epipolar_loss(Gs, attention_scores, train_val=train_val, 
                                epi_dist_scale=args.epi_dist_scale, epi_dist_sub=args.epi_dist_sub, 
                                first_head_only=args.first_head_only, epipolar_loss_both_dirs=args.epipolar_loss_both_dirs, JD=args.JD,
                                loss_on_each_head=args.loss_on_each_head)
                elif args.prediction_pose_type == 'classify':
                    poses_est, poses_est_mtx = model(images, Gs, intrinsics=intrinsics0)
                    geo_metrics, class_loss_tr, class_loss_rot = losses.cross_entropy_loss(class_rot, class_tr, poses_est, train_val='test')
                    trans_class = torch.argmax(poses_est[0][0,:32]).cpu().item()
                    rot_class = torch.argmax(poses_est[0][0,32:]).cpu().item()
                    preds = np.concatenate((kmeans_trans.cluster_centers_[trans_class], kmeans_rots.cluster_centers_[rot_class]))
                else:
                    poses_est, poses_est_mtx = model(images, Gs, intrinsics=intrinsics0)
                    geo_loss_tr, geo_loss_rot, rotation_mag, rotation_mag_gt, geo_metrics = losses.geodesic_loss(Ps, poses_est, \
                            graph, do_scale=False, train_val=train_val, gamma=args.gamma)

        predictions['camera']['gts']['tran'].append(dset['data'][i]['rel_pose']['position'])
        gt_rotation = dset['data'][i]['rel_pose']['rotation']
        if gt_rotation[0] < 0:
            gt_rotation[0] *= -1
            gt_rotation[1] *= -1
            gt_rotation[2] *= -1
            gt_rotation[3] *= -1
        predictions['camera']['gts']['rot'].append(gt_rotation)

        if not args.prediction_pose_type == 'classify':
            preds = poses_est[0][0][1].data.cpu().numpy() # .inv() preds = poses_est[0][0][1].inv().data.cpu().numpy() #     
            pr_copy = np.copy(preds)
            preds[3] = pr_copy[6] # swap 3 & 6, we used W last; want W first in quat
            preds[6] = pr_copy[3]
            preds[:3] = preds[:3] * DEPTH_SCALE

        predictions['camera']['preds']['tran'].append(preds[:3])
        predictions['camera']['preds']['rot'].append(preds[3:])

        if args.prediction_pose_type == 'classify':
            metrics['_class_loss_tr'].append(geo_metrics['test_class_loss_tr'])
            metrics['_class_loss_rot'].append(geo_metrics['test_class_loss_rot'])
        else:
            metrics['_geo_loss_tr'].append(geo_metrics['_geo_loss_tr'])
            metrics['_geo_loss_rot'].append(geo_metrics['_geo_loss_rot'])

    if args.prediction_pose_type == 'classify':
        print('mean class loss tr', np.mean(np.array(metrics['_class_loss_tr'])))
        print('mean class loss rot', np.mean(np.array(metrics['_class_loss_rot'])))
    else:
        print('mean geo tr', np.mean(np.array(metrics['_geo_loss_tr'])))
        print('mean geo rot', np.mean(np.array(metrics['_geo_loss_rot'])))

    camera_metrics = eval_camera(predictions)
    for k in camera_metrics:
        print(k, camera_metrics[k])
    
    with open(os.path.join('output', args.exp, output_folder, args.weights[:-4], 'results.txt'), 'w') as f:
        if args.prediction_pose_type == 'classify':
            print('mean class loss tr', np.mean(np.array(metrics['_class_loss_tr'])), file=f)
            print('mean class loss rot', np.mean(np.array(metrics['_class_loss_rot'])), file=f)
        else:
            print('mean geo tr', np.mean(np.array(metrics['_geo_loss_tr'])), file=f)
            print('mean geo rot', np.mean(np.array(metrics['_geo_loss_rot'])), file=f)
        for k in camera_metrics:
            print(k, camera_metrics[k], file=f)
        #print(camera_metrics, file=f)