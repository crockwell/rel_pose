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
    # rotaz = torch.zeros(batch, 1)

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


    '''
    acc_threshold = {
        "tran": 1.0,
        "rot": 30,
    }  # threshold for translation and rotation error to say prediction is correct.

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
    '''

def compute_gt_rmat(rotation_x1, rotation_y1, rotation_x2, rotation_y2, batch_size):
    gt_mtx1 = compute_rotation_matrix_from_viewpoint(rotation_x1, rotation_y1, batch_size).view(batch_size, 3, 3)
    gt_mtx2 = compute_rotation_matrix_from_viewpoint(rotation_x2, rotation_y2, batch_size).view(batch_size, 3, 3)
    gt_rmat_matrix = compute_rotation_matrix_from_two_matrices(gt_mtx2, gt_mtx1).view(batch_size, 3, 3)
    return gt_rmat_matrix

def compute_rotation_matrix_from_euler_angle(rotation_x, rotation_y, rotation_z=None, batch=None):
    rotax = rotation_x.view(batch, 1).type(torch.FloatTensor)
    rotay = rotation_y.view(batch, 1).type(torch.FloatTensor)
    if rotation_z is None:
        rotaz = torch.zeros(batch, 1)
    else:
        rotaz = rotation_z.view(batch, 1).type(torch.FloatTensor)

    c3 = torch.cos(rotax).view(batch, 1)  
    s3 = torch.sin(rotax).view(batch, 1)  
    c2 = torch.cos(rotay).view(batch, 1)  
    s2 = torch.sin(rotay).view(batch, 1)  
    c1 = torch.cos(rotaz).view(batch, 1)  
    s1 = torch.sin(rotaz).view(batch, 1) 

    row1 = torch.cat((c1 * c2, c1 * s2 * s3 - s1 * c3, c1 * s2 * c3 + s1 * s3), 1).view(-1, 1, 3)  # batch*1*3
    row2 = torch.cat((s1 * c2, s1 * s2 * s3 + c1 * c3, s1 * s2 * c3 - c1 * s3), 1).view(-1, 1, 3)  # batch*1*3
    row3 = torch.cat((-s2, c2 * s3, c2 * c3), 1).view(-1, 1, 3)  # batch*1*3

    matrix = torch.cat((row1, row2, row3), 1)  # batch*3*3

    return matrix
    
def compute_euler_angles_from_rotation_matrices(rotation_matrices):
    batch = rotation_matrices.shape[0]
    R = rotation_matrices
    sy = torch.sqrt(R[:, 0, 0] * R[:, 0, 0] + R[:, 1, 0] * R[:, 1, 0])
    singular = sy < 1e-6
    singular = singular.float()

    x = torch.atan2(R[:, 2, 1], R[:, 2, 2])
    y = torch.atan2(-R[:, 2, 0], sy)
    z = torch.atan2(R[:, 1, 0], R[:, 0, 0])

    xs = torch.atan2(-R[:, 1, 2], R[:, 1, 1])
    ys = torch.atan2(-R[:, 2, 0], sy)
    zs = R[:, 1, 0] * 0

    rotation_x = x * (1 - singular) + xs * singular
    rotation_y = y * (1 - singular) + ys * singular
    rotation_z = z * (1 - singular) + zs * singular

    return rotation_x, rotation_y, rotation_z
    

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
    parser.add_argument('--dataset', default='tartan', choices=("tartan", "matterport", "interiornet", 'streetlearn'))
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
    parser.add_argument('--viz', action='store_true')

    parser.add_argument('--sparse_plane_baseline', action='store_true')     
    parser.add_argument('--cnn_attn_plus_feats', action='store_true')
    parser.add_argument('--use_pwc_encoder', action="store_true", default=False)
    parser.add_argument('--attn_one_way', action='store_true')
    parser.add_argument('--streetlearn_interiornet_type', default='', choices=('',"nooverlap","T",'nooverlapT'))

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
            from data_readers.streetlearn import test_split_T, cur_path#, train_split_for_eval
            output_folder = 'streetlearnT_test'
            dset = test_split_T
            args.dataset = 'streetlearn_2016'
        else:
            from data_readers.streetlearn import test_split, cur_path#, train_split_for_eval
            output_folder = 'streetlearn_test'
            dset = test_split

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
    metrics = {'_geo_loss_tr': [], '_geo_loss_rot': [], 
                '_class_loss_x': [], '_class_loss_y': [],
                '_class_loss_z': []}

    sorted(dset.keys())

    for i, dset_i in tqdm(sorted(dset.items())):
        if args.dataset == 'interiornet': #if args.viz:
            if i > 999:
                continue
        base_pose = np.array([0,0,0,0,0,0,1])

        images = [cv2.imread(os.path.join(cur_path, 'data', args.dataset, dset[i]['img1']['path'])),
                    cv2.imread(os.path.join(cur_path, 'data', args.dataset, dset[i]['img2']['path']))]
        
        x1, y1 = dset[i]['img1']['x'], dset[i]['img1']['y']
        x2, y2 = dset[i]['img2']['x'], dset[i]['img2']['y']

        # compute rotation matrix
        gt_rmat = compute_gt_rmat(torch.tensor([[x1]]), torch.tensor([[y1]]), torch.tensor([[x2]]), torch.tensor([[y2]]), 1)

        angle_x, angle_y, angle_z = compute_euler_angles_from_rotation_matrices(gt_rmat)
        angles = np.array([angle_x.item(), angle_y.item(), angle_z.item()])
        angles = np.stack(angles).astype(np.float32)
        angles = torch.from_numpy(angles).unsqueeze(0).cuda()

        # get quaternions from rotation matrix
        r = R.from_matrix(gt_rmat)
        rotation = r.as_quat()[0]
        
        rel_pose = np.concatenate([np.array([0,0,0]), rotation]) # translation is 0
        '''
        try:
            
        except:
            import pdb; pdb.set_trace()
            print('no img index',i)
        continue
        '''
        #import pdb; pdb.set_trace()
        images = np.stack(images).astype(np.float32)
        images = torch.from_numpy(images).float()
        images = images.permute(0, 3, 1, 2)
        images = images.unsqueeze(0).cuda()

        intrinsics = np.stack([np.array([[128,128,128,128], [128,128,128,128]])]).astype(np.float32)
        intrinsics = torch.from_numpy(intrinsics).cuda()
        
        poses = np.vstack([base_pose, rel_pose]).astype(np.float32)
        poses = torch.from_numpy(poses).unsqueeze(0).cuda()
        Ps = SE3(poses)#.inv()

        Gs = SE3.IdentityLike(Ps)

        # only 2 images so frame graph has no randomness
        N=2
        graph = OrderedDict()
        for ll in range(N):
            graph[ll] = [j for j in range(N) if ll!=j and abs(ll-j) <= 2]
            
        intrinsics0 = intrinsics# / 8.0
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                if args.use_sigmoid_attn or args.supervise_epi:
                    poses_est, poses_est_mtx, attention_scores = model(images, Gs, intrinsics=intrinsics0)
                    if args.w_epi > 0:
                        epipolar_loss, epipolar_metrics = losses.epipolar_loss(Gs, attention_scores, train_val=train_val, 
                                epi_dist_scale=args.epi_dist_scale, epi_dist_sub=args.epi_dist_sub, 
                                first_head_only=args.first_head_only, epipolar_loss_both_dirs=args.epipolar_loss_both_dirs, JD=args.JD,
                                loss_on_each_head=args.loss_on_each_head)
                elif args.prediction_pose_type == 'classify':
                    poses_est, poses_est_mtx = model(images, Gs, intrinsics=intrinsics0)
                    geo_metrics, class_loss_x, class_loss_y, class_loss_z = losses.cross_entropy_loss_streetlearn_interiornet(angles, poses_est, train_val='test')                  
                    
                    pred_x = torch.argmax(poses_est[0][0,:360]).cpu().item()-180
                    pred_y = torch.argmax(poses_est[0][0,360:720]).cpu().item()-180
                    pred_z = torch.argmax(poses_est[0][0,720:]).cpu().item()-180
                    pred_rotation = compute_rotation_matrix_from_euler_angle(torch.tensor([pred_x]), torch.tensor([pred_y]), torch.tensor([pred_z]), 1)

                    r = R.from_matrix(pred_rotation)
                    rotation_pred = r.as_quat()[0]
                    import pdb; pdb.set_trace()

                    preds = np.concatenate([np.array([0,0,0]), rotation_pred])
                else:
                    poses_est, poses_est_mtx = model(images, Gs, intrinsics=intrinsics0)
                    geo_loss_tr, geo_loss_rot, rotation_mag, rotation_mag_gt, geo_metrics = losses.geodesic_loss(Ps, poses_est, \
                            graph, do_scale=False, train_val=train_val, gamma=args.gamma)
                    preds = poses_est[0][0][1].data.cpu().numpy()
                    #print(poses_est[0].data[0,1,3:].cpu())
                    #print(Ps.data[0,1,3:].cpu())
                    #print('-------------')

        predictions['camera']['gts']['tran'].append(np.array([0,0,0]))
        predictions['camera']['gts']['rot'].append(rotation)

        predictions['camera']['preds']['tran'].append(preds[:3])
        predictions['camera']['preds']['rot'].append(preds[3:])

        print(preds[3:])

        if args.prediction_pose_type == 'classify':
            metrics['_class_loss_x'].append(geo_metrics['test_class_loss_x'])
            metrics['_class_loss_y'].append(geo_metrics['test_class_loss_y'])
            metrics['_class_loss_z'].append(geo_metrics['test_class_loss_z'])
        else:
            metrics['_geo_loss_tr'].append(geo_metrics['_geo_loss_tr'])
            metrics['_geo_loss_rot'].append(geo_metrics['_geo_loss_rot'])

    print('did this many:',len(predictions['camera']['gts']['rot']))

    if args.prediction_pose_type == 'classify':
        print('mean class loss x', np.mean(np.array(metrics['_class_loss_x'])))
        print('mean class loss y', np.mean(np.array(metrics['_class_loss_y'])))
        print('mean class loss z', np.mean(np.array(metrics['_class_loss_z'])))
    else:
        print('mean geo tr', np.mean(np.array(metrics['_geo_loss_tr'])))
        print('mean geo rot', np.mean(np.array(metrics['_geo_loss_rot'])))

    full_output_folder = os.path.join('output', args.exp, output_folder, args.weights[:-4])
    camera_metrics = eval_camera(predictions, full_output_folder)
    

    with open(os.path.join(full_output_folder, 'results.txt'), 'w') as f:
        if args.prediction_pose_type == 'classify':
            print('mean class loss x', np.mean(np.array(metrics['_class_loss_x'])), file=f)
            print('mean class loss y', np.mean(np.array(metrics['_class_loss_y'])), file=f)
            print('mean class loss z', np.mean(np.array(metrics['_class_loss_z'])), file=f)
        else:
            print('mean geo tr', np.mean(np.array(metrics['_geo_loss_tr'])), file=f)
            print('mean geo rot', np.mean(np.array(metrics['_geo_loss_rot'])), file=f)
        for k in camera_metrics:
            print(k, camera_metrics[k], file=f)
        #print(camera_metrics, file=f)