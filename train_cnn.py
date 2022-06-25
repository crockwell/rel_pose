import cv2
import numpy as np
from collections import OrderedDict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data_readers.factory import dataset_factory

import lietorch
from lietorch import SO3, SE3, Sim3
from geom import losses, projective_ops
from geom.losses import geodesic_loss, fixed_geodesic_loss

import torch 
import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

# network
from model import ViTEss
from logger import Logger

# DDP training
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm
import random
from datetime import datetime

import os
#os.environ["NCCL_BLOCKING_WAIT"] = "1"
#os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
#os.environ['NCCL_DEBUG'] = 'INFO'
#torch.autograd.set_detect_anomaly(True)

#torch.backends.cudnn.benchmark = True
#torch.backends.cudnn.deterministic = False
import warnings
warnings.filterwarnings("ignore")

from os import listdir
from os.path import isfile, join

def setup_ddp(gpu, args):
    dist.init_process_group(                                   
    	backend='nccl',      # backend='gloo',#                                    
   		init_method='env://',     
    	world_size=args.world_size,                              
    	rank=gpu)

    torch.manual_seed(0)
    torch.cuda.set_device(gpu)

def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey()

def lietorch_numpy_mult(a, b):
    a = lietorch.SE3(torch.from_numpy(a).float())
    b = lietorch.SE3(torch.from_numpy(b).float())
    return (a * b).data.numpy()

def train(gpu, args):
    """ Test to make sure project transform correctly maps points """

    # coordinate multiple GPUs
    if not args.no_ddp:
        setup_ddp(gpu, args)
    rng = np.random.default_rng(12345)
    random.seed(0)

    thiscuda = 'cuda:%d' % gpu
    map_location = {'cuda:%d' % 0: thiscuda}
    args.map_location = map_location
    if args.no_ddp: 
        args.map_location = ''
        thiscuda = 'cuda:0'

    model = ViTEss(args)

    model.to(thiscuda)
    model.train()
        
    if not args.use_droidslam_encoder and not args.use_pwc_encoder and args.feature_resolution > 7:
        for param in model.resnet.layer4.parameters():
            param.requires_grad = False

        if args.feature_resolution > 14:
            for param in model.resnet.layer3.parameters():
                param.requires_grad = False

            if args.feature_resolution > 28:
                for param in model.resnet.layer2.parameters():
                    param.requires_grad = False

    if not args.no_ddp:
        model = DDP(model, device_ids=[gpu], find_unused_parameters=False)#True)

    if args.weight_decay > 1e-5:
        decay = []
        no_decay = []
        for name, param in model.named_parameters():
            print('checking {}'.format(name))
            if 'fusion_transformer' in name:
                decay.append(param)
            else:
                no_decay.append(param)
        optimizer = torch.optim.Adam([{'params': no_decay, 'weight_decay': 0}, {'params': decay}], lr=args.lr, weight_decay=args.weight_decay)        
    else:
        # fetch optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if args.warmup > 0:
        pct_warmup = args.warmup / args.steps
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
            args.lr, args.steps, pct_start=pct_warmup, div_factor=25, cycle_momentum=False)
    else:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
            args.lr, args.steps, pct_start=0.01, cycle_momentum=False)

    if args.ckpt is not None:
        print('loading separate checkpoint')

        if args.no_ddp:
            existing_ckpt = torch.load(args.ckpt)
        else:
            existing_ckpt = torch.load(args.ckpt, map_location=map_location)

        model.load_state_dict(existing_ckpt['model'], strict=False)
        optimizer.load_state_dict(existing_ckpt['optimizer'])
        #if 'scheduler' in existing_ckpt:
        #    scheduler.load_state_dict(existing_ckpt['scheduler'])
        del existing_ckpt
    elif args.existing_ckpt is not None:
        if args.no_ddp:
            existing_ckpt = torch.load(args.existing_ckpt)
            state_dict = OrderedDict([
                (k.replace("module.", ""), v) for (k, v) in existing_ckpt['model'].items()])
            model.load_state_dict(state_dict)#existing_ckpt)
            del state_dict
            optimizer.load_state_dict(existing_ckpt['optimizer'])
            if 'scheduler' in existing_ckpt:
                scheduler.load_state_dict(existing_ckpt['scheduler'])
        else:
            existing_ckpt = torch.load(args.existing_ckpt, map_location=map_location)
            model.load_state_dict(existing_ckpt['model'])
            optimizer.load_state_dict(existing_ckpt['optimizer'])
            if 'scheduler' in existing_ckpt:
                scheduler.load_state_dict(existing_ckpt['scheduler'])
        print('loading existing checkpoint')
        del existing_ckpt

    logger = Logger(args.name, scheduler)
    should_keep_training = True

    torch.autograd.set_detect_anomaly(True)

    if 'get_dataset' in args and args.get_dataset:
        subepoch = None 
    else:
        subepoch = 0

    subepoch = 10-args.dset_size_tenths
    print('using',args.dset_size_tenths,'tenths of the dataset')
    train_steps = scheduler.state_dict()['last_epoch']
    epoch_count = train_steps // (300000 / (args.gpus * args.batch))
    print(epoch_count, train_steps)
    while should_keep_training:
        is_training = True
        train_val = 'train'
        if subepoch == 10:
            """
            validate!
            """
            is_training = False
            train_val = 'val'
        
        db = dataset_factory([args.dataset], datapath=args.datapath, \
                subepoch=subepoch, scale_aug=(not args.no_scale_aug), \
                is_training=is_training, gpu=gpu, load_img_tensors=args.load_img_tensors,
                max_scale_aug=args.max_scale_aug, do_aug=(not args.no_aug), no_depth=args.no_depth, use_fixed_intrinsics=args.use_fixed_intrinsics, 
                blackwhite=args.blackwhite, 
                blackwhite_pt5=args.blackwhite_pt5, 
                streetlearn_interiornet_type=args.streetlearn_interiornet_type, use_mini_dataset=args.use_mini_dataset)
        if not args.no_ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                db, shuffle=is_training, num_replicas=args.world_size, rank=gpu)
            train_loader = DataLoader(db, batch_size=args.batch, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True) #  generator=g
        else:
            train_loader = DataLoader(db, batch_size=args.batch, num_workers=1,shuffle=False)#1)
        
        model.train()
        if not args.use_droidslam_encoder and not args.use_pwc_encoder and args.use_test_bn:
            if args.no_ddp:
                model.resnet.train(False)
            else:
                model.module.resnet.train(False)
            print('using test batchnorm for encoder')

        if not is_training:
            model.eval()

        with tqdm(train_loader, unit="batch") as tepoch:
            for i_batch, item in enumerate(tepoch):
                optimizer.zero_grad()

                if args.dataset == 'matterport':
                    images, poses, intrinsics, class_rot, class_tr = [x.to('cuda') for x in item]
                elif 'streetlearn' in args.dataset or 'interiornet' in args.dataset:
                    images, poses, intrinsics, angles = [x.to('cuda') for x in item]
                else:
                    images, poses, intrinsics = [x.to('cuda') for x in item]
                
                if args.dataset == 'matterport' or 'streetlearn' in args.dataset or 'interiornet' in args.dataset:
                    Ps = SE3(poses)
                else:
                    # convert poses w2c -> c2w
                    Ps = SE3(poses).inv()

                    if args.normalize_in_dataloader:
                        transformation = Ps[:,:1].inv()
                        www = [transformation] * 2
                        inverse_Ps = lietorch.cat(www, 1)
                        transformed_Ps = Ps * inverse_Ps 
                        Ps = transformed_Ps

                Gs = SE3.IdentityLike(Ps)

                Ps_out = SE3(Ps.data.clone())

                if args.clustered_dim > 0:
                    import pdb; pdb.set_trace()
                    clusters = args.cluster_fcn.predict(Ps.cpu().numpy()[:,3:])
                    clusters = torch.from_numpy(clusters).cuda()
                    #val_preds = args.cluster_fcn.labels_                  

                # only 2 images so frame graph has no randomness
                graph = OrderedDict()
                for i in range(N):
                    graph[i] = [j for j in range(N) if i!=j and abs(i-j) <= 2]

                r = 0
                while r < (1-args.restart_prob):
                    r = rng.random()

                    if not is_training:
                        r = 1
                    
                    intrinsics0 = intrinsics# / 8.0
                    
                    metrics = {}

                    if not is_training:
                        with torch.no_grad():
                            poses_est, poses_est_mtx = model(images, Gs, intrinsics=intrinsics0)
                            
                            if args.use_fixed_geodesic:
                                geo_loss_tr, geo_loss_rot, rotation_mag, geo_metrics = losses.fixed_geodesic_loss(Ps_out, poses_est, train_val=train_val)
                            elif args.prediction_pose_type == 'classify':
                                pass
                            else:
                                geo_loss_tr, geo_loss_rot, rotation_mag, rotation_mag_gt, geo_metrics = losses.geodesic_loss(Ps_out, poses_est, \
                                        graph, do_scale=False, train_val=train_val, gamma=args.gamma)

                            if args.w_rot_l2 > 0 or args.w_tr_l2 > 0:
                                geo_loss_tr_l2, geo_loss_rot_l2, geo_metrics_l2 = losses.geodesic_loss_l2(Ps_out, poses_est, \
                                        graph, do_scale=False, train_val=train_val, gamma=args.gamma)
                                metrics.update(geo_metrics_l2)
                    else:
                        poses_est, poses_est_mtx = model(images, Gs, intrinsics=intrinsics0)
                        if args.use_fixed_geodesic:
                            geo_loss_tr, geo_loss_rot, rotation_mag, geo_metrics = losses.fixed_geodesic_loss(Ps_out, poses_est, train_val=train_val)
                            pass
                        else:
                            geo_loss_tr, geo_loss_rot, rotation_mag, rotation_mag_gt, geo_metrics = losses.geodesic_loss(Ps_out, poses_est, 
                                        graph, do_scale=False, train_val=train_val, gamma=args.gamma)

                        if args.w_rot_l2 > 0 or args.w_tr_l2 > 0:
                            geo_loss_tr_l2, geo_loss_rot_l2, geo_metrics_l2 = losses.geodesic_loss_l2(Ps_out, poses_est, \
                                    graph, do_scale=False, train_val=train_val, gamma=args.gamma)
                            metrics.update(geo_metrics_l2)
                    
                    metrics.update(geo_metrics)

                    if is_training:
                        loss = args.w_tr * geo_loss_tr + args.w_rot * geo_loss_rot

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip) # if total_norm.isnan() or total_norm.isinf():
                        optimizer.step()
                        Gs = poses_est[-1].detach()
                        
                        scheduler.step() 
                        train_steps += 1
                                       

                    if gpu == 0 or args.no_ddp:
                        logger.push(metrics)

                        if i_batch % 20 == 0 and args.prediction_pose_type != 'classify':
                            torch.set_printoptions(sci_mode=False, linewidth=150)
                            for jjj in range(len(poses_est)):
                                print('pred number:', jjj, 'gamma', args.gamma ** (len(poses_est) - jjj - 1))
                                print('\n estimated pose')
                                print(poses_est[jjj].data[0,:7,:].cpu().detach())
                                print('ground truth pose')
                                print(Ps_out.data[0,:7,:].cpu().detach())
                                print('diff')
                                print(poses_est[jjj].data[0,:7,:].cpu().detach() - Ps_out.data[0,:7,:].cpu().detach())
                                print('')
                        if (i_batch + 10) % 20 == 0:
                            print('\n metrics:', metrics, '\n')
                        if i_batch % 100 == 0:
                            print('epoch', str(epoch_count))
                            print('subepoch: ', str(subepoch))
                            print('using', train_val, 'set')

                    if train_steps % 10000 == 0 and (gpu == 0 or args.no_ddp) and is_training:
                        PATH = 'output/%s/checkpoints/%06d.pth' % (args.name, train_steps)
                        checkpoint = {"model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict()}
                        torch.save(checkpoint, PATH)

                    if train_steps >= args.steps:
                        PATH = 'output/%s/checkpoints/%06d.pth' % (args.name, train_steps)
                        checkpoint = {"model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict()}
                        torch.save(checkpoint, PATH)
                        should_keep_training = False
                        break

                #dist.barrier()
       
        subepoch = (subepoch + 1)
        if subepoch == 11:
            subepoch = 10-args.dset_size_tenths
            epoch_count += 1

    dist.destroy_process_group()
        
                

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='bla', help='name your experiment')
    parser.add_argument('--ckpt', help='checkpoint to restore')
    parser.add_argument('--datapath', help="path to dataset directory")
    parser.add_argument('--gpus', type=int, default=4)
    parser.add_argument('--no_ddp', action="store_true", default=False)
    parser.add_argument('--fusion_transformer', action="store_true", default=False)
    parser.add_argument('--pool_transformer_output', action="store_true", default=False)
    parser.add_argument('--fc_hidden_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--normalize_in_dataloader', action="store_true", default=False)
    parser.add_argument('--no_scale_aug', action="store_true", default=False)
    parser.add_argument('--no_aug', action="store_true", default=False)
    parser.add_argument('--feature_resolution', type=int, default=7)
    parser.add_argument('--transformer_connectivity', default='cross_image', choices=("in_image","cross_image", "all", 'in_image_stacked', 'difference'))
    parser.add_argument('--dataset', default='matterport', choices=("matterport", "interiornet", 'streetlearn'))
    parser.add_argument('--cross_indices', nargs='+', type=int, help='indices for cross-attention, if using cross_image transformer connectivity')
    parser.add_argument('--positional_encoding', nargs='+', type=int, help='indices for positional_encoding if using cross_image transformer connectivity')
    parser.add_argument('--outer_prod', nargs='+', type=int, help='indices for fundamental calc if using cross_image transformer connectivity')    
    parser.add_argument('--weird_feats', action="store_true", default=False)
    parser.add_argument('--use_small_cross', action="store_true", default=False)
    parser.add_argument('--pool_size', type=int, default=12)
    parser.add_argument('--use_medium_transformer', action="store_true", default=False)
    parser.add_argument('--transformer_depth', type=int, default=12)
    parser.add_argument('--fundamental_temp', type=float, default=1.0)
    parser.add_argument('--max_scale_aug', type=float, default=0.25)
    parser.add_argument('--no_pretrained_transformer', action='store_true')
    parser.add_argument('--load_img_tensors', action='store_true')
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--use_essential_units', action='store_true')
    parser.add_argument('--fund_resid', action='store_true')
    parser.add_argument('--cross_features', action='store_true')
    parser.add_argument('--attn_scale', type=float, default=1.0)
    parser.add_argument('--attn_shift', type=float, default=0.0)
    parser.add_argument('--first_head_only', action='store_true')
    parser.add_argument('--epipolar_loss_both_dirs', action='store_true')
    parser.add_argument('--use_sigmoid_attn', action='store_true')
    parser.add_argument('--use_vo_mlp', action='store_true')
    parser.add_argument('--use_test_bn', action='store_true')
    parser.add_argument('--JD', action='store_true')
    parser.add_argument('--loss_on_each_head', action='store_true')
    parser.add_argument('--no_pretrained_resnet', action='store_true')
    parser.add_argument('--attn_one_way', action='store_true')
    parser.add_argument('--supervise_epi', action='store_true')  
    parser.add_argument('--use_single_softmax', action='store_true')  
    parser.add_argument('--cnn_attn_plus_feats', action='store_true')
    parser.add_argument('--no_depth', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--get_dataset', action='store_true')
    parser.add_argument('--seperate_tf_qkv', action='store_true')
    parser.add_argument('--use_fixed_intrinsics', action='store_true')
    parser.add_argument('--blackwhite', action='store_true')
    parser.add_argument('--blackwhite_pt5', action='store_true')
    parser.add_argument('--clustered_dim', type=int, default=0)
    parser.add_argument('--dset_size_tenths', type=int, default=10)
    parser.add_argument('--use_medium_transformer_3head', action='store_true')
    parser.add_argument('--streetlearn_interiornet_type', default='', choices=('',"nooverlap","T",'nooverlapT'))

    parser.add_argument('--use_cnn_decoder', action='store_true')
    parser.add_argument('--use_positional_images', action='store_true')
    parser.add_argument('--cnn_decoder_use_essential', action='store_true')
    parser.add_argument('--no_pos_encoding', action='store_true')
    parser.add_argument('--noess', action='store_true')

    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--iters', type=int, default=15)
    parser.add_argument('--steps', type=int, default=200001)
    parser.add_argument('--lr', type=float, default=0.00025)
    parser.add_argument('--clip', type=float, default=2.5)
    parser.add_argument('--normalize_quats', action="store_true", default=False)
    parser.add_argument('--prediction_pose_type', choices=("change", "absolute", 'classify'))
    parser.add_argument('--use_droidslam_encoder', action="store_true", default=False)
    parser.add_argument('--use_pwc_encoder', action="store_true", default=False)
    parser.add_argument('--use_fixed_geodesic', action="store_true", default=False)
    parser.add_argument('--cnn_decode_each_head', action='store_true')  
    parser.add_argument('--l1_pos_encoding', action='store_true')
    parser.add_argument('--use_mini_dataset', action='store_true')

    parser.add_argument('--w_tr', type=float, default=0.0)
    parser.add_argument('--w_rot', type=float, default=0.0)
    parser.add_argument('--w_rot_l2', type=float, default=0.0)
    parser.add_argument('--w_tr_l2', type=float, default=0.0)
    parser.add_argument('--w_epi', type=float, default=0.0)
    parser.add_argument('--w_epi_min', type=float, default=0.0)
    parser.add_argument('--w_epi_max', type=float, default=0.0)
    parser.add_argument('--w2', type=float, default=0.01)
    parser.add_argument('--w3', type=float, default=0.05)
    parser.add_argument('--w_cl_tr', type=float, default=0.0)
    parser.add_argument('--w_cl_rot', type=float, default=0.0)
    parser.add_argument('--warmup', type=int, default=0)

    parser.add_argument('--noise', action='store_true')
    parser.add_argument('--scale', action='store_true')
    parser.add_argument('--edges', type=int, default=24)
    parser.add_argument('--restart_prob', type=float, default=0.8)

    args = parser.parse_args()
    
    print(args)

    import os
    PATHS = ['output/%s/checkpoints' % (args.name), 'output/%s/runs' % (args.name), 'output/%s/train_output/images' % (args.name)]
    args.existing_ckpt = None
    #if not os.path.isdir(PATH):
    for PATH in PATHS:
        try:
            os.makedirs(PATH)
        except:
            if 'checkpoints' in PATH:
                ckpts = listdir(PATH)

                if len(ckpts) > 0:
                    if 'most_recent_ckpt.pth' in ckpts:
                        existing_ckpt = 'most_recent_ckpt.pth'
                    else:
                        ckpts = [int(i[:-4]) for i in ckpts]
                        ckpts.sort()
                        existing_ckpt = str(ckpts[-1]).zfill(6) +'.pth'
                
                    args.existing_ckpt = os.path.join(PATH, existing_ckpt)
                    print('existing',args.existing_ckpt)
            pass

    
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M")

    with open('output/%s/args_%s.txt' % (args.name, dt_string), 'w') as f:
        for k, v in vars(args).items():
            f.write(str(k) + '  '+ str(v) + '\n')
        #f.write(json.dumps(vars(args)))

    args.cluster_fcn = None
    if args.clustered_dim > 0:
        from coclust.clustering import SphericalKmeans
        cluster_input = np.load('validation_poses.npy')[:,3:]
        args.cluster_fcn = SphericalKmeans(n_clusters=32)
        args.cluster_fcn.fit(cluster_input)
        

    if args.no_ddp:
        train(args.gpus, args)
    else:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12356'
        args.world_size = args.gpus
        mp.spawn(train, nprocs=args.gpus, args=(args,))
    
