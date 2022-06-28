import numpy as np
from collections import OrderedDict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data_readers.factory import dataset_factory

import lietorch
from lietorch import SE3
from geom.losses import geodesic_loss, fixed_geodesic_loss

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

def setup_ddp(gpu, args):
    dist.init_process_group(                                   
    	backend='nccl',                                 
   		init_method='env://',     
    	world_size=args.world_size,                              
    	rank=gpu)

    torch.manual_seed(0)
    torch.cuda.set_device(gpu)

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
        
    # unused layers
    for param in model.resnet.layer4.parameters():
        param.requires_grad = False

    for param in model.resnet.layer3.parameters():
        param.requires_grad = False

    if not args.no_ddp:
        model = DDP(model, device_ids=[gpu], find_unused_parameters=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    pct_warmup = args.warmup / args.steps
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
        args.lr, args.steps, pct_start=pct_warmup, div_factor=25, cycle_momentum=False)

    if args.ckpt is not None:
        print('loading separate checkpoint')

        if args.no_ddp:
            existing_ckpt = torch.load(args.ckpt)
        else:
            existing_ckpt = torch.load(args.ckpt, map_location=map_location)

        model.load_state_dict(existing_ckpt['model'], strict=False)
        optimizer.load_state_dict(existing_ckpt['optimizer'])

        del existing_ckpt
    elif args.existing_ckpt is not None:
        if args.no_ddp:
            existing_ckpt = torch.load(args.existing_ckpt)
            state_dict = OrderedDict([
                (k.replace("module.", ""), v) for (k, v) in existing_ckpt['model'].items()])
            model.load_state_dict(state_dict)
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

    subepoch = 0

    train_steps = 0
    epoch_count = 0
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
                subepoch=subepoch,  \
                is_training=is_training, gpu=gpu, 
                streetlearn_interiornet_type=args.streetlearn_interiornet_type, use_mini_dataset=args.use_mini_dataset)
        if not args.no_ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                db, shuffle=is_training, num_replicas=args.world_size, rank=gpu)
            train_loader = DataLoader(db, batch_size=args.batch, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
        else:
            train_loader = DataLoader(db, batch_size=args.batch, num_workers=1,shuffle=False)
        
        model.train()

        if not is_training:
            model.eval()

        with tqdm(train_loader, unit="batch") as tepoch:
            for i_batch, item in enumerate(tepoch):
                optimizer.zero_grad()

                images, poses, intrinsics = [x.to('cuda') for x in item]
                Ps = SE3(poses)
                Gs = SE3.IdentityLike(Ps)
                Ps_out = SE3(Ps.data.clone())               

                graph = OrderedDict()
                for i in range(N):
                    graph[i] = [j for j in range(N) if i!=j and abs(i-j) <= 2]
                                    
                metrics = {}

                if not is_training:
                    with torch.no_grad():
                        poses_est, poses_est_mtx = model(images, Gs, intrinsics=intrinsics)
                        
                        if args.use_fixed_geodesic:
                            geo_loss_tr, geo_loss_rot, rotation_mag, geo_metrics = fixed_geodesic_loss(Ps_out, poses_est, train_val=train_val)
                        else:
                            geo_loss_tr, geo_loss_rot, rotation_mag, rotation_mag_gt, geo_metrics = geodesic_loss(Ps_out, poses_est, \
                                    graph, do_scale=False, train_val=train_val, gamma=args.gamma)
                else:
                    poses_est, poses_est_mtx = model(images, Gs, intrinsics=intrinsics)
                    if args.use_fixed_geodesic:
                        geo_loss_tr, geo_loss_rot, rotation_mag, geo_metrics = fixed_geodesic_loss(Ps_out, poses_est, train_val=train_val)
                    else:
                        geo_loss_tr, geo_loss_rot, rotation_mag, rotation_mag_gt, geo_metrics = geodesic_loss(Ps_out, poses_est, 
                                    graph, do_scale=False, train_val=train_val, gamma=args.gamma)

                    loss = args.w_tr * geo_loss_tr + args.w_rot * geo_loss_rot

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                    optimizer.step()
                    Gs = poses_est[-1].detach()
                    
                    scheduler.step() 
                    train_steps += 1
                
                metrics.update(geo_metrics)                                    

                if gpu == 0 or args.no_ddp:
                    logger.push(metrics)

                    if i_batch % 20 == 0:
                        torch.set_printoptions(sci_mode=False, linewidth=150)
                        for local_index in range(len(poses_est)):
                            print('pred number:', local_index, 'gamma', args.gamma ** (len(poses_est) - local_index - 1))
                            print('\n estimated pose')
                            print(poses_est[local_index].data[0,:7,:].cpu().detach())
                            print('ground truth pose')
                            print(Ps_out.data[0,:7,:].cpu().detach())
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
       
        subepoch = (subepoch + 1)
        if subepoch == 11:
            subepoch = 0
            epoch_count += 1

    dist.destroy_process_group()

if __name__ == '__main__':
    # TODO: get rid of gamma
    # make sure our 1k on streetlearn are correct 1k. 
    # turn off validation for interniornet & streetlearn
    # use_fixed_intrinsics inconsistency on some experiments? get rid of it.
    # get rid of absolute paths
    # 90fov is 128 focal right?
    # change impl to not predict dead first pose, only second?
    # clean up fixed_geodesic
    # clean up "for i in range(N): graph[i] = [j for j in range(N) if i!=j and abs(i-j) <= 2]"
    # clean for path in paths shit
    # whats going on with streetlearn in augmentation?
    # we forgot about cross_features ablation
    # cite `How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    # and `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877 for code
    # on matterport, we scale depths to balance rot & trans loss
    # DEPTH_SCALE = 5.0
    # careful on reshape size

    # debugging:
    #os.environ["NCCL_BLOCKING_WAIT"] = "1"
    #os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    #os.environ['NCCL_DEBUG'] = 'INFO'
    #torch.autograd.set_detect_anomaly(True)

    #torch.backends.cudnn.benchmark = True
    #torch.backends.cudnn.deterministic = False
    #import warnings
    #warnings.filterwarnings("ignore")
    #    torch.autograd.set_detect_anomaly(True)

    import argparse
    parser = argparse.ArgumentParser()

    # training
    parser.add_argument('--w_tr', type=float, default=10.0)
    parser.add_argument('--w_rot', type=float, default=10.0)
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument('--use_fixed_geodesic', action="store_true", default=False)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--steps', type=int, default=120000)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--clip', type=float, default=2.5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--no_ddp', action="store_true", default=False)
    parser.add_argument('--gpus', type=int, default=4)
    parser.add_argument('--ckpt', help='checkpoint to restore')
    parser.add_argument('--name', default='bla', help='name your experiment')
    # data
    parser.add_argument("--datapath")
    parser.add_argument("--weights")
    parser.add_argument("--image_size", default=[384,512])
    parser.add_argument("--exp")
    parser.add_argument("--checkpoint_dir")
    parser.add_argument('--gamma', type=float, default=0.9)    
    parser.add_argument('--use_mini_dataset', action='store_true')
    parser.add_argument('--streetlearn_interiornet_type', default='', choices=('',"nooverlap","T",'nooverlapT'))
    parser.add_argument('--dataset', default='matterport', choices=("matterport", "interiornet", 'streetlearn'))

    # model
    parser.add_argument('--no_pos_encoding', action='store_true')
    parser.add_argument('--noess', action='store_true')
    parser.add_argument('--cross_features', action='store_true')
    parser.add_argument('--use_single_softmax', action='store_true')  
    parser.add_argument('--l1_pos_encoding', action='store_true')
    parser.add_argument('--fusion_transformer', action="store_true", default=False)
    parser.add_argument('--cross_attn', nargs='+', type=int, help='indices for cross-attention, if using cross_image transformer connectivity')
    parser.add_argument('--fc_hidden_size', type=int, default=512)
    parser.add_argument('--pool_size', type=int, default=60)
    parser.add_argument('--transformer_depth', type=int, default=6)

    args = parser.parse_args()
    
    print(args)

    PATHS = ['output/%s/checkpoints' % (args.name), 'output/%s/runs' % (args.name), 'output/%s/train_output/images' % (args.name)]
    args.existing_ckpt = None

    for PATH in PATHS:
        try:
            os.makedirs(PATH)
        except:
            if 'checkpoints' in PATH:
                ckpts = os.listdir(PATH)

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
        
    if args.no_ddp:
        train(args.gpus, args)
    else:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12356'
        args.world_size = args.gpus
        mp.spawn(train, nprocs=args.gpus, args=(args,))
    
