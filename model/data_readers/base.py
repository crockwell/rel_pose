
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import csv
import os
import cv2
import math
import random
import json
import pickle
import os.path as osp
import time

from .augmentation import RGBDAugmentor
from .rgbd_utils import *

class RGBDDataset(data.Dataset):
    def __init__(self, name, datapath, n_frames=4, crop_size=[384,512], fmin=8.0, fmax=75.0, do_aug=True, subepoch=None, \
                scale_aug=True, is_training=True, next_frame_prob=0, gpu=0, load_img_tensors=False, max_scale_aug=0.25,
                no_depth=False, use_fixed_intrinsics=False, blackwhite=False, blackwhite_pt5=False, 
                use_optical_flow=False, use_tar_data=False, streetlearn_interiornet_type=None,
                use_mini_dataset=False):
        """ Base class for RGBD dataset """
        self.aug = None
        self.root = datapath
        self.name = name
        self.streetlearn_interiornet_type = streetlearn_interiornet_type

        self.n_frames = n_frames
        self.fmin = fmin # exclude very easy examples
        self.fmax = fmax # exclude very hard examples
        self.next_frame_prob = next_frame_prob
        self.use_tar_data = use_tar_data

        self.no_depth = no_depth

        self.use_optical_flow = use_optical_flow
        
        self.scale_aug = scale_aug
        if do_aug:
            self.aug = RGBDAugmentor(crop_size=crop_size, scale_aug=scale_aug, max_scale=max_scale_aug, 
                use_fixed_intrinsics=use_fixed_intrinsics, blackwhite=blackwhite, blackwhite_pt5=blackwhite_pt5, datapath=datapath)
        print(self.name)
        self.matterport = False
        self.streetlearn_interiornet = False
        if 'mp3d' in datapath:
            self.matterport = True
            self.scene_info = self._build_dataset(subepoch==10)
        elif 'StreetLearn' in self.name or 'InteriorNet' in self.name:
            self.use_mini_dataset = use_mini_dataset
            self.streetlearn_interiornet = True
            self.scene_info = self._build_dataset(subepoch)
        else:

            # building dataset is expensive, cache so only needs to be performed once
            #cur_path = osp.dirname(osp.abspath(__file__))

            cache_folder_name = 'cache'
            if self.root == '/z/cnris/tartanair/':
                cache_folder_name = 'cache_epicfail'
            elif self.root == '/y/cnris/tartanair/':
                cache_folder_name = 'cache_newbox'

            cur_path = osp.dirname(osp.abspath(__file__))
            if not os.path.isdir(osp.join(cur_path, cache_folder_name)):
                try:
                    os.mkdir(osp.join(cur_path, cache_folder_name))
                except:
                    pass
                    
            print(cache_folder_name)
            if load_img_tensors:
                self.load_img_tensors = True
                cache_path = osp.join(cur_path, cache_folder_name, 'splits_with_disps-gpu', '{}/GPU_{}.pickle'.format(self.name+'_'+str(subepoch), str(gpu)))
                image_path = osp.join(cur_path, cache_folder_name, 'splits_with_disps-gpu', '{}/GPU_{}.npy'.format(self.name+'_'+str(subepoch), str(gpu)))        
                print('loading images')
                start_time = time.time()
                self.all_images = np.load(image_path)
                print('finished loading images in %s seconds' % (time.time() - start_time))
            else:
                self.load_img_tensors = False
                if scale_aug or use_optical_flow:
                    cache_path = osp.join(cur_path, cache_folder_name, 'splits', '{}.pickle'.format(self.name+'_'+str(subepoch)))
                else:
                    cache_path = osp.join(cur_path, cache_folder_name, 'splits_with_disps', '{}.pickle'.format(self.name+'_'+str(subepoch)))
            
            if subepoch is not None:
                scene_info = pickle.load(open(cache_path, 'rb'))[0]
            else:
                cache_path = osp.join(cur_path, cache_folder_name, '{}.pickle'.format(self.name))
                if osp.isfile(cache_path):
                    scene_info = pickle.load(open(cache_path, 'rb'))[0]
                else:
                    print('building dataset. cache path:', cache_path)
                    scene_info = self._build_dataset()
                    with open(cache_path, 'wb') as cachefile:
                        pickle.dump((scene_info,), cachefile)

            self.scene_info = scene_info
            if load_img_tensors:
                accumulated_num_images = 0
                for scene in scene_info:
                    print('gpu',str(gpu), scene)
                    num_in_scene = len(scene_info[scene]['images'])
                    self.scene_info[scene]['indices'] = np.arange(num_in_scene) + accumulated_num_images
                    accumulated_num_images += num_in_scene     
            
            self._build_dataset_index(is_training)
            if use_tar_data:
                import tarfile
                self.image_tar_files=tarfile.open('/tmpssd/tartanair.tar.gz')
                
    def _build_dataset_index(self, is_training=True):
        self.dataset_index = []
        held_out_type = 'validation'
        if not is_training:
            held_out_type = 'training'
        for scene in self.scene_info:
            skip_scene = False
            if (not self.__class__.is_test_scene(scene) and is_training) or \
                (self.__class__.is_test_scene(scene) and not is_training):            
                graph = self.scene_info[scene]['graph']
                for i in graph:
                    if len(graph[i][0]) > self.n_frames:
                        self.dataset_index.append((scene, i))
            else:
                print("Reserving {} for {}".format(scene, held_out_type))

    @staticmethod
    def image_read(image_file):
        if self.use_tar_data:
            print(image_file)
            image = self.image_tar_files.getmember(image_file)
            return image
        else:
            return cv2.imread(image_file)

    @staticmethod
    def depth_read(depth_file):
        if self.use_tar_data:
            print(depth_file)
            depth = self.image_tar_files.getmember(depth_file)
            return depth
        else:
            return np.load(depth_file)

    def build_frame_graph(self, poses, depths, intrinsics, f=16, max_flow=256):
        """ compute optical flow distance between all pairs of frames """
        def read_disp(fn):
            depth = self.__class__.depth_read(fn)[f//2::f, f//2::f]
            depth[depth < 0.01] = np.mean(depth)
            return 1.0 / depth

        poses = np.array(poses)
        intrinsics = np.array(intrinsics) / f
        
        disps = np.stack(list(map(read_disp, depths)), 0)
        d = f * compute_distance_matrix_flow(poses, disps, intrinsics)

        # uncomment for nice visualization
        # import matplotlib.pyplot as plt
        # plt.imshow(d)
        # plt.show()

        graph = {}
        for i in range(d.shape[0]):
            j, = np.where(d[i] < max_flow)
            graph[i] = (j, d[i,j])

        return graph

    def __getitem__(self, index):
        """ return training video """
        if self.matterport:
            images_list = self.scene_info['images'][index]
            poses = self.scene_info['poses'][index]
            intrinsics = self.scene_info['intrinsics'][index]
            class_rot = self.scene_info['class_rot'][index]
            class_tr = self.scene_info['class_tr'][index]

            images = []
            for i in range(2):
                images.append(self.__class__.image_read(images_list[i]))

            poses = np.stack(poses).astype(np.float32)
            intrinsics = np.stack(intrinsics).astype(np.float32)
            class_rot = np.stack(class_rot).astype(np.float32)
            class_tr = np.stack(class_tr).astype(np.float32)

            images = np.stack(images).astype(np.float32)
            images = torch.from_numpy(images).float()
            images = images.permute(0, 3, 1, 2)

            poses = torch.from_numpy(poses)
            intrinsics = torch.from_numpy(intrinsics)
            class_rot = torch.from_numpy(class_rot)
            class_tr = torch.from_numpy(class_tr)

            if self.aug is not None:
                images, poses, intrinsics, _ = self.aug(images, poses, intrinsics)
            
            return images, poses, intrinsics, class_rot, class_tr
        elif self.streetlearn_interiornet:
            local_index = index
            while True:
                try:
                    images_list = self.scene_info['images'][local_index]
                    poses = self.scene_info['poses'][local_index]
                    intrinsics = self.scene_info['intrinsics'][local_index]
                    angles = self.scene_info['angles'][local_index]

                    images = []
                    for i in range(2):
                        images.append(self.__class__.image_read(images_list[i]))

                    poses = np.stack(poses).astype(np.float32)
                    intrinsics = np.stack(intrinsics).astype(np.float32)
                    angles = np.stack(angles).astype(np.float32)
                    
                    images = np.stack(images).astype(np.float32)
                    images = torch.from_numpy(images).float()
                    images = images.permute(0, 3, 1, 2)

                    poses = torch.from_numpy(poses)
                    intrinsics = torch.from_numpy(intrinsics)

                    if self.aug is not None:
                        images, poses, intrinsics, _ = self.aug(images, poses, intrinsics)
                    
                    return images, poses, intrinsics, angles
                except:
                    local_index += 1
                    continue

        index = index % len(self.dataset_index)
        scene_id, ix = self.dataset_index[index]

        frame_graph = self.scene_info[scene_id]['graph']
        images_list = self.scene_info[scene_id]['images']
        if self.scale_aug or self.use_optical_flow:
            depths_list = self.scene_info[scene_id]['depths']
        else:
            disps_sum_list = self.scene_info[scene_id]['disps_sum']
            disps_ct_list = self.scene_info[scene_id]['disps_count']
        poses_list = self.scene_info[scene_id]['poses']
        intrinsics_list = self.scene_info[scene_id]['intrinsics']
        if self.load_img_tensors:
            indices_list = self.scene_info[scene_id]['indices']

        inds = [ ix ]

        if self.next_frame_prob > np.random.uniform():
            while len(inds) < self.n_frames:
                k = (frame_graph[ix][1] > 0)
                frames = frame_graph[ix][0][k]

                # prefer frames forward in time
                if np.count_nonzero(frames[frames > ix]):
                    ix = np.min(frames[frames > ix])
                
                elif np.count_nonzero(frames): # if last frame pick prior
                    ix = np.max(frames[frames < ix])
                
                inds += [ ix ]

            #k = (frame_graph[ix][1] > self.fmin) & (frame_graph[ix][1] < self.fmax)
            #frames2 = frame_graph[ix][0][k]
            #ix2 = np.random.choice(frames[frames > ix])
        else:
            while len(inds) < self.n_frames:
                # get other frames within flow threshold
                k = (frame_graph[ix][1] > self.fmin) & (frame_graph[ix][1] < self.fmax)
                frames = frame_graph[ix][0][k]

                # prefer frames forward in time
                if np.count_nonzero(frames[frames > ix]):
                    ix = np.random.choice(frames[frames > ix])
                
                elif np.count_nonzero(frames):
                    ix = np.random.choice(frames)

                inds += [ ix ]

        images, poses, disps_sum, disps_ct, depths, intrinsics = [], [], [], [], [], []
        for i in inds:
            if self.load_img_tensors:
                images.append(self.all_images[indices_list[i]])
            else:
                images.append(self.__class__.image_read(images_list[i]))
            if self.scale_aug or self.use_optical_flow:
                depths.append(self.__class__.depth_read(depths_list[i]))
            else:
                disps_sum.append(disps_sum_list[i])
                disps_ct.append(disps_ct_list[i])
            poses.append(poses_list[i])
            intrinsics.append(intrinsics_list[i])

        poses = np.stack(poses).astype(np.float32)
        intrinsics = np.stack(intrinsics).astype(np.float32)

        images = np.stack(images).astype(np.float32)
        images = torch.from_numpy(images).float()
        images = images.permute(0, 3, 1, 2)

        poses = torch.from_numpy(poses)
        intrinsics = torch.from_numpy(intrinsics)

        if self.use_optical_flow or (self.scale_aug and self.aug is not None):
            depths = np.stack(depths).astype(np.float32)
            disps = torch.from_numpy(1.0 / depths)
            images, poses, intrinsics, disps = self.aug(images, poses, intrinsics, disps)
            if len(disps[disps>0.01]) > 0:
                # scale scene via augmented depth 
                s = disps[disps>0.01].mean()
                disps = disps / s
                poses[...,:3] *= s
        else:
            disps_ct = np.stack(disps_ct).astype(np.float32)
            disps_sum = np.stack(disps_sum).astype(np.float32)
        
            if self.aug is not None:
                images, poses, intrinsics, _ = self.aug(images, poses, intrinsics)

                if not self.no_depth:
                    # scale scene via averages (no loaded / augmented depth)
                    if disps_ct.sum() > 0:
                        s = disps_sum.sum() / disps_ct.sum()
                        poses[...,:3] *= s


        if self.aug is None:
            return images, poses, intrinsics, images_list[inds[0]], images_list[inds[1]]

        elif self.use_optical_flow:
            return images, poses, intrinsics, disps

        return images, poses, intrinsics 

    def __len__(self):
        if self.matterport:
            return len(self.scene_info['images'])
        elif self.streetlearn_interiornet:
            return len(self.scene_info['images'])
        return len(self.dataset_index)

    def __imul__(self, x):
        if self.matterport:
            self.scene_info['images'] *= x
            return self
        self.dataset_index *= x
        return self
