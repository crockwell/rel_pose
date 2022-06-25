
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
    def __init__(self, name, datapath, n_frames=4, crop_size=[384,512], do_aug=True, subepoch=None, \
                scale_aug=True, is_training=True, gpu=0, load_img_tensors=False, max_scale_aug=0.25,
                no_depth=False, use_fixed_intrinsics=False, blackwhite=False, blackwhite_pt5=False, 
                use_optical_flow=False, streetlearn_interiornet_type=None,
                use_mini_dataset=False):
        """ Base class for RGBD dataset """
        self.aug = None
        self.root = datapath
        self.name = name
        self.streetlearn_interiornet_type = streetlearn_interiornet_type

        self.n_frames = n_frames

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
        return cv2.imread(image_file)

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

    def __len__(self):
        if self.matterport:
            return len(self.scene_info['images'])
        elif self.streetlearn_interiornet:
            return len(self.scene_info['images'])

    def __imul__(self, x):
        if self.matterport:
            self.scene_info['images'] *= x
            return self
        self.dataset_index *= x
        return self
