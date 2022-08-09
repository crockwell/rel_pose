
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

class RGBDDataset(data.Dataset):
    def __init__(self, name, datapath, reshape_size=[384,512], subepoch=None, \
                is_training=True, gpu=0, use_fixed_intrinsics=False, 
                streetlearn_interiornet_type=None,
                use_mini_dataset=False):
        """ Base class for RGBD dataset """
        self.root = datapath
        self.name = name
        self.streetlearn_interiornet_type = streetlearn_interiornet_type
        
        self.aug = RGBDAugmentor(reshape_size=reshape_size, 
            use_fixed_intrinsics=use_fixed_intrinsics, datapath=datapath)
        print(self.name)
        self.matterport = False
        if 'mp3d' in datapath:
            self.matterport = True
            self.scene_info = self._build_dataset(subepoch==10)
        elif 'StreetLearn' in self.name or 'InteriorNet' in self.name:
            self.use_mini_dataset = use_mini_dataset
            self.scene_info = self._build_dataset(subepoch)

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

    def __getitem__(self, index):
        """ return training video """
        if self.matterport:
            images_list = self.scene_info['images'][index]
            poses = self.scene_info['poses'][index]
            intrinsics = self.scene_info['intrinsics'][index]

            images = []
            for i in range(2):
                images.append(self.__class__.image_read(images_list[i]))

            poses = np.stack(poses).astype(np.float32)
            intrinsics = np.stack(intrinsics).astype(np.float32)

            images = np.stack(images).astype(np.float32)
            images = torch.from_numpy(images).float()
            images = images.permute(0, 3, 1, 2)

            poses = torch.from_numpy(poses)
            intrinsics = torch.from_numpy(intrinsics)

            images, poses, intrinsics = self.aug(images, poses, intrinsics)
            
            return images, poses, intrinsics
        else:
            local_index = index
            # in case index fails
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

                    images, poses, intrinsics = self.aug(images, poses, intrinsics)
                    
                    return images, poses, intrinsics, angles
                except:
                    local_index += 1
                    continue

    def __len__(self):
        return len(self.scene_info['images'])
