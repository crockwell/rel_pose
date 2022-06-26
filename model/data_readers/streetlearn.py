
import numpy as np
import torch
import glob
import cv2
import os
import os.path as osp

from lietorch import SE3
from .base import RGBDDataset
import json

from scipy.spatial.transform import Rotation as R

cur_path = '/home/cnris/vl/ExtremeRotation_code/'

test_split = np.load(osp.join(cur_path, 'metadata/streetlearn/test_pair_rotation.npy'), allow_pickle=True)
test_split = np.array(test_split, ndmin=1)[0]

test_split_T = np.load(osp.join(cur_path, 'metadata/streetlearnT/test_pair_translation.npy'), allow_pickle=True)
test_split_T = np.array(test_split_T, ndmin=1)[0]

class StreetLearn(RGBDDataset):

    def __init__(self, mode='training', **kwargs):
        self.mode = mode
        self.n_frames = 2

        super(StreetLearn, self).__init__(name='StreetLearn', **kwargs)

    @staticmethod 
    def is_test_scene(scene):
        return any(x in scene for x in test_split)

    def compute_rotation_matrix_from_two_matrices(self, m1, m2):
        batch = m1.shape[0]
        m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3
        return m

    def compute_rotation_matrix_from_viewpoint(self, rotation_x, rotation_y, batch):
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

    def compute_gt_rmat(self, rotation_x1, rotation_y1, rotation_x2, rotation_y2, batch_size):
        gt_mtx1 = self.compute_rotation_matrix_from_viewpoint(rotation_x1, rotation_y1, batch_size).view(batch_size, 3, 3)
        gt_mtx2 = self.compute_rotation_matrix_from_viewpoint(rotation_x2, rotation_y2, batch_size).view(batch_size, 3, 3)
        gt_rmat_matrix = self.compute_rotation_matrix_from_two_matrices(gt_mtx2, gt_mtx1).view(batch_size, 3, 3)
        return gt_rmat_matrix

    
    def compute_euler_angles_from_rotation_matrices(self, rotation_matrices):
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
    

    def _build_dataset(self, subepoch):
        valid = (subepoch==10)

        np.seterr(all="ignore")
        from tqdm import tqdm
        print("Building StreetLearn dataset")

        scene_info = {'images': [], 'poses': [], 'intrinsics': [], 'angles': []}
        base_pose = np.array([0,0,0,0,0,0,1])
        
        dset_name = 'streetlearn'

        if valid:
            if self.streetlearn_interiornet_type == '' or \
               self.streetlearn_interiornet_type == 'nooverlap':
                path = 'metadata/streetlearn/test_pair_rotation.npy'
            else:
                path = 'metadata/streetlearnT/test_pair_translation.npy'
                dset_name = 'streetlearn_2016'
        else:
            if self.streetlearn_interiornet_type == '':
                path = 'metadata/streetlearn/train_pair_rotation_overlap.npy'
                print('training with no translation and only overlapping images')
            elif self.streetlearn_interiornet_type == 'nooverlap':
                path = 'metadata/streetlearn/train_pair_rotation.npy'
                print('training with no translation and include non-overlapping images')
            elif self.streetlearn_interiornet_type == 'T':
                path = 'metadata/streetlearnT/train_pair_translation_overlap.npy'
                print('training with translation but only overlapping images')
                dset_name = 'streetlearn_2016'
            elif self.streetlearn_interiornet_type == 'nooverlapT':
                path = 'metadata/streetlearnT/train_pair_translation.npy'
                print('training with translation and include non-overlapping images')

        split = np.load(osp.join(cur_path, path), allow_pickle=True)
        split = np.array(split, ndmin=1)[0]

        if not valid:
            split_size = len(split.keys()) // 10
            start = split_size * (subepoch)
            end = split_size * (subepoch+1)
            if self.use_mini_dataset:
                start = 0
                end = 32000
        else:
            start = 0
            end = 9999999
            if self.use_mini_dataset:
                start = 0
                end = 5000

        for i in split.keys():  
            if i < start or i >= end:
                continue

            images = [os.path.join(cur_path, 'data', dset_name, split[i]['img1']['path']),
                        os.path.join(cur_path, 'data', dset_name, split[i]['img2']['path'])]
            
            x1, y1 = split[i]['img1']['x'], split[i]['img1']['y']
            x2, y2 = split[i]['img2']['x'], split[i]['img2']['y']

            # compute rotation matrix
            gt_rmat = self.compute_gt_rmat(torch.tensor([[x1]]), torch.tensor([[y1]]), torch.tensor([[x2]]), torch.tensor([[y2]]), 1)
            angle_x, angle_y, angle_z = self.compute_euler_angles_from_rotation_matrices(gt_rmat)
            angles = np.array([angle_x.item(), angle_y.item(), angle_z.item()])

            # get quaternions from rotation matrix
            r = R.from_matrix(gt_rmat)
            rotation = r.as_quat()[0]

            rel_pose = np.concatenate([np.array([0,0,0]), rotation]) # translation is 0

            poses = np.vstack([base_pose, rel_pose])

            intrinsics = np.array([[128,128,128,128], [128,128,128,128]]) # 256x256 imgs

            scene_info['images'].append(images)
            scene_info['poses'] += [poses]
            scene_info['intrinsics'] += [intrinsics] 
            scene_info['angles'] += [angles]   

        return scene_info

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

