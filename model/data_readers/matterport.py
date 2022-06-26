
import numpy as np
import torch
import glob
import cv2
import os
import os.path as osp
import pickle

from lietorch import SE3
from .base import RGBDDataset
import json

cur_path = '/home/cnris/data/mp3d_rpnet_v4_sep20'

with open(osp.join(cur_path, 'mp3d_planercnn_json/cached_set_test.json')) as f:
    test_split = json.load(f)

class Matterport(RGBDDataset):

    # scale depths to balance rot & trans
    DEPTH_SCALE = 5.0

    def __init__(self, mode='training', **kwargs):
        self.mode = mode
        self.n_frames = 2

        kmeans_trans_path = '/home/cnris/vl/SparsePlanes/sparsePlane/models/kmeans_trans_32.pkl'
        kmeans_rots_path = '/home/cnris/vl/SparsePlanes/sparsePlane/models/kmeans_rots_32.pkl'
        assert os.path.exists(kmeans_trans_path)
        assert os.path.exists(kmeans_rots_path)
        with open(kmeans_trans_path, "rb") as f:
            self.kmeans_trans = pickle.load(f)
        with open(kmeans_rots_path, "rb") as f:
            self.kmeans_rots = pickle.load(f)

        super(Matterport, self).__init__(name='Matterport', **kwargs)

    @staticmethod 
    def is_test_scene(scene):
        return any(x in scene for x in test_split)

    def class2xyz(self, cls):
        assert (cls >= 0).all() and (cls < self.kmeans_trans.n_clusters).all()
        return self.kmeans_trans.cluster_centers_[cls]

    def class2quat(self, cls):
        assert (cls >= 0).all() and (cls < self.kmeans_rots.n_clusters).all()
        return self.kmeans_rots.cluster_centers_[cls]

    def xyz2class(self, x, y, z):
        return self.kmeans_trans.predict([[x, y, z]])

    def quat2class(self, w, xi, yi, zi):
        return self.kmeans_rots.predict([[w, xi, yi, zi]])

    def _build_dataset(self, valid=False):
        np.seterr(all="ignore")
        from tqdm import tqdm
        print("Building Matterport dataset")

        scene_info = {'images': [], 'poses': [], 'intrinsics': [], 'class_rot': [], 'class_tr': []}
        base_pose = np.array([0,0,0,0,0,0,1])
        
        path = 'cached_set_train.json'
        if valid:
            path = 'cached_set_val.json'
        with open(osp.join(cur_path, 'mp3d_planercnn_json', path)) as f:
            split = json.load(f)

        for i in range(len(split['data'])):
            images = []
            for imgnum in ['0', '1']:
                img_name = os.path.join(cur_path, '/'.join(str(split['data'][i][imgnum]['file_name']).split('/')[6:]))
                images.append(img_name)
            
            rel_pose = np.array(split['data'][i]['rel_pose']['position'] + split['data'][i]['rel_pose']['rotation'])
            og_rel_pose = np.copy(rel_pose)
            rel_pose[:3] /= Matterport.DEPTH_SCALE
            cprp = np.copy(rel_pose)
            rel_pose[6] = cprp[3] # swap 3 & 6, we want W last.
            rel_pose[3] = cprp[6]
            if rel_pose[6] < 0:
                rel_pose[3:] *= -1
            poses = np.vstack([base_pose, rel_pose])

            intrinsics = np.array([[517.97, 517.97, 320, 240], [517.97, 517.97, 320, 240]]) # 480 x 640 imgs

            scene_info['images'].append(images)
            scene_info['poses'] += [poses]
            scene_info['intrinsics'] += [intrinsics]     
            scene_info['class_rot'] += [self.quat2class(og_rel_pose[3], og_rel_pose[4], og_rel_pose[5], og_rel_pose[6])]    
            scene_info['class_tr'] += [self.xyz2class(og_rel_pose[0], og_rel_pose[1], og_rel_pose[2])]

        return scene_info

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)
