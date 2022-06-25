import cv2
import pickle
import os 
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import math


extension = ''
DO_IMGS = False
DO_PKL = True

scene_info = pickle.load(open('droid_slam/data_readers/cache/TartanAir'+extension+'.pickle', 'rb'))[0]

# 369 scenes

try:
    os.makedirs('droid_slam/data_readers/cache/splits'+extension+'/')
except:
    pass

for i in range(10):
    subscene_info = {}
    cache_path = 'droid_slam/data_readers/cache/splits'+extension+'/TartanAir_'+str(i)+'.pickle'

    size = 0
    for j in range(37):
        scene_number = i + 10 * j 
        if scene_number < len(scene_info):
            key = [*scene_info.keys()][scene_number]
            size += len(scene_info[key]['images'])

    actual_images = np.zeros([size,384,512,3], dtype=np.uint8)
    print(size)
    current_count = 0
    for j in tqdm(range(37)):
        scene_number = i + 10 * j 
        # used to do this: i * 37 + j But want more diversity in each mini-batch!
        if scene_number < len(scene_info):
            key = [*scene_info.keys()][scene_number]
            if DO_PKL:
                subscene_info[key] = scene_info[key]

            if DO_IMGS:
                # TODO: also save tensor of all images from this subsplit
                images = scene_info[key]['images']
                for k in range(len(images)):
                    image = images[k]
                    local_image = cv2.imread(image)
                    local_image = torch.from_numpy(local_image).float().unsqueeze(dim=0)
                    local_image = local_image.permute(0, 3, 1, 2)
                    local_image = F.interpolate(local_image, size=[384,512])
                    local_image = local_image.permute(0, 2, 3, 1)[0]
                    actual_images[current_count] = local_image.numpy().astype(np.uint8)
                    #import pdb; pdb.set_trace()
                    current_count += 1

    if DO_PKL:
        with open(cache_path, 'wb') as cachefile:
            pickle.dump((subscene_info,), cachefile)

    if DO_IMGS:
        # TODO: dump images_real
        image_path = 'droid_slam/data_readers/cache/splits'+extension+'/TartanAir_'+str(i)+'_images.npy'
        np.save(image_path, actual_images)


'''
extension = '_with_disps-gpu'
DO_IMGS = True
DO_PKL = True
NUM_GPUS = 8

scene_info = pickle.load(open('droid_slam/data_readers/cache/TartanAir'+extension+'.pickle', 'rb'))[0]

# 369 scenes

try:
    os.makedirs('droid_slam/data_readers/cache/splits'+extension+'/')
except:
    pass

MAX = 37
mapping_j_gpu = {}

for i in range(8,10):
    subscene_info = {}

    actual_images = {}
    size = np.zeros(NUM_GPUS)
    for j in range(MAX):
        scene_number = i + 10 * j 
        #gpu = j // BUCKETS
        if scene_number < len(scene_info):
            key = [*scene_info.keys()][scene_number]

            gpu = np.argmin(size)
            size[gpu] += len(scene_info[key]['images'])
            mapping_j_gpu[j] = gpu

    for j in range(NUM_GPUS):
        actual_images[j] = np.zeros([int(size[j]),384,512,3], dtype=np.uint8)
        subscene_info[j] = {}

    print(size)
    current_count = np.zeros(10, dtype=np.int)
    for j in tqdm(range(37)):
        scene_number = i + 10 * j 
        # used to do this: i * 37 + j But want more diversity in each mini-batch!
        if scene_number < len(scene_info):
            key = [*scene_info.keys()][scene_number]
            gpu = mapping_j_gpu[j]
            #gpu = j // BUCKETS

            if DO_PKL:
                subscene_info[gpu][key] = scene_info[key]

            if DO_IMGS:
                # TODO: also save tensor of all images from this subsplit
                images = scene_info[key]['images']
                for k in range(len(images)):
                    subsplit = (k * NUM_GPUS) // len(images) 
                    image = images[k]
                    local_image = cv2.imread(image)
                    local_image = torch.from_numpy(local_image).float().unsqueeze(dim=0)
                    local_image = local_image.permute(0, 3, 1, 2)
                    local_image = F.interpolate(local_image, size=[384,512])
                    local_image = local_image.permute(0, 2, 3, 1)[0]
                    #import pdb; pdb.set_trace()
                    actual_images[gpu][current_count[gpu]] = local_image.numpy().astype(np.uint8)
                    current_count[gpu] += 1
                    #print(j, gpu, current_count[gpu], key)

    for gpu in range(NUM_GPUS):
        try: 
            os.makedirs('droid_slam/data_readers/cache/splits'+extension+'/TartanAir_'+str(i))
        except:
            pass

        if DO_PKL:
            cache_path = 'droid_slam/data_readers/cache/splits'+extension+'/TartanAir_'+str(i)+'/GPU_'+str(gpu)+'.pickle'
            with open(cache_path, 'wb') as cachefile:
                pickle.dump((subscene_info[gpu],), cachefile)

        if DO_IMGS:
            image_path = 'droid_slam/data_readers/cache/splits'+extension+'/TartanAir_'+str(i)+'/GPU_'+str(gpu)+'.npy'
            np.save(image_path, actual_images[gpu])
'''