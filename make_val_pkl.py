import cv2
import pickle
import os 
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


scene_info = pickle.load(open('droid_slam/data_readers/cache/TartanAir.pickle', 'rb'))[0]

# 369 scenes

test_split = os.path.join('droid_slam/data_readers', 'tartan_test.txt')
test_split = open(test_split).read().split()

cache_path = 'droid_slam/data_readers/cache/splits/TartanAir_10.pickle'

subscene_info = {}
for scene_number in range(len([*scene_info.keys()])):
    key = [*scene_info.keys()][scene_number]
    if any(x in key for x in test_split):
        subscene_info[key] = scene_info[key]

with open(cache_path, 'wb') as cachefile:
    pickle.dump((subscene_info,), cachefile)

print(len([*subscene_info.keys()]))


'''
extension = '_with_disps-gpu'
scene_info = pickle.load(open('droid_slam/data_readers/cache/TartanAir'+extension+'.pickle', 'rb'))[0]
NUM_GPUS = 8
DO_IMGS = True 
DO_PKL = True
test_split = os.path.join('droid_slam/data_readers', 'tartan_test.txt')
test_split = open(test_split).read().split()

# 369 scenes

try:
    os.makedirs('droid_slam/data_readers/cache/splits'+extension+'/')
except:
    pass

subscene_info = {}
cache_path = 'droid_slam/data_readers/cache/splits'+extension+'/TartanAir_10.pickle'

MAX = 37

actual_images = {}
subscene_info = {}
current_count = np.zeros(10, dtype=np.int)
mapping_j_gpu = {}

size = np.zeros(NUM_GPUS)
for scene_number in range(len([*scene_info.keys()])):
    key = [*scene_info.keys()][scene_number]
    if any(x in key for x in test_split):
        gpu = np.argmin(size)
        size[gpu] += len(scene_info[key]['images'])
        mapping_j_gpu[scene_number] = gpu

for j in range(NUM_GPUS):
    actual_images[j] = np.zeros([int(size[j]),384,512,3], dtype=np.uint8)
    subscene_info[j] = {}


for scene_number in tqdm(range(len([*scene_info.keys()]))):
    key = [*scene_info.keys()][scene_number]
    if any(x in key for x in test_split):
        gpu = mapping_j_gpu[scene_number]
        if DO_PKL:
            subscene_info[gpu][key] = scene_info[key]

        if DO_IMGS:
            images = scene_info[key]['images']
            for k in range(len(images)):
                subsplit = (k * NUM_GPUS) // len(images) 
                image = images[k]
                local_image = cv2.imread(image)
                local_image = torch.from_numpy(local_image).float().unsqueeze(dim=0)
                local_image = local_image.permute(0, 3, 1, 2)
                local_image = F.interpolate(local_image, size=[384,512])
                local_image = local_image.permute(0, 2, 3, 1)[0]
                actual_images[gpu][current_count[gpu]] = local_image.numpy().astype(np.uint8)
                current_count[gpu] += 1


print(len([*subscene_info.keys()]))

i = 10
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