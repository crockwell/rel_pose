import os
import pandas as pd 
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import cv2

# inputs to modify
radius = 20
line_width = 15
epipolar_points_x = 3
epipolar_points_y = 3
 
csv = pd.read_csv('matterport_all_gts.csv')
rel_pose = csv.iloc[0].to_numpy() # location (xyz) and rotation (quaternion: wxyz) e.g. [-0.11032, -0.53813, 2.46958, 0.92072, 0.00000, -0.38306, -0.07446]

curr_path1 = "0.png"  
out_path1 = "0_epipolar.png"

curr_path2 = "1.png"  
out_path2 = "1_epipolar.png"
# end inputs


def pos_quat2SE(quat_data):
    SO = R.from_quat(quat_data[3:7]).as_matrix()
    SE = np.matrix(np.eye(4))
    SE[0:3,0:3] = np.matrix(SO)
    SE[0:3,3]   = np.matrix(quat_data[0:3]).T
    SE = np.array(SE[0:3,:]).reshape(1,12)
    return SE

def get_epipolar(y, P):
    # given relative pose, have function that maps a point in image 1 to another point (or line) in image 2
    # E = [t]×R
    R = P[0,:3,:3]
    t = P[0,:,3]

    # cross prod rep of t
    t_x = torch.tensor([[0, -t[2], t[1]],
                        [t[2], 0, -t[0]],
                        [-t[1], t[0], 0]])
    E = (R @ t_x).numpy()
    
    y_prime = y @ E

    y_prime = y_prime / y_prime[1] # divide by y coord
    m = -y_prime[0]
    b = -y_prime[2]

    return m, b


colors = [
        np.array([197, 27, 125]),  # 'pink': 
        np.array([215, 48, 39]),  #  'red': 
        np.array([252, 141, 89]) - 60,  #  'light_orange': 
        np.array([175, 141, 195]),  #  'light_purple': 
        np.array([145, 191, 219]),  #  'light_blue': 
        np.array([161, 215, 106]) + 20,  # 'light_green': 
        np.array([77, 146, 33])+ 20,  # 'green': 
        np.array([118, 42, 131])+ 20,  #  'purple': 
        np.array([240, 10, 20]),  # red
]

startx = -1 + 2/(epipolar_points_x+1)
stopx = 1
stepx = 2/(epipolar_points_x+1)
starty = -1 + 2/(epipolar_points_y+1)
stopy = 1
stepy = 2/(epipolar_points_y+1)


# epipolar: dots on img 1
image_bg = cv2.imread(curr_path1)

for y1 in np.arange(startx, stopx, stepx):
    for y2 in np.arange(starty, stopy, stepy):
        pctx = (y1-startx)/(stopx-startx)
        pcty = (y2-starty)/(stopy-starty)
        color_num = int(pctx*(epipolar_points_x-1)*epipolar_points_x + pcty*epipolar_points_y)# int((y1+.5)*2*3 + 2*(y2+.5))
        color = ( int (colors[color_num] [ 0 ]), int (colors[color_num] [ 1 ]), int (colors[color_num] [ 2 ])) 
        y1_img = int((y1 + 1)/2 * image_bg.shape[1])
        y2_img = int((y2 + 1)/2 * image_bg.shape[0])
        cv2.circle(image_bg, (y1_img, y2_img), radius, color, -1)
cv2.imwrite(out_path1, image_bg)


# epipolar: lines across img 2
image_bg = np.array(cv2.imread(curr_path2))
image_epipolar = image_bg * 0

for y1 in np.arange(startx, stopx, stepx):
    for y2 in np.arange(starty, stopy, stepy):
        pctx = (y1-startx)/(stopx-startx)
        pcty = (y2-starty)/(stopy-starty)
        color_num = int(pctx*(epipolar_points_x-1)*epipolar_points_x + pcty*epipolar_points_y)
        color = ( int (colors[color_num] [ 0 ]), int (colors[color_num] [ 1 ]), int (colors[color_num] [ 2 ])) 
        rot_mtx = pos_quat2SE(rel_pose).reshape([1,3,4])
        y = np.array([y1,y2,1.0], dtype=np.float64)
        m, b = get_epipolar(y, torch.from_numpy(rot_mtx))
        point = np.array([-10.0, -10.0 * m + b])
        point_pa = np.array([10.0, 10.0 * m + b])

        point[0] = (point[0] + 1)/2 * image_epipolar.shape[1]
        point_pa[0] = (point_pa[0] + 1)/2 * image_epipolar.shape[1]
        point[1] = (point[1] + 1)/2 * image_epipolar.shape[0]
        point_pa[1] = (point_pa[1] + 1)/2 * image_epipolar.shape[0]
        cv2.line(image_epipolar, (int(point[0]), int(point[1])), (int(point_pa[0]), int(point_pa[1])),
            color, line_width)

image = cv2.addWeighted(image_epipolar,0.6,image_bg,0.8,0) 
cv2.imwrite(out_path2, image)
