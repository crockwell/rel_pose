import os
import pandas as pd 
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import cv2
import json
import shutil

### start inputs
curr_path1 = "demo/matterport_1.png"
curr_path2 = "demo/matterport_2.png"
rel_pose_input = np.array([2.17275, 0.1722, -0.87071, 0.00044, 0.54702, 0.10733, 0.83021]) # in format x,y,z,qx,qy,qz,qw

out_path1_points = "demo/matterport_1_epipolar_points.png"
out_path2_lines = "demo/matterport_2_epipolar_lines.png"

INTRINSICS = torch.zeros(1,3,3).double()
INTRINSICS[0,0,0] = 517.97
INTRINSICS[0,1,1] = 517.97
INTRINSICS[0,0,2] = 320
INTRINSICS[0,1,2] = 240
INTRINSICS[0,2,2] = 1

radius = 20
line_width = 15
epipolar_points_x = 3
epipolar_points_y = 3

### end inputs

def transform_helper(mtx, transform):
    """
    input: 4x4 rotation mtx and 4x4 rotation matrix (transform)
        transform should be from tgt_coord -> src_coord 
    output: matrix after transform is applied
    """        
    transformed_mtx = np.linalg.inv(transform) @ mtx @ transform
    return transformed_mtx

def vec2mtx(vec):
    """
    output is 7D vec, fmt x,y,z,qx,qy,qz,qw
    input 4x4 rotation mtx
    """    
    mtx = np.eye(4)
    mtx[:3,:3] = R.as_matrix(R.from_quat(vec[3:]))
    mtx[:3,3] = vec[:3]
    return mtx

def mtx2vec(mtx):
    """
    input 4x4 rotation mtx
    output is 7D vec, fmt x,y,z,qx,qy,qz,qw
    """
    vec = np.zeros(7)
    quat = R.as_quat(R.from_matrix(mtx[:3,:3]))
    vec[3:] = quat
    vec[:3] = mtx[:3,3]
    return vec

def apply_transform(vector, transform):
    mtx = vec2mtx(vector)
    transformed_mtx = transform_helper(mtx, transform)
    transformed_vector = mtx2vec(transformed_mtx)
    return transformed_vector

def transform_x(th):
    rot = np.eye(4)
    rot[:3,:3] = np.array([
        [1,          0,           0],
        [0, np.cos(th), -np.sin(th)],
        [0, np.sin(th),  np.cos(th)],
    ])
    return rot

def pos_quat2SE(quat_data):
    SO = R.from_quat(quat_data[3:7]).as_matrix()
    SE = np.matrix(np.eye(4))
    SE[0:3,0:3] = np.matrix(SO)
    SE[0:3,3]   = np.matrix(quat_data[0:3]).T
    SE = np.array(SE[0:3,:]).reshape(1,12)
    return SE

def compute_correspond_epilines(points, F_mat):
    if points.shape[-1] == 2:
        import pdb; pdb.set_trace()
        # points_h = pad(points, [0, 1], "constant", 1.0)
    elif points.shape[-1] == 3:
        points_h = points
    else:
        raise AssertionError(points.shape)
    # project points and retrieve lines components
    points_h = torch.transpose(points_h, dim0=-2, dim1=-1)
    a, b, c = torch.chunk(F_mat @ points_h, dim=-2, chunks=3)

    # compute normal and compose equation line
    nu = a * a + b * b
    nu = torch.where(nu > 0.0, 1.0 / torch.sqrt(nu), torch.ones_like(nu))

    line = torch.cat([a * nu, b * nu, c * nu], dim=-2)  # *x3xN
    return torch.transpose(line, dim0=-2, dim1=-1)  # *xNx3

def fundamental_from_essential(E_mat, K1, K2):
    return K2.inverse().transpose(-2, -1) @ E_mat @ K1.inverse()

def get_epipolar(y, P):
    # given relative pose, have function that maps a point in image 1 to another point (or line) in image 2
    # E = [t]×R
    R2 = P[0,:3,:3]
    t2 = P[0,:,3]

    # cross prod rep of t
    t_x = torch.tensor([[0, -t2[2], t2[1]],
                        [t2[2], 0, -t2[0]],
                        [-t2[1], t2[0], 0]])

    E = (t_x @ R2).numpy()

    points = torch.from_numpy(y).unsqueeze(0)

    K1 = INTRINSICS
    K2 = K1
    F_mat = fundamental_from_essential(torch.from_numpy(E).unsqueeze(0), K1, K2)

    epiline = compute_correspond_epilines(points, F_mat)[0,0]

    m = -epiline[0] / epiline[1]
    b = -epiline[2] / epiline[1]

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
cv2.imwrite(out_path1_points, image_bg)

rel_pose = np.copy(rel_pose_input)
tf_x = transform_x(np.pi)
rel_pose = apply_transform(rel_pose, tf_x)

# epipolar: lines across img 2
image_bg = np.array(cv2.imread(curr_path2))
image_epipolar = image_bg * 0
img_width = image_bg.shape[1]

for y1 in np.arange(startx, stopx, stepx):
    for y2 in np.arange(starty, stopy, stepy):
        pctx = (y1-startx)/(stopx-startx)
        pcty = (y2-starty)/(stopy-starty)
        color_num = int(pctx*(epipolar_points_x-1)*epipolar_points_x + pcty*epipolar_points_y)
        color = ( int (colors[color_num] [ 0 ]), int (colors[color_num] [ 1 ]), int (colors[color_num] [ 2 ])) 
        rot_mtx = pos_quat2SE(rel_pose).reshape([1,3,4])
        y = np.array([(y1+1)/2*image_bg.shape[1],(y2+1)/2*image_bg.shape[0],1.0], dtype=np.float64)
        m, b = get_epipolar(y, torch.from_numpy(rot_mtx))

        x0, y0 = map(int, [0, b])
        x_end, y_end = map(int, [img_width, b+m*img_width])
        cv2.line(image_epipolar, (x0, y0), (x_end, y_end), color, line_width)

image = cv2.addWeighted(image_epipolar,0.6,image_bg,0.8,0) 
cv2.imwrite(out_path2_lines, image)

