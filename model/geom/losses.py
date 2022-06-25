from collections import OrderedDict
import numpy as np
import torch
from lietorch import SO3, SE3, Sim3
from .graph_utils import graph_to_edge_list
import math
from scipy.spatial.transform import Rotation as R
import torch.nn as nn

def pose_metrics(dE):
    """ Translation/Rotation/Scaling metrics from Sim3 """
    t, q, s = dE.data.split([3, 4, 1], -1)
    ang = SO3(q).log().norm(dim=-1)

    # convert radians to degrees
    r_err = (180 / np.pi) * ang
    t_err = t.norm(dim=-1)
    s_err = (s - 1.0).abs()
    return r_err, t_err, s_err

def fit_scale(Ps, Gs):
    b = Ps.shape[0]
    t1 = Ps.data[...,:3].detach().reshape(b, -1)
    t2 = Gs.data[...,:3].detach().reshape(b, -1)

    s = (t1*t2).sum(-1) / ((t2*t2).sum(-1) + 1e-8)
    return s

def euler_from_quaternion(xyzw):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    [x, y, z, w] = xyzw
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    return roll_x * 57.2958, pitch_y * 57.2958, yaw_z * 57.2958 # in degrees (was radians)

def fixed_geodesic_loss(Ps, Gs, gamma=0.9, train_val='train'):
    """ Loss function for training network """

    dP = Ps[:,1]

    n = len(Gs)
    geodesic_loss_tr = 0.0
    geodesic_loss_rot = 0.0
    rotation_mag = 0.0

    for i in range(n):
        w = gamma ** (n - i - 1)
        dG = Gs[i][:,1]
        multed = dG * dP.inv()
        d = multed.log()

        tau, phi = d.split([3,3], dim=-1)
        geodesic_loss_tr += w * tau.norm(dim=-1).mean()
        geodesic_loss_rot += w * phi.norm(dim=-1).mean()

        x,y,z = euler_from_quaternion(multed.data[0,3:].detach().cpu().numpy())
        rotation_mag += w * math.sqrt(x*x+y*y+z*z)
        #print(57.2958 * x, 57.2958 * y, 57.2958 * z)
        #import pdb; pdb.set_trace()

    metrics = {
        train_val+'_geo_loss_tr': (geodesic_loss_tr / len(Gs)).detach().item(),
        train_val+'_geo_loss_rot': (geodesic_loss_rot / len(Gs)).detach().item(),
    }

    return geodesic_loss_tr, geodesic_loss_rot, rotation_mag, metrics


def geodesic_loss(Ps, Gs, graph, gamma=0.9, do_scale=True, train_val='train'):
    """ Loss function for training network """

    # to summarize, this compares relative predicted pose to ground truth; so loss is over lie space

    # what is lie, se3 etc?
    # a lie group is a group that is a differentiable manifold (continuous)
    # a manifold locally resembles euclidean space
    # a group has multiplication and inverses
    # To clarify, SO3 is the group of 3d rotations about the origin "special orthogonal group"
    # SE3 "special euclidian group in 3 dim" is R t (4x4)
    # Sim3 - I believe similarity i.e. s* R t (4x4, then 1) 

    # relative pose
    # graph is two to the left and 2 to the right for each image
    # e.g. ([(0, [1, 2]), (1, [0, 2, 3]), (2, [0, 1, 3, 4]), (3, [1, 2, 4, 5]), (4, [2, 3, 5, 6]), (5, [3, 4, 6]), (6, [4, 5])])
    # jj is to the nearby images e.g.               [1, 2, 0, 2, 3, 0, 1, 3, 4, 1, 2, 4, 5, 2, 3, 5, 6, 3, 4, 6, 4, 5]
    # ii is what each image each elt in jj is e.g.  [0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6]
    ## kk appears to be same as ii
    ii, jj, kk = graph_to_edge_list(graph) 
    # Ps[:,jj] is surrounding images, Ps[:,ii] is individual images
    # so dP multiplies each image's pose by inverse of surrounding image's pose 
    dP = Ps[:,jj] * Ps[:,ii].inv()

    n = len(Gs)
    geodesic_loss = 0.0
    geodesic_loss_tr = 0.0
    geodesic_loss_rot = 0.0
    rotation_mag = 0.0

    multed_Ps = Ps[:,1] * Ps[:,0].inv()
    x,y,z = euler_from_quaternion(multed_Ps.data[0,3:].detach().cpu().numpy())
    rotation_mag_gt = math.sqrt(x*x+y*y+z*z)

    for i in range(n):
        w = gamma ** (n - i - 1)
        #print(i)
        #print(w)
        #print(Gs[i].data)
        dG = Gs[i][:,jj] * Gs[i][:,ii].inv()

        if do_scale:
            s = fit_scale(dP, dG)
            dG = dG.scale(s[:,None])

        # pose error - compare log of surrounding inverses of ground truth to pred
        # somehow this loses a dimension, not sure how (1,24,7) --> (1,24,6) from log
        multed = Gs[i][:,1] * Ps[:,1].inv()

        d = (dG * dP.inv()).log()

        if isinstance(dG, SE3):
            tau, phi = d.split([3,3], dim=-1) # tau I think is xyz loss, phi is rot loss
            #geodesic_loss += w * (
            #    tau.norm(dim=-1).mean() + 
            #    phi.norm(dim=-1).mean())
            #import pdb; pdb.set_trace()
            geodesic_loss_tr += tau.norm(dim=-1).mean()
            geodesic_loss_rot += phi.norm(dim=-1).mean()

            x,y,z = euler_from_quaternion(multed.data[0,3:].detach().cpu().numpy())
            rotation_mag += w * math.sqrt(x*x+y*y+z*z)

        elif isinstance(dG, Sim3):
            tau, phi, sig = d.split([3,3,1], dim=-1)
            geodesic_loss += w * (
                tau.norm(dim=-1).mean() + 
                phi.norm(dim=-1).mean() + 
                0.05 * sig.norm(dim=-1).mean())

        #import pdb; pdb.set_trace()
        # calculating errors on surrounding inverses of ground truth to pred 
        dE = Sim3(dG * dP.inv()).detach()
        r_err, t_err, s_err = pose_metrics(dE)

    metrics = {
        train_val+'_geo_loss_tr': (geodesic_loss_tr / len(Gs)).detach().item(),
        train_val+'_geo_loss_rot': (geodesic_loss_rot / len(Gs)).detach().item(),
    }

    return geodesic_loss_tr, geodesic_loss_rot, rotation_mag, rotation_mag_gt, metrics
