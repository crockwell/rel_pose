from collections import OrderedDict
import numpy as np
import torch
from lietorch import SO3, SE3, Sim3
import math
from scipy.spatial.transform import Rotation as R
import torch.nn as nn

def graph_to_edge_list(graph):
    ii, jj, kk = [], [], []
    for s, u in enumerate(graph):
        for v in graph[u]:
            ii.append(u)
            jj.append(v)
            kk.append(s)

    ii = torch.as_tensor(ii)
    jj = torch.as_tensor(jj)
    kk = torch.as_tensor(kk)
    return ii, jj, kk

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

    metrics = {
        train_val+'_geo_loss_tr': (geodesic_loss_tr / len(Gs)).detach().item(),
        train_val+'_geo_loss_rot': (geodesic_loss_rot / len(Gs)).detach().item(),
    }

    return geodesic_loss_tr, geodesic_loss_rot, rotation_mag, metrics


def geodesic_loss(Ps, Gs, graph, gamma=0.9, do_scale=True, train_val='train'):
    """ Loss function for training network """

    ii, jj, kk = graph_to_edge_list(graph) 
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
        dG = Gs[i][:,jj] * Gs[i][:,ii].inv()

        if do_scale:
            s = fit_scale(dP, dG)
            dG = dG.scale(s[:,None])

        multed = Gs[i][:,1] * Ps[:,1].inv()

        d = (dG * dP.inv()).log()

        if isinstance(dG, SE3):
            tau, phi = d.split([3,3], dim=-1) 
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

        # calculating errors on surrounding inverses of ground truth to pred 
        dE = Sim3(dG * dP.inv()).detach()
        r_err, t_err, s_err = pose_metrics(dE)

    metrics = {
        train_val+'_geo_loss_tr': (geodesic_loss_tr / len(Gs)).detach().item(),
        train_val+'_geo_loss_rot': (geodesic_loss_rot / len(Gs)).detach().item(),
    }

    return geodesic_loss_tr, geodesic_loss_rot, rotation_mag, rotation_mag_gt, metrics
