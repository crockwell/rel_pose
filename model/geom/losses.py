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

def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append((index % dim).item())
        index = index // dim
    return list(reversed(out))

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

def get_point_line_distance(UV, E, B, h, w):
    UV = UV.reshape([B,-1,3])
    lines = (UV @ E).reshape([B,h,w,3]) 

    L = torch.unsqueeze(torch.unsqueeze(lines, 1), 1)

    UVP = UV.reshape([B,h,w,3])
    UVP = torch.unsqueeze(torch.unsqueeze(UVP, 3), 3)

    normalize = torch.sqrt(torch.sum(lines[:,:,:,:2]**2,axis=3))
    N = torch.unsqueeze(torch.unsqueeze(normalize, 1), 1)

    # line distance = |ax + by + c| / sqrt(a^2+b^2)
    # point_line_distance: i,j,h,w is at point i,j; we are this distance from epipolar line h,w
    # identity is invalid and will thus be nan; we thus set this to 0
    point_line_distance = torch.nan_to_num(torch.abs(L[:,:,:,:,:,0] * UVP[:,:,:,:,:,0] + L[:,:,:,:,:,1] * UVP[:,:,:,:,:,1]+ L[:,:,:,:,:,2]) / N)

    return point_line_distance

def get_epipolar_loss_one_dir(UV, E, attention_scores, first_head_only, epi_dist_scale, epi_dist_sub, B, h, w, use_sigmoid_attn):
    point_line_distance = get_point_line_distance(UV, E, B, h, w)

    # scores: abn,hn,i,j,h,w shows point i,j has probability of each h,w corresponding to it for attention branch abn and head num hn
    scores = attention_scores.reshape([B,1,3,h,w,h,w])

    #### by constraining scores to be at least 1e-9, they are more numerically stable
    # sum will always be at least 1e-4
    minimum_score = torch.ones_like(scores) * 1e-9
    scores = torch.max(minimum_score, scores)

    # close distance is used to determine weight of preds within 5% of epipolar lines
    close_distance = (point_line_distance < .1).float()

    #if use_sigmoid_attn:
    # we want to use sigmoid of point line distance as our function
    # then, we try to predict heavy weights on stuff not punished much!
    # first apply squeeze & stretch to sigmoid, so 0 line distance -> ~0 loss
    shifted_line_distance = point_line_distance * epi_dist_scale - epi_dist_sub
    final_line_distance = torch.sigmoid(shifted_line_distance)
    #else:
    #    final_line_distance = point_line_distance

    #num_cells = h*w*h*w
    # we multiply prob at each point times the distance of that point to epipolar line. Then pass through sigmoid. 
    if first_head_only:
        # the below is only for one attention branch (second one would have diff epipolar lines), and one head.
        sum_scores = scores[:,0,0].sum(-1).sum(-1).sum(-1).sum(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # scores now sum up to 1 per head.
        normalized_scores = scores[:,0,0] / sum_scores

        epipolar_pct_within_5cpt = (scores[:,0,0] * close_distance).sum() / sum_scores.sum()
    else:
        # divide scores by sum across it's head
        # scores now sum up to 1 per head.
        # if single softmax, would've been 24x24! if sigmoid, could be whatever.
        sum_scores = scores[:,0].sum(-1).sum(-1).sum(-1).sum(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        normalized_scores = scores[:,0] / sum_scores
        
        epipolar_pct_within_5cpt = (scores[:,0] * close_distance.unsqueeze(1)).sum() / sum_scores.sum()

    # loss is the average of sigmoid distance model predicts from epipolar basin. 
    penalties = normalized_scores * final_line_distance.unsqueeze(1)
    loss = penalties.sum(-1).sum(-1).sum(-1).sum(-1).mean()
    
    # make minimum loss 0
    loss -= torch.sigmoid(-1 * torch.tensor([epi_dist_sub]).cuda())[0]

    #print(loss.mean())
    #print(sum_scores.mean())

    return loss, epipolar_pct_within_5cpt

def get_epipolar_loss_one_dir_JD(UV, E, attention_scores, first_head_only, epi_dist_scale, epi_dist_sub, B, h, w, loss_on_each_head=False):
    point_line_distance = get_point_line_distance(UV, E, B, h, w)

    # scores: abn,hn,i,j,h,w shows point i,j has probability of each h,w corresponding to it for attention branch abn and head num hn
    scores = attention_scores.reshape([B,1,3,h,w,h,w])

    #### by constraining scores to be at least 1e-9, they are more numerically stable
    # sum will always be at least 1e-4
    minimum_score = torch.ones_like(scores) * 1e-9
    scores = torch.max(minimum_score, scores)

    # close distance is used to determine weight of preds within 5% of epipolar lines
    close_distance = (point_line_distance < .1).unsqueeze(1) # 6, 1, 24, 24, 24, 24

    # we want to use sigmoid of point line distance as our function
    # then, we try to predict heavy weights on stuff not punished much!
    # first apply squeeze & stretch to sigmoid, so 0 line distance -> ~0 loss
    shifted_line_distance = point_line_distance * epi_dist_scale - epi_dist_sub
    sigmoid_line_distance = (1 - torch.sigmoid(shifted_line_distance)).unsqueeze(1)

    sum_scores = scores[:,0].sum(-1).sum(-1).sum(-1).sum(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    
    epipolar_pct_within_5cpt = (scores[:,0] * close_distance.float()).sum() / sum_scores.sum()

    # loss is max(0, attention_score - sigmoid(point line distance)) + max(0, sigmoid_line_distance - argmax_in_epipolar(attention_scores)). 
    penalty_min = torch.max(torch.zeros_like(scores[:,0]), scores[:,0] - sigmoid_line_distance).sum() / (B * 3)

    penalty_max = 0
    if loss_on_each_head:
        # WE could try to use for each head separately, but some don't even have any elements that are withing the threshold!
        for b in range(B):
            for i in range(3):
                for k in range(24):
                    for l in range(24):
                        # for each image, head, and location, we get argmax & compare it to sigmoid to encourage scores to go up
                        close_tensor = scores[b,:,i,k,l][close_distance[b,:,k,l]]
                        if close_tensor.shape[0] == 0:
                            # no elements in the head that are close.
                            continue
                        index = torch.argmax(close_tensor)
                        penalty_max +=torch.max(torch.zeros([1]).cuda(), sigmoid_line_distance[b,:,k,l][close_distance[b,:,k,l]][index] - scores[b,:,i,k,l][close_distance[b,:,k,l]][index])[0]
        penalty_max /= (B * 3 * 24 * 24)

    else:
        for b in range(B):
            for i in range(3):
                # for each image and head, we get argmax & compare it to sigmoid to encourage scores to go up
                index = torch.argmax(scores[b,:,i][close_distance[b]])
                penalty_max +=torch.max(torch.zeros([1]).cuda(), sigmoid_line_distance[b][close_distance[b]][index] - scores[b,:,i][close_distance[b]][index])[0]

        penalty_max /= (B * 3)
    

    return penalty_min, penalty_max, epipolar_pct_within_5cpt


def get_epipolar_line(Ps, B):
    Rt = Ps.matrix()[:,:3]
    R = Rt[:,:3,:3]
    t = Rt[:,:,3]

    # cross prod rep of t
    t_x = torch.zeros([B, 3, 3]).cuda()
    t_x[:,0,1] = -t[:,2]
    t_x[:,0,2] = t[:,1]
    t_x[:,1,0] = t[:,2]
    t_x[:,1,2] = -t[:,0]
    t_x[:,2,0] = -t[:,1]
    t_x[:,2,1] = t[:,0]
    E = (R @ t_x)
    return E

'''
Distribution loss: abs(prob-(1-sigmoid of distance)) {handles points that are too high} and then possibly one that handles points that are too low

'''

def epipolar_loss(Ps, attention_scores, train_val='train', epi_dist_scale=1, epi_dist_sub=1, first_head_only=False, 
                    epipolar_loss_both_dirs=False, JD=False, loss_on_each_head=False, use_sigmoid_attn=False):
    """ Loss by getting epipolar lines given ground truth pose, comparing to attention scores"""

    B, _, _, N, _ = attention_scores.shape
    h,w = 48,64
    if N == 24*24:
        h,w = 24,24
    elif N != 48*64:
        print('unexpected resolution for epipolar loss')
        assert(False)

    E1 = get_epipolar_line(Ps[:,1], B)
    E2 = get_epipolar_line(Ps[:,1].inv(), B)

    # calcing line distance
    x_ = torch.linspace(-1,1,w)
    y_ = torch.linspace(-1,1,h)
    xy = torch.meshgrid(x_,y_)
    z_ = torch.ones([w,h])
    UV = torch.stack([xy[1],xy[0],z_]).transpose(0,1).transpose(1,2).double().unsqueeze(0).repeat(B,1,1,1).cuda().float()#(1,2,0)

    if JD:
        epipolar_loss_min, epipolar_loss_max, epipolar_pct_within_5cpt = get_epipolar_loss_one_dir_JD(UV, E1, attention_scores[:,0], first_head_only, epi_dist_scale, epi_dist_sub, B, h, w, loss_on_each_head)
    else:
        epipolar_loss, epipolar_pct_within_5cpt = get_epipolar_loss_one_dir(UV, E1, attention_scores[:,0], first_head_only, epi_dist_scale, epi_dist_sub, B, h, w, use_sigmoid_attn)
    epipolar_pct_within_5cpt = [epipolar_pct_within_5cpt]
    if epipolar_loss_both_dirs:
        if JD:
            loss_2_1, epipolar_loss_max2, epipolar_pct_within_5cpt_2_1 = get_epipolar_loss_one_dir_JD(UV, E2, attention_scores[:,1], first_head_only, epi_dist_scale, epi_dist_sub, B, h, w, loss_on_each_head)
            epipolar_loss_max = (epipolar_loss_max + epipolar_loss_max2) / 2
            epipolar_loss_min = (epipolar_loss_min + loss_2_1) / 2
        else:
            loss_2_1, epipolar_pct_within_5cpt_2_1 = get_epipolar_loss_one_dir(UV, E2, attention_scores[:,1], first_head_only, epi_dist_scale, epi_dist_sub, B, h, w, use_sigmoid_attn)
            epipolar_loss = (epipolar_loss + loss_2_1) / 2
        epipolar_pct_within_5cpt += [epipolar_pct_within_5cpt_2_1]
    
    #print(point_line_distance.max(), Rt)
    #print(epipolar_loss, epipolar_pct_within_5cpt)
    #import pdb; pdb.set_trace()

    if JD:
        metrics = {
            train_val+'_epipolar_loss_min': epipolar_loss_min.detach().item(),
            train_val+'_epipolar_loss_max': epipolar_loss_max.detach().item(),
            train_val+'_attn_weight_within_5pct_of_epipolar': epipolar_pct_within_5cpt[0].detach().item(),
        }
    else:
        metrics = {
            train_val+'_epipolar_loss': epipolar_loss.detach().item(),
            train_val+'_attn_weight_within_5pct_of_epipolar': epipolar_pct_within_5cpt[0].detach().item(),
        }

    if epipolar_loss_both_dirs:
        metrics[train_val+'_attn_weight_within_5pct_of_epipolar_img2_1'] = epipolar_pct_within_5cpt[1].detach().item()

    if JD:
        return (epipolar_loss_min, epipolar_loss_max), metrics

    return epipolar_loss, metrics



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

def gt_geodesic_loss_l1(Ps, Gs, graph, gamma=0.9, do_scale=True, train_val='train'):
    """ Loss function for training network """

    # to summarize, this compares relative predicted pose to ground truth; so loss is over lie space
    # instead of computing difference in relative pose, however, we compute difference in actual pose

    ii, jj, kk = graph_to_edge_list(graph) 
    # Ps[:,jj] is surrounding images, Ps[:,ii] is individual images
    # so dP multiplies each image's pose by inverse of surrounding image's pose 
    dP = Ps[:,jj] * Ps[:,ii].inv()

    n = len(Gs)
    geodesic_loss = 0.0

    for i in range(n):
        w = gamma ** (n - i - 1)
        #dG = Gs[i][:,jj] * Gs[i][:,ii].inv()

        #if do_scale:
        #    s = fit_scale(dP, dG)
        #    dG = dG.scale(s[:,None])

        # pose error - compare log of surrounding inverses of ground truth to pred
        # somehow this loses a dimension, not sure how (1,24,7) --> (1,24,6) from log
        #d = (dG * dP.inv()).log()
        d = torch.abs(Gs[i].data - Ps.data)
        #d2 = torch.square(Gs[i].data - Ps.data)
        
        tau, phi = d.split([3,4], dim=-1) # tau I think is xyz loss, phi is rot loss
        #tau2, phi2 = d2.split([3,4], dim=-1)
        geodesic_loss += w * (
            tau.norm(dim=-1).mean() + 
            phi.norm(dim=-1).mean())
        #geodesic_loss += w * (
        #    tau.mean() + 
        #    phi.mean())
        
        # calculating errors on surrounding inverses of ground truth to pred 
        r_err = phi
        t_err = tau

    metrics = {
        train_val+'_percent_quat_within_pt1': (r_err < .1).float().mean().item(),
        train_val+'_percent_trans_within_pt1': (t_err < .1).float().mean().item(),
        train_val+'_l1_loss': (geodesic_loss / len(Gs)).detach().item(),
    }

    return geodesic_loss, metrics

