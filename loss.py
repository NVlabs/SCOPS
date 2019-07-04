"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import utils
from model.feature_extraction import featureL2Norm

softmax = nn.Softmax(dim=1)

def get_variance(part_map, x_c, y_c):

    h,w = part_map.shape
    x_map, y_map = utils.get_coordinate_tensors(h,w)

    v_x_map = (x_map - x_c) * (x_map - x_c)
    v_y_map = (y_map - y_c) * (y_map - y_c)

    v_x = (part_map * v_x_map).sum()
    v_y = (part_map * v_y_map).sum()
    return v_x, v_y

def concentration_loss(pred):

    pred_softmax = softmax(pred)[:,1:,:,:]
    B,C,H,W = pred_softmax.shape

    loss = 0
    epsilon = 1e-3
    centers_all = utils.batch_get_centers(pred_softmax)
    for b in range(B):
        centers = centers_all[b]
        for c in range(C):
            # normalize part map as spatial pdf
            part_map = pred_softmax[b,c,:,:] + epsilon # prevent gradient explosion
            k = part_map.sum()
            part_map_pdf = part_map/k
            x_c, y_c = centers[c]
            v_x, v_y = get_variance(part_map_pdf, x_c, y_c)
            loss_per_part = (v_x + v_y)
            loss = loss_per_part + loss
    loss = loss/B
    return loss/B

def semantic_consistency_loss(features, pred, basis):
    # get part maps
    pred_softmax = nn.Softmax(dim=1)(pred)
    part_softmax = pred_softmax[:, 1:, :, :]

    flat_part_softmax = part_softmax.permute(
        0, 2, 3, 1).contiguous().view((-1, part_softmax.size(1)))
    flat_features = features.permute(
        0, 2, 3, 1).contiguous().view((-1, features.size(1)))

    return nn.MSELoss()(torch.mm(flat_part_softmax, basis), flat_features)


def orthonomal_loss(w):
    K, C = w.shape
    w_norm = featureL2Norm(w)
    WWT = torch.matmul(w_norm, w_norm.transpose(0, 1))

    return F.mse_loss(WWT - torch.eye(K).cuda(), torch.zeros(K, K).cuda(), size_average=False)
