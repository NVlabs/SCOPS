"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import itertools

from tps_stn_pytorch.tps_grid_gen import TPSGridGen
from tps.grid_sample import grid_sample
from torch.autograd import Variable

import random
import numpy as np


def generate_perspective_matrices(batch_size=32, img_sz=128, random_scale=(0.7,1.1), rotate=(10,10,60)):
    angle_random_x = np.random.uniform(size=[batch_size,], low=-rotate[0], high = rotate[0])
    angle_random_x = angle_random_x / 180.0*np.pi

    angle_random_y = np.random.uniform(size=[batch_size,], low=-rotate[1], high = rotate[1])
    angle_random_y = angle_random_y / 180.0*np.pi

    angle_random_z = np.random.uniform(size=[batch_size,], low=-rotate[2], high = rotate[2])
    angle_random_z = angle_random_z / 180.0*np.pi

    ones = np.ones(batch_size).reshape((-1,1))
    zeros = np.zeros(batch_size).reshape((-1,1))

    scale_change = np.random.uniform(size=[batch_size,], low=random_scale[0], high=random_scale[1])

    cos_vx = (np.cos(angle_random_x)).reshape((-1, 1))
    sin_vx = (np.sin(angle_random_x)).reshape((-1, 1))
    rotation_x_mat = np.concatenate([ones, zeros, zeros, zeros, cos_vx, -sin_vx, zeros, sin_vx, cos_vx], axis=1).reshape(batch_size,3,3)
    rotation_x_mat = torch.from_numpy(rotation_x_mat).transpose(1,2)

    cos_vy = (np.cos(angle_random_y)).reshape((-1, 1))
    sin_vy = (np.sin(angle_random_y)).reshape((-1, 1))
    rotation_y_mat = np.concatenate([cos_vy, zeros, sin_vy, zeros, ones, zeros, -sin_vy, zeros, cos_vy], axis=1).reshape(batch_size,3,3)
    rotation_y_mat = torch.from_numpy(rotation_y_mat).transpose(1,2)

    cos_vz = (scale_change*np.cos(angle_random_z)).reshape((-1, 1))
    sin_vz = (scale_change*np.sin(angle_random_z)).reshape((-1, 1))
    rotation_z_mat = np.concatenate([cos_vz, -sin_vz, zeros, sin_vz, cos_vz, zeros, zeros, zeros, ones], axis=1).reshape(batch_size,3,3)
    rotation_z_mat = torch.from_numpy(rotation_z_mat).transpose(1,2)

    R = torch.matmul(rotation_z_mat, torch.matmul(rotation_y_mat, rotation_x_mat))

    return R

def generate_transformer_matrices(batch_size=32, img_sz=128, translate=0.1, random_scale=(0.7,1.1), rotate=60):
    angle_random = np.random.uniform(size=[batch_size,], low=-rotate, high = rotate)
    angle_random = angle_random / 180.0*np.pi

    scale_change = np.random.uniform(size=[batch_size,], low=random_scale[0], high=random_scale[1])

    random_shift_x = np.random.uniform(size=(batch_size,),
                                                low=-translate,
                                                high=translate)
    random_shift_x = random_shift_x.reshape((-1, 1))
    random_shift_y = np.random.uniform(size=(batch_size,),
                                                low=-translate,
                                                high=translate)
    random_shift_y = random_shift_y.reshape((-1, 1))

    img_sz_f = np.float32(img_sz)
    cos_v = (scale_change * np.cos(angle_random)).reshape((-1, 1))
    sin_v = (scale_change * np.sin(angle_random)).reshape((-1, 1))

    zeros = np.zeros_like(cos_v)
    half_ones = np.ones_like(cos_v) * np.float32(+img_sz_f/2.0)
    half_ones_x = random_shift_x * np.float32(img_sz_f/2.0)
    half_ones_y = random_shift_y * np.float32(img_sz_f/2.0)

    theta = np.concatenate([cos_v, -sin_v, half_ones_x, sin_v, cos_v,
                           half_ones_y], axis=1)

    return theta


class RandTPS(nn.Module):
    def __init__(self, width, height, batch_size=16, sigma=0.01, border_padding=False, random_mirror=True, random_scale=(0.7,1.1), mode='affine'):
        super(RandTPS, self).__init__()
        self.width = width
        self.height = height
        self.batch_size = batch_size
        self.sigma = sigma
        self.random_scale = (1.0/random_scale[1], 1.0/random_scale[0]) #inverse scale because it's applied in target->source
        if border_padding:
            self.padding_mode = 'border'
        else:
            self.padding_mode = 'zeros'

        self.rand_mirror = random_mirror
        self.mode = mode

        # set constant tensors
        # 25 control points
        # target_control_points : (25x2) tensor
        self.target_control_points = torch.Tensor(list(itertools.product(
            torch.arange(-1.0, 1.00001, 2.0 / 4),
            torch.arange(-1.0, 1.00001, 2.0 / 4),
        )))

        self.grid = nn.Parameter(torch.zeros(self.batch_size, self.height, self.width, 2))

        self.reset_control_points()

    def reset_control_points(self):

        # source_control_points: apply tps
        source_control_points = self.target_control_points.unsqueeze(0).repeat(self.batch_size, 1, 1)

        # purturb
        source_control_points += torch.Tensor(source_control_points.size()).uniform_(-self.sigma, self.sigma)

        if self.mode == 'affine':
            # apply similarity transform
            theta = generate_transformer_matrices(batch_size=self.batch_size, img_sz=2.0, random_scale=self.random_scale)
            transformer_matrix_reshaped_torch = torch.from_numpy(theta.reshape((-1,2,3)).copy()).type(torch.FloatTensor).transpose(1,2)

            source_control_points = torch.cat((source_control_points, torch.ones(*source_control_points.shape[0:2],1)), dim=2)
            source_control_points = torch.matmul(source_control_points,transformer_matrix_reshaped_torch)
        elif self.mode == 'projective':
            R = generate_perspective_matrices(batch_size=self.batch_size, img_sz=2.0, random_scale=self.random_scale).type(torch.FloatTensor).detach()
            source_control_points = torch.cat((source_control_points, torch.ones(*source_control_points.shape[0:2],1)), dim=2)
            source_control_points = torch.matmul(source_control_points,R)

            # projection
            source_control_points[:,:,0] = source_control_points[:,:,0] / source_control_points[:,:,2]
            source_control_points[:,:,1] = source_control_points[:,:,1] / source_control_points[:,:,2]
            source_control_points = source_control_points[:,:,:2]


        if self.rand_mirror:
            if random.randint(0,1):
                source_control_points[:,:,0] = -source_control_points[:,:,0]

        tps = TPSGridGen(self.height, self.width, self.target_control_points)
        source_coordinate = tps(source_control_points)
        grid = source_coordinate.view(-1, self.height, self.width, 2)
        self.grid.data.copy_(grid)
        self.grid.requires_grad = False


    def forward(self, x, padding_mode=None):
        if padding_mode is None:
            x = grid_sample(x, self.grid, canvas=None, padding_mode=self.padding_mode)
        else:
            x = grid_sample(x, self.grid, canvas=None, padding_mode=padding_mode)

        return x


class ControlTPS(nn.Module):
    def __init__(self, width, height, batch_size=16, border_padding=False):
        super(ControlTPS, self).__init__()
        self.width = width
        self.height = height
        self.batch_size = int(batch_size)

        if border_padding:
            self.padding_mode = 'border'
        else:
            self.padding_mode = 'zeros'

    def set_control_points(self, source_points, target_points, out_h=None, out_w=None):

        # points: Bx5x2
        B,K,_ = source_points.shape

        grids = []

        if out_h == None and out_w == None:
            out_h, out_w = self.height, self.width

        grid = torch.zeros(B, out_h, out_w, 2)

        for b in range(B):
            target_control_points = target_points[b,...].squeeze()
            source_control_points = source_points[b,...].squeeze().unsqueeze(dim=0)

            tps = TPSGridGen(out_h, out_w, target_control_points.detach().cpu())
            tps.cuda()
            source_coordinate = tps(source_control_points)
            grids.append(source_coordinate.view(-1, out_h, out_h, 2))

        self.grid = torch.cat(grids,dim=0)


    def forward(self, x, padding_mode=None):
        if padding_mode is None:
            x = grid_sample(x, self.grid, canvas=None, padding_mode=self.padding_mode)
        else:
            x = grid_sample(x, self.grid, canvas=None, padding_mode=padding_mode)

        return x
