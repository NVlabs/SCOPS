"""
Code adapted from: https://github.com/akanazawa/cmr/blob/master/data/cub.py
MIT License

Copyright (c) 2018 akanazawa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

CUB has 11788 images total, for 200 subcategories.
5994 train, 5794 test images.

After removing images that are truncated:
min kp threshold 6: 5964 train, 5771 test.
min_kp threshold 7: 5937 train, 5747 test.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np

import scipy.io as sio
from absl import flags, app

import torch
from torch.utils.data import Dataset

from . import base as base_data
from utils import transformations

# -------------- flags ------------- #
# ---------------------------------- #
if osp.exists('/scratch1/storage'):
    kData = '/scratch1/storage/CUB'
elif osp.exists('/data1/shubhtuls'):
    kData = '/data0/shubhtuls/datasets/CUB'
else:  # Savio
    kData = '/global/home/users/kanazawa/scratch/CUB'

kData = '/xtli-correspondence/CUB_200_2011'
cub_dir = kData
cub_cache_dir = '/xtli-correspondence/nmr/cachedir/cub'

# -------------- Dataset ------------- #
# ------------------------------------ #
class CUBDataset(base_data.BaseDataset):
    '''
    CUB Data loader
    '''

    def __init__(self, opts, filter_key=None):
        super(CUBDataset, self).__init__(opts, filter_key=filter_key)
        #self.data_dir = opts.cub_dir
        #self.data_cache_dir = opts.cub_cache_dir
        self.data_dir = cub_dir
        self.data_cache_dir = cub_cache_dir
        split = 'test'

        self.img_dir = osp.join(self.data_dir, 'images')
        self.anno_path = osp.join(self.data_cache_dir, 'data', '%s_cub_cleaned.mat' % split)
        self.anno_sfm_path = osp.join(self.data_cache_dir, 'sfm', 'anno_%s.mat' % split)

        if not osp.exists(self.anno_path):
            print('%s doesnt exist!' % self.anno_path)
            import pdb; pdb.set_trace()
        self.filter_key = filter_key

        # Load the annotation file.
        print('loading %s' % self.anno_path)
        self.anno = sio.loadmat(
            self.anno_path, struct_as_record=False, squeeze_me=True)['images']
        self.anno_sfm = sio.loadmat(
            self.anno_sfm_path, struct_as_record=False, squeeze_me=True)['sfm_anno']

        self.num_imgs = len(self.anno)
        print('%d images' % self.num_imgs)
        self.kp_perm = np.array([1, 2, 3, 4, 5, 6, 11, 12, 13, 10, 7, 8, 9, 14, 15]) - 1;


#----------- Data Loader ----------#
#----------------------------------#
def data_loader(opts, shuffle=True):
    return base_data.base_loader(CUBDataset, opts.batch_size, opts, filter_key=None, shuffle=shuffle)


def kp_data_loader(batch_size, opts):
    return base_data.base_loader(CUBDataset, batch_size, opts, filter_key='kp')


def mask_data_loader(batch_size, opts):
    return base_data.base_loader(CUBDataset, batch_size, opts, filter_key='mask')


def sfm_data_loader(batch_size, opts):
    return base_data.base_loader(CUBDataset, batch_size, opts, filter_key='sfm_pose')
