"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
from model.deeplab import Res50_Deeplab, Res101_Deeplab
from model.deeplab_2branch import Res50_Deeplab_2branch, Res101_Deeplab_2branch
from torch.utils import model_zoo

PRETRAINED_MODEL = {
    'resnet-101-caffe': 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/resnet101COCO-41f33a49.pth',
    'resnet-50-caffe':  'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/resnet50-caffe-22edcc82.pth'
}


def model_generator(args):
    # create network
    if args.model == 'DeepLab':
        model = Res101_Deeplab(num_classes=args.num_parts+1)
        if args.restore_from is None:
            args.restore_from = PRETRAINED_MODEL['resnet-101-caffe']
    elif args.model == 'DeepLab50':
        model = Res50_Deeplab(num_classes=args.num_parts+1)
        if args.restore_from is None:
            args.restore_from = PRETRAINED_MODEL['resnet-50-caffe']
    elif args.model == 'DeepLab_2branch':
        model = Res101_Deeplab_2branch(num_classes=args.num_parts+1)
        if args.restore_from is None:
            args.restore_from = PRETRAINED_MODEL['resnet-101-caffe']
    elif args.model == 'DeepLab50_2branch':
        model = Res50_Deeplab_2branch(num_classes=args.num_parts+1)
        if args.restore_from is None:
            args.restore_from = PRETRAINED_MODEL['resnet-50-caffe']
    else:
        print('Model "{}" not exist!'.format(args.model))

    # load pretrained parameters
    if args.restore_from != 'None':
        print('load model from {}'.format(args.restore_from))
        if args.restore_from[:4] == 'http':
            saved_state_dict = model_zoo.load_url(args.restore_from)
        else:
            saved_state_dict = torch.load(args.restore_from)

        # only copy the params that exist in current model (caffe-like)
        new_params = model.state_dict().copy()
        for name, param in new_params.items():
            if name in saved_state_dict and param.size() == saved_state_dict[name].size():
                new_params[name].copy_(saved_state_dict[name])
        model.load_state_dict(new_params)

    return model
