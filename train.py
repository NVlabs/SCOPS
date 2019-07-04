"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import argparse
import json
import os
import os.path as osp
import pickle
import random
import sys
from argparse import Namespace

import numpy as np

import cv2
import scipy.misc
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

import scops_trainer
from dataset.dataset_factory import dataset_generator
from model.model_factory import model_generator


# solve potential deadlock https://github.com/pytorch/pytorch/issues/1355
cv2.setNumThreads(0)


EXP_NAME = 'SCOPS-Test'
MODEL = 'DeepLab'
BATCH_SIZE = 10
NUM_WORKERS = 4
DATASET = 'PASCAL'
DATA_DIRECTORY = ''
DATA_LIST_PATH = ''
INPUT_SIZE = '128,128'
LEARNING_RATE = 1e-5
MOMENTUM = 0.9
NUM_CLASSES = 6
NUM_STEPS = 2000
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = None
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 5000
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005
CLIP_GRAD_NORM = 5
VIS_TNTERVAL = 100

TPS_SIGMA = 0.01
RAND_SCALE_LOW = 0.7
RAND_SCALE_HIGH = 1.1

NUM_PARTS = 10

LAMBDA_CON = 1e-1
LAMBDA_EQV = 10.0
LAMBDA_SC = 0.1


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="SCOPS: Self-supervised Co-part Segmentation")

    # load args from json files
    parser.add_argument("-f", "--arg-file", type=str, default=None,
                        help="load args from json file")

    # Exp
    parser.add_argument("--exp-name", type=str, default=MODEL,
                        help="Experiment name. Default: Part-Test")
    # Model/Data description
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab/DeepLab_2branch")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--dataset", type=str, default=DATASET,
                        help="Dataset selection")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--celeba-th", type=float, default=0.0,
                        help="iou_threshold for celebAWild")

    # Data Augmentation
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")

    # Training hyper parameters
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--clip-gradients", type=float, default=CLIP_GRAD_NORM,
                        help="Clip gradients norm. Default:5.")

    # constraints weighting
    parser.add_argument("--lambda-con", type=float, default=LAMBDA_CON,
                        help="weighting parameter for concentration")
    parser.add_argument("--lambda-eqv", type=float, default=LAMBDA_EQV,
                        help="weighting parameter for equivariance")
    parser.add_argument("--lambda-lmeqv", type=float, default=LAMBDA_EQV,
                        help="weighting parameter for equivariance")

    # Equivariance setting
    parser.add_argument("--tps-mode", type=str, default='affine',
                        help="tps mode: affine/projective")
    parser.add_argument("--tps-sigma", type=float, default=TPS_SIGMA,
                        help="peturbation of tps in equivariance loss")
    parser.add_argument("--eqv-random-mirror", action="store_true",
                        help="Whether to use random mirror in equvariance.")
    parser.add_argument("--eqv-border-padding", action="store_true",
                        help="Whether to use border padding in equvariance.")
    parser.add_argument("--random-scale-low", type=float, default=RAND_SCALE_LOW,
                        help="lower bound of random scaling.")
    parser.add_argument("--random-scale-high", type=float, default=RAND_SCALE_HIGH,
                        help="higher bound of random scaling.")

    # part training config
    parser.add_argument("--ignore-saliency-fg", action="store_true",
                        help="Whether to ignore saliency foreground and only enforce BG")
    parser.add_argument("--ignore-small-parts", action="store_true",
                        help="Whether to ignore small parts.")
    parser.add_argument("--center-crop", action="store_true",
                        help="Whether to crop center (MAFL).")
    parser.add_argument("--self-ref-coord", action="store_true",
                        help="Whether to use self-referenced centroid.")
    parser.add_argument("--kp-threshold", type=int, default=6,
                        help="maximum number of missing keypoints/landmarks")

    # Semantic Consistency config
    parser.add_argument("--num-parts", type=int, default=NUM_PARTS,
                        help="Number of parts")
    parser.add_argument("--lambda-sc", type=float, default=LAMBDA_SC,
                        help="weighting parameter for semantic consistency constraint")
    parser.add_argument("--learning-rate-w", type=float, default=1e-3,
                        help="learning rate for DFF basis")
    parser.add_argument("--ref-net", type=str, default='vgg19',
                        help="reference feature network. default: vgg19")
    parser.add_argument("--ref-layer", type=str, default='relu5_4',
                        help="default: vgg19")
    parser.add_argument("--ref-norm", action="store_true",
                        help="normalize reference feature map")
    parser.add_argument("--lambda-orthonamal", type=float, default=1e2,
                        help="weighting parameter for DFF orthonormal loss")

    parser.add_argument("--detach-k", action="store_true",
                        help="detach k")
    parser.add_argument("--no-sal-masking", action="store_true",
                        help="disable saliency constraint")

    parser.add_argument("--restore-part-basis", type=str, default='',
                        help="load part basis weights")

    # Save/visualization
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--vis-interval", type=int, default=VIS_TNTERVAL,
                        help="visualization interval.")
    parser.add_argument("--tb-dir", type=str, default='tb_logs',
                        help="tensorbard dir.")

    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    return parser.parse_args()


def main():

    args = get_arguments()
    args_dict = vars(args)

    if args.arg_file is not None:
        with open(args.arg_file, 'r') as f:
            arg_str = f.read()
            file_args = json.loads(arg_str)
            args_dict.update(file_args)
            args = Namespace(**args_dict)

    args_str = '{}'.format(json.dumps(vars(args), sort_keys=False, indent=4))
    print(args_str)

    if not os.path.exists(os.path.join(args.snapshot_dir, args.exp_name)):
        os.makedirs(os.path.join(args.snapshot_dir, args.exp_name))

    # save args to file
    with open(os.path.join(args.snapshot_dir, args.exp_name, 'exp_args.json'), 'w') as f:
        print(args_str, file=f)

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    args.input_size = input_size

    cudnn.enabled = True
    gpu = args.gpu

    # create network
    model = model_generator(args)
    model.train()
    model.cuda(args.gpu)
    cudnn.benchmark = True

    # Initialize SCOPS trainer
    trainer = scops_trainer.SCOPSTrainer(args, model)

    train_dataset = dataset_generator(args)
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)
    trainloader_iter = enumerate(trainloader)


    for i_iter in range(args.num_steps):

        try:
            _, batch = trainloader_iter.__next__()
        except:
            trainloader_iter = enumerate(trainloader)
            _, batch = trainloader_iter.__next__()

        trainer.step(batch, i_iter)

        if i_iter >= args.num_steps - 1:
            print('save model ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir,
                                                    args.exp_name, 'model_' + str(args.num_steps) + '.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir,
                                                    args.exp_name, 'model_' + str(i_iter) + '.pth'))


if __name__ == '__main__':
    main()
