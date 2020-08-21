"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import argparse
import scipy
from scipy import ndimage
import scipy.io
import imageio as iio
import cv2
import numpy as np
import sys
from collections import OrderedDict
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo

from model.model_factory import model_generator
from loss import softmax
from utils import utils
from visualize import Batch_Draw_Landmarks, Batch_Get_Centers, Batch_Draw_GT_Landmarks

from PIL import Image
import matplotlib.pyplot as plt

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

MODEL = 'DeepLab50_2branch'
DATASET = 'cub'
DATA_DIRECTORY = ''
DATA_LIST_PATH = ''
NUM_PARTS = 4
RESTORE_FROM = ''
SAVE_DIRECTORY = 'results'
INPUT_SIZE='256,256'


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="VOC evaluation script")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab/DRN")
    parser.add_argument("--dataset", type=str, default=DATASET, help = "dataset : PASCAL/MAFL")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--split", type=str, default='test',
                        help="train/val/test split.")
    parser.add_argument("--lm-count", type=int, default=5,
                        help="how many landmarks")
    parser.add_argument("--num-parts", type=int, default=NUM_PARTS,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIRECTORY,
                        help="Directory to store results")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--crf", action="store_true",
                        help="crf")
    parser.add_argument("--save-viz", action="store_true",
                        help="save visualization")
    return parser.parse_args()

def read_frame(fpath, bbox):
    img = iio.imread(fpath)
    # crop img w.r.t bbox
    img = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]), :]
    img = np.asarray(img, np.float32)
    img -= IMG_MEAN
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).unsqueeze(0)

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()

    gpu0 = args.gpu

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    save_part_dir = os.path.join(args.save_dir, 'part_map')
    if not os.path.exists(save_part_dir):
        os.makedirs(save_part_dir)

    save_overlay_dir = os.path.join(args.save_dir, 'part_overlay')
    if not os.path.exists(save_overlay_dir):
        os.makedirs(save_overlay_dir)

    save_part_dcrf_dir = os.path.join(args.save_dir, 'part_map_dcrf')
    if not os.path.exists(save_part_dcrf_dir):
        os.makedirs(save_part_dcrf_dir)

    save_dcrf_overlay_dir = os.path.join(args.save_dir, 'part_dcrf_overlay')
    if not os.path.exists(save_dcrf_overlay_dir):
        os.makedirs(save_dcrf_overlay_dir)

    save_lm_dir = os.path.join(args.save_dir, 'landmarks')
    if not os.path.exists(save_lm_dir):
        os.makedirs(save_lm_dir)

    save_seg_dir = os.path.join(args.save_dir, 'seg')
    if not os.path.exists(save_seg_dir):
        os.makedirs(save_seg_dir)

    save_prob_dir = os.path.join(args.save_dir, 'prob')
    if not os.path.exists(save_prob_dir):
        os.makedirs(save_prob_dir)

    save_dcrf_prob_dir = os.path.join(args.save_dir, 'dcrf_prob')
    if not os.path.exists(save_dcrf_prob_dir):
        os.makedirs(save_dcrf_prob_dir)


    # create network
    model = model_generator(args)
    #model.load_state_dict(torch.load("snapshots_CelebA/SCOPS_K8_retrain/model_100000.pth"))
    model.load_state_dict(torch.load(args.restore_from))

    model.eval()
    model.cuda(gpu0)

    if args.dataset == 'CelebAWild':
        from dataset.celeba_wild_dataset import CelebAWildDataset
        dataset = CelebAWildDataset
        testloader = data.DataLoader(dataset(args.data_dir, args.data_list, crop_size=input_size,
		                scale=False, mirror=False, mean=IMG_MEAN,
		                center_crop=False, ignore_saliency_fg=False, iou_threshold=0.3),
		                batch_size=1, shuffle=False, pin_memory=True)
    elif args.dataset == 'cub':
        from dataset import cub
        args.batch_size = 1
        testloader = cub.data_loader(args)
    elif args.dataset == 'ImageNet':
        from dataset import imagenet as imnet_data
        args.batch_size = 1
        testloader = imnet_data.imnet_dataloader(args)
    elif args.dataset == 'p3d':
        from dataset import p3d as p3d_data
        args.batch_size = 1
        testloader = p3d_data.p3d_dataloader(args)
    else:
        print('Not supported dataset {}'.format(args.dataset))

    interp = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
    colorize = utils.Colorize(args.num_parts+1)
    N = len(testloader)

    landmarks = np.zeros((N, args.num_parts,2))
    #landmarks_gt = np.zeros((N,args.lm_count,2))

    with torch.no_grad():
        for index, batch in enumerate(testloader):
            """
            import pickle
            with open("/xtli-correspondence/Data/trim/bboxes.npy", 'rb') as f:
                bboxes = pickle.load(f)
            bbox = np.squeeze(bboxes[0])
            fpath = "/xtli-correspondence/Data/trim/output_00001.jpg"
            fpath = "/xtli-correspondence/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0039_796132.jpg"
            bbox = np.array([122, 128, 481, 379])
            img = read_frame(fpath, bbox)
            batch = {}
            batch['img'] = img
            """
            if index % 100 == 0:
                path_split = args.save_dir.split('/')
                print('{} processd: {}/{}'.format(index, path_split[-4], path_split[-3]))
            image = batch['img']
            #label = batch['mask']
            img_path = batch['img_path']
            tmp = img_path[0].split("/")
            img_folder = tmp[4]
            img_nm = tmp[5].replace(".jpg", ".png")

            size = input_size
            output = model(image.cuda(gpu0))
            output_raw = interp(output[2])
            #permute = [0, 1, 4, 3, 2]
            #import pdb; pdb.set_trace()
            #output_raw = output_raw[:, permute, :, :]

            lms = Batch_Get_Centers(output_raw)
            landmarks[index,:,:] = lms

            if args.save_viz:
                mean_tensor = torch.tensor(IMG_MEAN).float().expand(int(size[1]), int(size[0]), 3).transpose(0,2)
                imgs_viz = torch.clamp(image+mean_tensor, 0.0, 255.0)
                #landmark visualization
                lms_viz = Batch_Draw_GT_Landmarks(imgs_viz, output_raw, lms)

                output = softmax(output_raw)
                # normalize part
                output /= output.max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0]
                output[:,0,:,:] = 0.1

                output = output.cpu().data[0].numpy()

                #output = output[:,:size[0],:size[1]]
                #gt = np.asarray(label[0].numpy()[:size[0],:size[1]], dtype=np.int)

                output_np = output.transpose(1,2,0)
                output_np = np.asarray(np.argmax(output_np, axis=2), dtype=np.int)

                filename = os.path.join(save_prob_dir, '{}/{}'.format(img_folder, img_nm.replace("png","pth")))
                file_dir = os.path.dirname(filename)
                if not os.path.exists(file_dir):
                    os.makedirs(file_dir)

                #save probility map before softmax
                torch.save(output_raw.cpu(), filename)

                #Image.fromarray(output_np, 'P').save(filename)
                filename = os.path.join(save_seg_dir, '{}/{}'.format(img_folder, img_nm))
                file_dir = os.path.dirname(filename)
                if not os.path.exists(file_dir):
                    os.makedirs(file_dir)
                pil_image = Image.fromarray(output_np.astype(dtype=np.uint8))
                pil_image.save(filename, 'PNG')

                seg_viz = colorize(output_np)
                filename = os.path.join(save_part_dir, '{}/{}'.format(img_folder, img_nm))
                file_dir = os.path.dirname(filename)
                if not os.path.exists(file_dir):
                    os.makedirs(file_dir)

                Image.fromarray(seg_viz.squeeze().transpose(1, 2, 0), 'RGB').save(filename)

                seg_overlay_viz = (imgs_viz.numpy()*0.8+ seg_viz*0.7).clip(0,255.0).astype(np.uint8)
                filename = os.path.join(save_overlay_dir, '{}/{}'.format(img_folder, img_nm))
                file_dir = os.path.dirname(filename)
                if not os.path.exists(file_dir):
                    os.makedirs(file_dir)

                Image.fromarray(seg_overlay_viz.squeeze().transpose(1, 2, 0), 'RGB').save(filename)

                if args.crf:
                    output_dcrf_prob = utils.denseCRF(imgs_viz.numpy().squeeze().transpose(1,2,0).astype(np.uint8).copy(), output)

                    filename = os.path.join(save_dcrf_prob_dir, '{}/{}'.format(img_folder, img_nm.replace(".png", ".npy").replace(".JPEG", ".npy")))
                    file_dir = os.path.dirname(filename)
                    if not os.path.exists(file_dir):
                        os.makedirs(file_dir)
                    np.save(filename, output_dcrf_prob)

                    output_dcrf = np.asarray(np.argmax(output_dcrf_prob, axis=2), dtype=np.int)
                    seg_dcrf_viz = colorize(output_dcrf)

                    filename = os.path.join(save_part_dcrf_dir, '{}/{}'.format(img_folder, img_nm))
                    file_dir = os.path.dirname(filename)
                    if not os.path.exists(file_dir):
                        os.makedirs(file_dir)
                    Image.fromarray(seg_dcrf_viz.squeeze().transpose(1, 2, 0), 'RGB').save(filename)

                    seg_dcrf_overlay_viz = (imgs_viz.numpy()*0.8+ seg_dcrf_viz*0.7).clip(0,255.0).astype(np.uint8)
                    filename = os.path.join(save_dcrf_overlay_dir, '{}/{}'.format(img_folder, img_nm))
                    file_dir = os.path.dirname(filename)
                    if not os.path.exists(file_dir):
                        os.makedirs(file_dir)
                    Image.fromarray(seg_dcrf_overlay_viz.squeeze().transpose(1, 2, 0), 'RGB').save(filename)


                filename_lm = os.path.join(save_lm_dir, '{}/{}'.format(img_folder, img_nm))
                file_dir = os.path.dirname(filename_lm)
                if not os.path.exists(file_dir):
                    os.makedirs(file_dir)
                Image.fromarray(lms_viz[0,:,:,:].transpose(1, 2, 0), 'RGB').save(filename_lm)

    #np.save(os.path.join(args.save_dir, 'pred_kp.npy'), landmarks)
    #np.save(os.path.join(args.save_dir, 'gt_kp.npy'), landmarks_gt)


if __name__ == '__main__':
    main()
