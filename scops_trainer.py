"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import loss
from utils import utils
from model.feature_extraction import FeatureExtraction, featureL2Norm
from torchvision import transforms
from tps.rand_tps import RandTPS
from visualize import Visualizer

IMG_MEAN = np.array((104.00698793, 116.66876762,
                     122.67891434), dtype=np.float32)


class PartBasisGenerator(nn.Module):
    def __init__(self, feature_dim, K, normalize=False):
        super(PartBasisGenerator, self).__init__()
        self.w = nn.Parameter(
            torch.abs(torch.cuda.FloatTensor(K, feature_dim).normal_()))
        self.normalize = normalize

    def forward(self, x=None):
        out = nn.ReLU()(self.w)
        if self.normalize:
            return featureL2Norm(out)
        else:
            return out


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter)**(power))


def adjust_learning_rate(optimizer, i_iter, args):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


class SCOPSTrainer(object):

    def __init__(self, args, model):
        self.args = args
        self.model = model

        # Initialize spatial/color transform for Equuivariance loss.
        self.tps = RandTPS(args.input_size[1], args.input_size[0],
                           batch_size=args.batch_size,
                           sigma=args.tps_sigma,
                           border_padding=args.eqv_border_padding,
                           random_mirror=args.eqv_random_mirror,
                           random_scale=(args.random_scale_low,
                                         args.random_scale_high),
                           mode=args.tps_mode).cuda(args.gpu)

        # Color Transorm.
        self.cj_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.2),
            transforms.ToTensor(), ])

        # KL divergence loss for equivariance
        self.kl = nn.KLDivLoss().cuda(args.gpu)

        # loss/ bilinear upsampling
        self.interp = nn.Upsample(
            size=(args.input_size[1], args.input_size[0]), mode='bilinear', align_corners=True)

        # Initialize feature extractor and part basis for the semantic consistency loss.
        self.zoo_feat_net = FeatureExtraction(
            feature_extraction_cnn=args.ref_net, normalization=args.ref_norm, last_layer=args.ref_layer)
        self.zoo_feat_net.eval()

        self.part_basis_generator = PartBasisGenerator(self.zoo_feat_net.out_dim,
                                                       args.num_parts, normalize=args.ref_norm)
        self.part_basis_generator.cuda(args.gpu)
        self.part_basis_generator.train()

        if args.restore_part_basis != '':
            self.part_basis_generator.load_state_dict(
                {'w': torch.load(args.restore_part_basis)})

        # Initialize optimizers.
        self.optimizer_seg = optim.SGD(self.model.optim_parameters(args),
                                       lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        self.optimizer_seg.zero_grad()

        self.optimizer_sc = optim.SGD(self.part_basis_generator.parameters(
        ), lr=args.learning_rate_w, momentum=args.momentum, weight_decay=args.weight_decay)
        self.optimizer_sc.zero_grad()

        # visualizor
        self.viz = Visualizer(args)

    def step(self, batch, current_step):
        loss_con_value = 0
        loss_eqv_value = 0
        loss_lmeqv_value = 0
        loss_sc_value = 0
        loss_orthonamal_value = 0

        self.optimizer_seg.zero_grad()
        self.optimizer_sc.zero_grad()
        adjust_learning_rate(self.optimizer_seg, current_step, self.args)

        images_cpu = batch['img']
        labels = batch['saliency'] if 'saliency' in batch.keys() else None
        edges = batch['edge'] if 'edge' in batch.keys() else None
        gts = batch['gt'] if 'gt' in batch.keys() else None

        landmarks = batch['landmarks'] if 'landmarks' in batch.keys() else None
        bbox = batch['bbox'] if 'bbox' in batch.keys() else None

        images = images_cpu.cuda(self.args.gpu)
        feature_instance, feature_part, pred_low = self.model(images)
        pred = self.interp(pred_low)

        # prepare for torch model_zoo models images
        zoo_mean = np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
        zoo_var = np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))
        images_zoo_cpu = (images_cpu.numpy() +
                          IMG_MEAN.reshape((1, 3, 1, 1))) / 255.0
        images_zoo_cpu -= zoo_mean
        images_zoo_cpu /= zoo_var

        images_zoo_cpu = torch.from_numpy(images_zoo_cpu)
        images_zoo = images_zoo_cpu.cuda(self.args.gpu)

        with torch.no_grad():

            zoo_feats = self.zoo_feat_net(images_zoo)
            zoo_feat = torch.cat([self.interp(zoo_feat)
                                  for zoo_feat in zoo_feats], dim=1)
            # saliency masking
            if not self.args.no_sal_masking and labels is not None:
                zoo_feat = zoo_feat * \
                    labels.unsqueeze(dim=1).expand_as(
                        zoo_feat).cuda(self.args.gpu)

        loss_sc = loss.semantic_consistency_loss(
            features=zoo_feat, pred=pred, basis=self.part_basis_generator())
        loss_sc_value += self.args.lambda_sc * loss_sc.data.cpu().numpy()

        # orthonomal_loss
        loss_orthonamal = loss.orthonomal_loss(self.part_basis_generator())
        loss_orthonamal_value += self. args.lambda_orthonormal * \
            loss_orthonamal.data.cpu().numpy()

        # Concentratin Loss
        loss_con = loss.concentration_loss(pred)
        loss_con_value += self.args.lambda_con * loss_con.data.cpu().numpy()

        # Equivariance Loss
        images_cj = torch.from_numpy(
            ((images_cpu.numpy() + IMG_MEAN.reshape((1, 3, 1, 1))) / 255.0).clip(0, 1.0))
        for b in range(images_cj.shape[0]):
            images_cj[b] = torch.from_numpy(self.cj_transform(
                images_cj[b]).numpy() * 255.0 - IMG_MEAN.reshape((1, 3, 1, 1)))
        images_cj = images_cj.cuda()

        self.tps.reset_control_points()
        images_tps = self.tps(images_cj)
        feature_instance_tps, feature_part_tps, pred_low_tps = self.model(
            images_tps)
        pred_tps = self.interp(pred_low_tps)
        pred_d = pred.detach()
        pred_d.requires_grad = False
        # no padding in the prediction space
        pred_tps_org = self.tps(pred_d, padding_mode='zeros')

        loss_eqv = self.kl(F.log_softmax(pred_tps, dim=1),
                           F.softmax(pred_tps_org, dim=1))
        loss_eqv_value += self.args.lambda_eqv * loss_eqv.data.cpu().numpy()

        centers_tps = utils.batch_get_centers(nn.Softmax(dim=1)(pred_tps)[:, 1:, :, :])
        pred_tps_org_dif = self.tps(pred, padding_mode='zeros')
        centers_tps_org = utils.batch_get_centers(nn.Softmax(
            dim=1)(pred_tps_org_dif)[:, 1:, :, :])

        loss_lmeqv = F.mse_loss(centers_tps, centers_tps_org)
        loss_lmeqv_value += self.args.lambda_lmeqv * loss_lmeqv.data.cpu().numpy()

        # visualization

        if current_step % self.args.vis_interval == 0:
            with torch.no_grad():
                pred_softmax = nn.Softmax(dim=1)(pred)
                part_softmax = pred_softmax[:, 1:, :, :]
                # normalize
                part_softmax /= part_softmax.max(dim=3, keepdim=True)[
                    0].max(dim=2, keepdim=True)[0]
                self.viz.vis_images(current_step, images_cpu, images_tps.cpu(
                ), labels, edges, IMG_MEAN, pred.float())
                self.viz.vis_part_heatmaps(
                    current_step, part_softmax, threshold=0.1, prefix='pred')

                if landmarks is not None:
                    self.viz.vis_landmarks(current_step, images_cpu,
                                           IMG_MEAN, pred, landmarks)
                if bbox is not None:
                    self.viz.vis_bboxes(current_step, bbox)

                print('saving part basis')
                torch.save({'W': self.part_basis_generator().detach().cpu(), 'W_state_dict': self.part_basis_generator.state_dict()},
                           osp.join(self.args.snapshot_dir, self.args.exp_name, 'BASIS_' + str(current_step) + '.pth'))

            self.viz.vis_losses(current_step, [self.part_basis_generator.w.mean(), self.part_basis_generator.w.std()], [
                'part_basis_mean', 'part_basis_std'])

        # sum all loss terms
        total_loss = self.args.lambda_con * loss_con \
            + self.args.lambda_eqv * loss_eqv \
            + self.args.lambda_lmeqv * loss_lmeqv \
            + self.args.lambda_sc * loss_sc \
            + self.args.lambda_orthonormal * loss_orthonamal

        total_loss.backward()

        # visualize loss curves
        self.viz.vis_losses(current_step,
                            [loss_con_value, loss_eqv_value, loss_lmeqv_value,
                             loss_sc_value, loss_orthonamal_value],
                            ['loss_con', 'loss_eqv', 'loss_lmeqv', 'loss_sc', 'loss_orthonamal'])
        # clip gradients
        nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_gradients)
        self.optimizer_seg.step()

        nn.utils.clip_grad_norm_(
            self.part_basis_generator.parameters(), self.args.clip_gradients)
        self.optimizer_sc.step()

        print('exp = {}'.format(osp.join(self.args.snapshot_dir, self.args.exp_name)))
        print(('iter = {:8d}/{:8d}, ' +
               'loss_con = {:.3f}, ' +
               'loss_eqv = {:.3f}, ' +
               'loss_lmeqv = {:.3f}, ' +
               'loss_sc = {:.3f}, ' +
               'loss_orthonamal = {:.3f}')
              .format(current_step, self.args.num_steps,
                      loss_con_value,
                      loss_eqv_value,
                      loss_lmeqv_value,
                      loss_sc_value,
                      loss_orthonamal_value))
