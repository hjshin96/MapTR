# Copyright (c) Megvii Inc. All rights reserved.
import torch
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer
from mmcv.runner import force_fp32
from mmdet3d.models import build_neck
from mmdet.models import build_backbone
from mmdet.models.backbones.resnet import BasicBlock
from mmdet.core import multi_apply
from torch import nn
from torch.cuda.amp.autocast_mode import autocast
from .builder import DEPTHNET
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class _ASPPModule(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):

    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(inplanes, mid_channels, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, mid_channels, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, mid_channels, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        # chgd
        # self.aspp4 = _ASPPModule(inplanes, mid_channels, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        # chgd
        # self.conv1 = nn.Conv2d(int(mid_channels * 5), mid_channels, 1, bias=False)
        self.conv1 = nn.Conv2d(int(mid_channels * 4), mid_channels, 1, bias=False)
        self.bn1 = BatchNorm(mid_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        # x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        # x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x5 = F.interpolate(x5, size=x3.size()[2:], mode='bilinear', align_corners=True)
        # x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = torch.cat((x1, x2, x3, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features,
                 out_features,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()
        out_features = out_features
        hidden_features = hidden_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class SELayer(nn.Module):

    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


@DEPTHNET.register_module()
class BEVDepthNet(nn.Module):

    def __init__(self, in_channels, mid_channels, context_channels,
                 downsample, loss_depth_weight, grid_cfg):
        super(BEVDepthNet, self).__init__()
        self.downsample = downsample
        self.loss_depth_weight = loss_depth_weight
        self.grid_cfg = grid_cfg
        self.D = int(grid_cfg["depth"][1])
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.context_conv = nn.Conv2d(mid_channels, context_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm1d(22)
        self.depth_mlp = Mlp(22, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.context_mlp = Mlp(22, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.depth_conv = nn.Sequential(
            # chgd
            # BasicBlock(mid_channels, mid_channels),
            # BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            ASPP(mid_channels, mid_channels),
            build_conv_layer(cfg=dict(
                type='DCN',
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
                groups=4,
                im2col_step=128,
            )),
            nn.Conv2d(mid_channels, self.D, kernel_size=1, stride=1, padding=0),
        )
        

    def forward(self, x, img_metas):
        camera_awareness = self.create_camera_awareness(img_metas).to(x[0].device)
        mlp_input = self.bn(camera_awareness.reshape(-1, camera_awareness.shape[-1]))
        mlvl = len(x)
        B, N, C, H, W = x[0].shape
        depth_feats = []
        context_feats = []              # TODO: multi-level feature 구현?
        x = x[0].reshape(B * N, C, H, W)
        x = self.reduce_conv(x)
        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x, context_se)
        context = self.context_conv(context)
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se)
        depth = self.depth_conv(depth)
        context_feats.append(context.view(B, N, -1, H, W))
        depth_feats = depth.view(B, N, self.D, H, W)
        return depth_feats, context_feats
    
    def create_camera_awareness(self, img_metas):
        tmp = {
                'camera_intrinsics' : [],
                'lidar2img' : [],
                'img_aug_matrix' : [],
                'camera2ego' : []
                }
        
        for img_meta in img_metas:
            tmp['camera_intrinsics'].append(img_meta['camera_intrinsics'])        # [bs, num_cam, 4, 4]
            tmp['img_aug_matrix'].append(img_meta['img_aug_matrix'])
            tmp['camera2ego'].append(img_meta['camera2ego'])
        num_cam = len(tmp['camera_intrinsics'][0])
        bs = len(img_metas)
        tmp['camera_intrinsics'] = torch.as_tensor(tmp['camera_intrinsics'], dtype=torch.float32)
        tmp['img_aug_matrix'] = torch.as_tensor(tmp['img_aug_matrix'], dtype=torch.float32)
        tmp['camera2ego'] = torch.as_tensor(tmp['camera2ego'], dtype=torch.float32)[:,:,:3,:]
        
        info = torch.cat(
            [
                torch.stack(
                    [
                        tmp['camera_intrinsics'][..., 0, 0],
                        tmp['camera_intrinsics'][..., 1, 1],
                        tmp['camera_intrinsics'][..., 0, 2],
                        tmp['camera_intrinsics'][..., 1, 2],
                        tmp['img_aug_matrix'][..., 0, 0],
                        tmp['img_aug_matrix'][..., 1, 1],
                        tmp['img_aug_matrix'][..., 2, 2],
                        tmp['img_aug_matrix'][..., 1, 0],
                        tmp['img_aug_matrix'][..., 1, 1],
                        tmp['img_aug_matrix'][..., 1, 3],
                        # tmp['lidar2ego'][..., 0, 0],
                        # tmp['lidar2ego'][..., 0, 1],
                        # tmp['lidar2ego'][..., 1, 0],
                        # tmp['lidar2ego'][..., 1, 1],
                        # tmp['lidar2ego'][..., 2, 2],
                    ],
                    dim=-1,
                ),
                tmp['camera2ego'].view(bs, num_cam, -1),
            ],
            -1,
        )
        return info
    
    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N, H // self.downsample, self.downsample, W // self.downsample, self.downsample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0, 1e5 * torch.ones_like(gt_depths), gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample, W // self.downsample)

        gt_depths = (gt_depths - (self.grid_cfg['depth'][0] - self.grid_cfg['depth'][2])) / self.grid_cfg['depth'][2]
        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0), gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:, 1:]
        return gt_depths.float()
    
    @force_fp32()
    def get_depth_loss(self, depth_labels, depth_feats):
        
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds_prob = depth_feats.softmax(dim=2).permute(0, 1, 3, 4, 2).contiguous().view(-1, self.D)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds_prob[fg_mask]
        
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(depth_preds, depth_labels, reduction='none',).sum() / max(1.0, fg_mask.sum())
        # depth_loss = F.binary_cross_entropy(depth_preds, depth_labels, reduction='none',).sum() / max(1.0, fg_mask.sum())
        return {'loss_depth' : self.loss_depth_weight * depth_loss}
    
    def loss(self, depth_labels, depth_feats):
        losses = multi_apply(self.get_depth_loss, depth_labels, depth_feats)
        return losses
        
