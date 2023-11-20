# ------------------------------------------------------------------------
# PowerBEV
# Copyright (c) 2023 Peizheng Li. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from FIERY (https://github.com/wayveai/fiery)
# Copyright (c) 2021 Wayve Technologies Limited. All Rights Reserved.
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialRegressionLoss(nn.Module):
    def __init__(self, norm, ignore_index=255, future_discount=1.0):
        super(SpatialRegressionLoss, self).__init__()
        self.norm = norm
        self.ignore_index = ignore_index
        self.future_discount = future_discount

        if norm == 1:
            self.loss_fn = F.l1_loss
        elif norm == 2:
            self.loss_fn = F.mse_loss
        elif norm == 1.5:
            self.loss_fn = F.smooth_l1_loss
        else:
            raise ValueError(f'Expected norm 1 or 2, but got norm={norm}')
    
    def forward(self, prediction, target):       
        assert len(prediction.shape) == 5, 'Must be a 5D tensor'
        # ignore_index is the same across all channels
        mask = target[:, :, :1] != self.ignore_index
        if mask.sum() == 0:
            return prediction.new_zeros(1)[0].float()

        loss = self.loss_fn(prediction, target, reduction='none')

        # Sum channel dimension
        loss = torch.sum(loss, dim=-3, keepdims=True)

        seq_len = loss.shape[1]
        future_discounts = self.future_discount ** torch.arange(seq_len, device=loss.device, dtype=loss.dtype)
        future_discounts = future_discounts.view(1, seq_len, 1, 1, 1)
        loss = loss * future_discounts

        return loss[mask].mean()


class SegmentationLoss(nn.Module):
    def __init__(self, class_weights, ignore_index=255, use_top_k=False, top_k_ratio=1.0, future_discount=1.0):
        super().__init__()
        self.class_weights = class_weights
        self.ignore_index = ignore_index
        self.use_top_k = use_top_k
        self.top_k_ratio = top_k_ratio
        self.future_discount = future_discount
        
    def forward(self, prediction, target):
        if target.shape[-3] != 1:
            raise ValueError('segmentation label must be an index-label with channel dimension = 1.')
        b, s, c, h, w = prediction.shape

        prediction = prediction.view(b * s, c, h, w)
        target = target.view(b * s, h, w)
        loss = F.cross_entropy(
            prediction,
            target,
            ignore_index=self.ignore_index,
            reduction='none',
            weight=self.class_weights.to(target.device),
        )
        
        loss = loss.view(b, s, h, w)

        future_discounts = self.future_discount ** torch.arange(s, device=loss.device, dtype=loss.dtype)
        future_discounts = future_discounts.view(1, s, 1, 1)
        loss = loss * future_discounts

        loss = loss.view(b, s, -1)
        if self.use_top_k:
            # Penalises the top-k hardest pixels
            k = int(self.top_k_ratio * loss.shape[2])
            loss, _ = torch.sort(loss, dim=2, descending=True)
            loss = loss[:, :, :k]

        return torch.mean(loss)
    
backwarp_tenGrid = {}
backwarp_tenPartial = {}

def warp(im, flow, device='cuda'):
    """
    This function warps an image using 'flow'

    im: tensor of shape (N, C, W, H)
    flow: tensor of shape (N, 2, W, H)
    """
    if str(flow.size()) not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, flow.shape[3]).view(1, 1, 1, flow.shape[3]).expand(flow.shape[0], -1, flow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, flow.shape[2]).view(1, 1, flow.shape[2], 1).expand(flow.shape[0], -1, -1, flow.shape[3])
        backwarp_tenGrid[str(flow.size())] = torch.cat([ tenHorizontal, tenVertical ], 1).to(device)

    if str(flow.size()) not in backwarp_tenPartial:
        backwarp_tenPartial[str(flow.size())] = flow.new_ones([ flow.shape[0], 1, flow.shape[2], flow.shape[3] ])

    # flow is supplied with pixel units, to use grid_sample we scale flow to [-1, 1]
    flow = torch.cat([ flow[:, 0:1, :, :] / ((im.shape[3] - 1.0) / 2.0), flow[:, 1:2, :, :] / ((im.shape[2] - 1.0) / 2.0) ], 1)
    im = torch.cat([ im, backwarp_tenPartial[str(flow.size())] ], 1)

    grid = (backwarp_tenGrid[str(flow.size())] + flow).permute(0, 2, 3, 1)

    out = F.grid_sample(input=im, grid=grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    mask = out[:, -1:, :, :]
    mask[mask > 0.999] = 1.0
    mask[mask < 1.0] = 0.0

    return (out[:, :-1, :, :] * mask).contiguous()

class SelfSupervisedLoss(torch.nn.Module):
    def __init__(self):
        super(SelfSupervisedLoss, self).__init__()
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, pred):
        warpped_feature_map = []
        forward_loss = 0
        for i in range(1, 3):
            warpped_feature_map = warp(pred['raw_bev_feat'][:, i], pred['instance_flow'][:, i])
            forward_loss += self.loss_fn(pred['raw_bev_feat'][:, i - 1], warpped_feature_map)

        return forward_loss