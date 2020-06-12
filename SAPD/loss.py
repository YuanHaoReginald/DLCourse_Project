import numpy as np
import torch
import torch.nn as nn

class IoULoss(nn.Module):
    def forward(self, y_true, y_pred):
        y_true = torch.max(y_true, 0)
        pred_left = y_pred[:, :, 0]
        pred_top = y_pred[:, :, 1]
        pred_right = y_pred[:, :, 2]
        pred_bottom = y_pred[:, :, 3]

        target_left = y_true[:, :, 0]
        target_top = y_true[:, :, 1]
        target_right = y_true[:, :, 2]
        target_bottom = y_true[:, :, 3]
        target_area = (target_left + target_right) * (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)
        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_top, target_top) + torch.min(pred_bottom, target_bottom)
        area_intersection = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersection
        iou = area_intersection / area_union
        iou = torch.clamp(iou, min=1e-6)
        loss = -torch.log(iou)
        loss = loss.mean()
        return loss

class IoULossWithWeightAndMask(nn.Module):
    def forward(self, target, pred):
        y_true, y_pred, weight, mask = target[..., :4], pred, target[..., 4], target[..., 5]
        y_true = torch.max(y_true, 0)
        pred_left = y_pred[:, :, 0]
        pred_top = y_pred[:, :, 1]
        pred_right = y_pred[:, :, 2]
        pred_bottom = y_pred[:, :, 3]

        target_left = y_true[:, :, 0]
        target_top = y_true[:, :, 1]
        target_right = y_true[:, :, 2]
        target_bottom = y_true[:, :, 3]
        target_area = (target_left + target_right) * (target_top + target_bottom)
        masked_target_area = torch.masked_select(target_area, mask)
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)
        masked_pred_area = torch.masked_select(pred_area, mask)
        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_top, target_top) + torch.min(pred_bottom, target_bottom)
        area_intersection = w_intersect * h_intersect
        masked_inter_area = torch.masked_select(area_intersection, mask)
        masked_area_union = masked_target_area + masked_pred_area - masked_inter_area
        masked_weight = torch.masked_select(weight, mask)
        masked_iou = (masked_inter_area + 1e-7) / (masked_area_union+1e-7)
        iou = torch.clamp(masked_iou, min=1e-6)
        loss = -torch.log(iou) * masked_weight
        iou = torch.clamp(iou, min=1e-6)
        loss = -torch.log(iou)
        loss = loss.mean()
        return loss  

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_true, y_pred):
        N = y_pred.size(0)
        C = y_pred.size(1)
        mask = -y_pred + 1
        mask.cuda()
        indices = y_true.eq(1)
        p = y_pred.masked_scatter(indices, mask)
        if self.alpha:
            alpha_mask = y_pred.data.new(N, C).fill_(1 - self.alpha)
            alpha_mask = alpha_mask.masked_fill(indices, self.alpha)
        else:
            alpha_mask = y_pred.data.new(N, C).fill_(1)

        batch_loss = - alpha_mask * torch.pow(p, self.gamma) * torch.log(1 - p)
        loss = batch_loss.sum(dim=1)
        loss = loss.mean()
        return loss

class FocalLossWithWeightAndMask(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLossWithWeightAndMask, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_target, y_pred):
        y_true, soft_weight, mask = y_target[..., :-2], y_target[..., -2], y_target[..., -1]
        N = y_true.size(0)
        C = y_pred.size(1)
        class_mask = -y_pred + 1
        class_mask.cuda()
        indices = y_true.eq(1)
        p = y_pred.masked_scatter(indices, class_mask)
        if self.alpha:
            alpha_mask = y_pred.data.new(N, C).fill_(1 - self.alpha)
            alpha_mask = alpha_mask.masked_fill(indices, self.alpha)
        else:
            alpha_mask = y_pred.data.new(N, C).fill_(1)
        soft_weight = torch.unsqueeze(self.weight, -1)
        batch_loss = - alpha_mask * torch.pow(p, self.gamma) * soft_weight * torch.log(1 - p)
        # compute the normalizer
        num_pos = torch.sum(mask * soft_weight)
        normalizer = torch.max(1.0, num_pos)
        loss = batch_loss.sum(dim=1) / normalizer
        loss = loss.mean()
        return loss