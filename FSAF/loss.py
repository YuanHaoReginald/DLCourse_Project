import torch
import torch.nn as nn
import torch.nn.functional as F

class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()
    def forward(self, pred_box, true_box):
        x1_pred, x2_pred, y1_pred, y2_pred = pred_box[:,0], pred_box[:,1], pred_box[:,2], pred_box[:,3]
        x1_true, x2_true, y1_true, y2_true = true_box[:,0], true_box[:,1], true_box[:,2], true_box[:,3]
        inter_x1 = torch.max(x1_pred, x1_true)
        inter_y1 = torch.max(y1_pred, y1_true)
        inter_x2 = torch.min(x2_pred, x2_true)
        inter_y2 = torch.min(y2_pred, y2_true)
        inter_area = torch.clamp(inter_x2 - inter_x1 + 1, min = 0) * torch.clamp(inter_y2 - inter_y1, min=0)
        pred_area = (y2_pred - y1_pred + 1) * (x2_pred - x1_pred + 1)
        true_area = (y2_true - y1_true + 1) * (x2_true - x2_pred + 1)
        iou = inter_area / (pred_area + true_area - inter_area + 1e-5)
        loss = -torch.log(iou)
        #loss = loss.mean()
        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)

        class_mask = inputs.data.new(N, C).fill(1)
        indicies = targets.eq(1)
        class_mask = class_mask.masked_fill(indicies, -1)

        p = class_mask * targets
        
        class_mask2 = inputs.data.new(N, C).fill(0)
        class_mask2 = class_mask2.masked_fill(indicies, 1)

        p = p + class_mask2

        alpha_mask = None
        if self.alpha:
            alpha_mask = inputs.data.new(N, C).fill(1 - self.alpha)
            alpha_mask = alpha_mask.masked_fill(indicies, self.alpha)
        else:
            alpha_mask = inputs.data.new(N, C).fill(1)
        batch_loss = - alpha_mask * torch.pow(p, self.gamma) * p.log()
        batch_loss = batch_loss.sum(axis=1)
        #loss = batch_loss.mean()
        return batch_loss
