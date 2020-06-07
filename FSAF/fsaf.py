import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import FocalLoss, IoULoss
from utils import trim_zeros_graph, prop_box_graph
import numpy as np
from branch import *
import math

def level_select(cls_pred, regr_pred, gt_boxes, feature_shapes, strides, pos_scale=0.2):
    MAX_NUM_GT_BOXES = gt_boxes.shape[0]
    gt_boxes = gt_boxes[gt_boxes[:, 4] != -1]
    gt_labels = gt_boxes[:, 4].int()
    gt_boxes = gt_boxes[:, :4]
    focal_loss = FocalLoss(alpha=0.25)
    iou_loss = IoULoss()

    gt_boxes, non_zeros = trim_zeros_graph(gt_boxes)
    num_gt_boxes = gt_boxes.shape[0]
    gt_labels = gt_labels[non_zeros]
    level_losses = []
    total_loss = 0
    for level_id in range(len(strides)):
        stride = strides[level_id]
        fh = math.ceil(feature_shapes[level_id][0])
        fw = math.ceil(feature_shapes[level_id][1])
        fa = torch.prod(torch.ceil(feature_shapes), dim=1)
        start_index = torch.sum(fa[:level_id]).int()
        start_index = start_index.item()
        end_index = start_index + fh * fw
        cls_pred_i = cls_pred[start_index:end_index, :].reshape(fh, fw, -1)
        regr_pred_i = regr_pred[start_index:end_index, :].reshape(fh, fw, -1)
        proj_boxes = gt_boxes / stride
        x1, y1, x2, y2 = prop_box_graph(proj_boxes, pos_scale, fw, fh)
        level_loss = []
        for i in range(num_gt_boxes):
            x1_ = x1[i]
            y1_ = y1[i]
            x2_ = x2[i]
            y2_ = y2[i]
            gt_box = gt_boxes[i]
            gt_label = gt_labels[i]
            locs_cls_pred_i = cls_pred_i[y1_:y2_, x1_:x2_, :].reshape(-1, cls_pred_i.shape[2])
            locs_a, num_class = locs_cls_pred_i.shape
            locs_cls_true_i = torch.zeros(num_class).cuda()
            locs_cls_true_i[gt_label] = 1
            locs_cls_true_i = locs_cls_true_i.repeat(locs_a, 1)
            loss_cls = focal_loss(locs_cls_pred_i, locs_cls_true_i)
            locs_regr_pred_i = regr_pred_i[y1_:y2_, x1_:x2_, :].reshape(-1, 4)
            shift_x = (np.arange(x1_.cpu(), x2_.cpu()) + 0.5) * stride
            shift_y = (np.arange(y1_.cpu(), y2_.cpu()) + 0.5) * stride

            shift_x, shift_y = np.meshgrid(shift_x, shift_y)
            shifts = np.vstack((
                shift_x.ravel(), shift_y.ravel(),
                shift_x.ravel(), shift_y.ravel()
            )).transpose()

            shifts = torch.from_numpy(shifts).cuda()
            l = shifts[:, 0] - gt_box[0]
            t = shifts[:, 1] - gt_box[1]
            r = gt_box[2] - shifts[:, 2]
            b = gt_box[3] - shifts[:, 3]
            locs_regr_true_i = torch.stack([l, t, r, b], dim=1)
            locs_regr_true_i /= 4.0
            loss_regr = iou_loss(locs_regr_pred_i, locs_regr_true_i)
            level_loss.append(loss_cls + loss_regr)
            #print(loss_cls.item(), loss_regr.item())
            total_loss += (loss_cls + loss_regr)
        level_losses.append(level_loss)
    level_losses = torch.FloatTensor(level_losses)
    if level_losses.shape[1] != 0:
        gt_box_levels = torch.argmin(level_losses, dim=0).int()
    else:
        gt_box_levels = torch.zeros(0).int()
    padding_gt_box_levels = torch.ones((MAX_NUM_GT_BOXES - num_gt_boxes), dtype=torch.int32) * -1
    gt_box_levels = torch.cat((gt_box_levels, padding_gt_box_levels))
    if total_loss != 0:
        total_loss /= 1.0 * (num_gt_boxes * len(strides))
    return gt_box_levels, total_loss


class Net(nn.Module):
    def __init__(self, class_num=80):
        super(Net, self).__init__()
        self.classification_branch = ClassificationBranch(class_num, 256)
        self.regression_branch = RegressionBranch(256)

    def forward(self, p_feature_maps):
        cls_anchor_based = []
        cls_anchor_free = []
        regr_anchor_based = []
        regr_anchor_free = []
        for p_feature in p_feature_maps:
            cb, cf = self.classification_branch(p_feature)
            rb, rf = self.regression_branch(p_feature)
            cls_anchor_based.append(cb)
            cls_anchor_free.append(cf)
            regr_anchor_based.append(rb)
            regr_anchor_free.append(rf)
        return cls_anchor_based, regr_anchor_based, torch.cat(cls_anchor_free, dim=1), torch.cat(regr_anchor_free,dim=1)

