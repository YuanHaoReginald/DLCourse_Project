import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import IoULoss, FocalLoss
from util_graphs import *
from roi_align import CropAndResize

class MetaSelectInput(nn.Module):
    def __init__(self, strides=(8, 16, 32, 64, 128), pool_size=7):
        super(MetaSelectInput, self).__init__()
        self.strides = strides
        self.pool_size = pool_size

    def forward(self, batch_gt_boxes, list_batch_fms):
        batch_gt_boxes = batch_gt_boxes[..., :4]
        batch_size = batch_gt_boxes.size(0)
        max_gt_boxes = batch_gt_boxes.size(1)
        gt_boxes_batch_ids = torch.unsqueeze(torch.range(0, batch_size), -1)
        gt_boxes_batch_ids = gt_boxes_batch_ids.repeat(1, max_gt_boxes)
        gt_boxes_batch_ids = torch.reshape(gt_boxes_batch_ids, (-1,))
        batch_gt_boxes = torch.reshape(batch_gt_boxes, (-1, batch_gt_boxes.size(-1)))
        gt_boxes, non_zeros = trim_padding_boxes(batch_gt_boxes)
        gt_boxes_batch_ids = torch.masked_select(gt_boxes_batch_ids, non_zeros)

        rois_from_fms = []
        for i, batch_fm in enumerate(list_batch_fms):
            stride = torch.FloatTensor(self.strides[i])
            fm_height = torch.FloatTensor(batch_fm.size(1))
            fm_width =  torch.FloatTensor(batch_fm.size(2))
            normed_gt_boxes = normalize_boxes(gt_boxes, width=fm_width, height=fm_height, stride=stride)
            crop_and_resize = CropAndResize(self.pool_size, self.pool_size)
            rois = crop_and_resize(batch_fm, normed_gt_boxes, gt_boxes_batch_ids)
            rois_from_fms.append(rois)
        rois = torch.cat(rois_from_fms, -1)
        return rois, gt_boxes_batch_ids

class MetaSelectTarget(nn.Module):
    def __init__(self, strides=(8, 16, 32, 64, 128), shrink_ratio=0.2):
        super(MetaSelectTarget, self).__init__()
        self.strides = strides
        self.shrink_ratio = shrink_ratio
        
    def forward(self, inputs, batch_cls_pred, batch_regr_pred, feature_shapes, batch_gt_boxes):
        feature_shapes = feature_shapes[0]
        batch_size = batch_gt_boxes.shape[0]
        batch_box_levels = []
        for i in range(batch_size):
            batch_box_level = build_meta_select_target(batch_cls_pred[i], batch_regr_pred[i],
                            feature_shapes, self.strides, self.shrink_ratio)
            batch_box_levels.append(batch_box_level)
        batch_box_levels = torch.stack(batch_box_levels)
        batch_box_levels = torch.reshape(batch_box_levels, (-1,))
        mask = torch.ne(batch_box_levels, -1)
        valid_box_levels = torch.masked_select(batch_box_levels, mask)
        return valid_box_levels

def build_meta_select_target(cls_pred, regr_pred, gt_boxes, feature_shapes, strides, shrink_ratio=0.2):
    gt_labels = gt_boxes[:, 4].int()
    gt_boxes = gt_boxes[:, :4]
    MAX_NUM_GT_BOXES = gt_boxes.size(0)
    focal_loss = FocalLoss()
    iou_loss = IoULoss()
    gt_boxes, non_zeros = trim_padding_boxes(gt_boxes) # TODO
    num_gt_boxes = gt_boxes.size(0)
    gt_labels = torch.masked_select(gt_labels, non_zeros) # 是否一致？
    level_losses = []
    for level_id in range(len(strides)):
        stride = strides[level_id]
        fh = feature_shapes[level_id][0].int().item()
        fw = feature_shapes[level_id][1].int().item()
        fa = torch.prod(feature_shapes, dim=-1).int()
        start_idx = torch.sum(fa[:level_id])
        end_idx = start_idx + fh * fw
        cls_pred_i = cls_pred[start_idx:end_idx, :].reshape(fh, fw, -1)
        regr_pred_i = regr_pred[start_idx:end_idx, :].reshape(fh, fw, -1)
        # (num_gt_boxes, )
        x1, y1, x2, y2 = shrink_and_project_boxes(gt_boxes, fw, fh, stride, shrink_ratio=shrink_ratio) #
        level_loss = []
        for i in range(num_gt_boxes):
            x1_ = x1[i]
            y1_ = y1[i]
            x2_ = x2[i]
            y2_ = y2[i]
            gt_box = gt_boxes[i]
            gt_label = gt_labels[i]

            def do_match_pixels_in_level():
                locs_cls_pred_i = cls_pred_i[y1_:y2_, x1_:x2_, :].reshape(-1, cls_pred_i.shape[2])
                locs_a, num_class = locs_cls_pred_i.shape
                locs_cls_true_i = torch.zeros(num_class).cuda()
                locs_cls_true_i[gt_label] = 1
                locs_cls_true_i = locs_cls_true_i.repeat(locs_a, 1)
                loss_cls = focal_loss(locs_cls_pred_i, locs_cls_true_i)
                locs_regr_pred_i = regr_pred_i[y1_:y2_, x1_:x2_, :].reshape(-1, 4)
                shift_x = (np.arange(x1_.cpu(), x2_.cpu()) + 0.5) * stride
                shift_y = (np.arange(y1_.cpu(), y2_.cpu()) + 0.5) * stride

                shift_x, shift_y = torch.meshgrid(shift_x, shift_y)
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
                return loss_cls + loss_regr

            def do_not_match_pixels_in_level():
                box_loss = torch.FloatTensor(1e7)
                return box_loss
            
            if torch.equal(x1_.int(), x2_.int()) | torch.equal(y1_.int(), y2_.int()):
                not_match_loss = do_not_match_pixels_in_level()
                level_loss.append(not_match_loss)
            else:
                match_loss = do_match_pixels_in_level()
                level_loss.append(match_loss)
        level_losses.append(level_loss)
    level_losses = torch.FloatTensor(level_losses)
    if level_losses.shape[1] != 0:
        gt_box_levels = torch.argmin(level_losses, dim=0).int()
    else:
        gt_box_levels = torch.zeros(0).int()
    padding_gt_box_levels = torch.ones((MAX_NUM_GT_BOXES - num_gt_boxes), dtype=torch.int32) * -1
    gt_box_levels = torch.cat((gt_box_levels, padding_gt_box_levels))
    return gt_box_levels


class MetaSelectWeight(nn.Module):
    def __init__(self, max_gt_boxes=100, soft_select=True):
        super(MetaSelectWeight, self).__init__()
        self.max_gt_boxes = max_gt_boxes
        self.soft_select = soft_select

    def forward(self, gt_boxes_select_weight, gt_boxes_batch_ids, batch_num_gt_boxes):
        # (b, 1) --> (b, )
        batch_size = batch_num_gt_boxes.shape[0]
        batch_num_gt_boxes = batch_num_gt_boxes[:, 0]
        batch_select_weight = []
        for i in range(batch_size):
            batch_item_select_weight = torch.masked_select(gt_boxes_select_weight, torch.eq(gt_boxes_batch_ids, i))
            pad_top_bot = torch.stack([torch.Tensor(0), self.max_gt_boxes - batch_num_gt_boxes[i]], dim=0)
            pad = torch.stack([pad_top_bot, torch.Tensor([0, 0])], dim=0)
            batch_select_weight.append(F.pad(batch_item_select_weight, pad, "constant", -1))
        batch_select_weight = torch.stack(batch_select_weight, dim=0)
        return batch_select_weight


def build_sapd_target(gt_boxes, meta_select_weight, fm_shapes, num_classes, strides, shrink_ratio=0.2):
    gt_labels = torch.IntTensor(gt_boxes[:, 4])
    gt_boxes = gt_boxes[:, :4]
    gt_boxes, non_zeros = trim_padding_boxes(gt_boxes)
    gt_labels = torch.masked_select(gt_labels, non_zeros)
    meta_select_weight = torch.masked_select(meta_select_weight, non_zeros)

    def do_have_gt_boxes():
        cls_target = torch.zeros(0, num_classes + 1 + 1).float()
        regr_target = torch.zeros(0, 4 + 1 + 1).float()
        for level_id in range(len(strides)):
            level_meta_select_weight = meta_select_weight[:, level_id]

            fm_shape = fm_shapes[level_id]
            stride = strides[level_id]
            fh = fm_shape[0]
            fw = fm_shape[1]

            pos_x1, pos_y1, pos_x2, pos_y2 = shrink_and_project_boxes(gt_boxes, fw, fh, stride, shrink_ratio)

            def build_single_gt_box_sapd_target(pos_x1_, pos_y1_, pos_x2_, pos_y2_, gt_box, gt_label, level_box_meta_select_weight):
                level_pos_box_cls_target = torch.zeros(pos_y2_ - pos_y1_, pos_x2_ - pos_x1_, num_classes).float()
                level_pos_box_gt_label_col = torch.ones(pos_y2_ - pos_y1_, pos_x2_ - pos_x1_, 1).float()
                level_pos_box_cls_target = torch.cat((level_pos_box_cls_target[..., :gt_label],
                                                      level_pos_box_gt_label_col,
                                                      level_pos_box_cls_target[..., gt_label + 1:]), -1)
                neg_top_bot = torch.stack((pos_y1_, fh - pos_y2_), dim=0)
                neg_lef_rit = torch.stack((pos_x1_, fw - pos_x2_), dim=0)
                neg_pad = torch.stack([neg_top_bot, neg_lef_rit], dim=0)
                level_box_cls_target = F.pad(level_pos_box_cls_target,
                                              torch.cat((neg_pad, torch.Tensor([[0, 0]])), "constant", 0))
                pos_locs_x = torch.range(pos_x1_, pos_x2_).float()
                pos_locs_y = torch.range(pos_y1_, pos_y2_).float()
                pos_shift_x = (pos_locs_x + 0.5) * stride
                pos_shift_y = (pos_locs_y + 0.5) * stride
                pos_shift_yy, pos_shift_xx = torch.meshgrid(pos_shift_y, pos_shift_x)
                pos_shifts = torch.stack((pos_shift_xx, pos_shift_yy, pos_shift_xx, pos_shift_yy), dim=-1)
                dl = torch.max(pos_shifts[:, :, 0] - gt_box[0], 0)
                dt = torch.max(pos_shifts[:, :, 1] - gt_box[1], 0)
                dr = torch.max(gt_box[2] - pos_shifts[:, :, 2], 0)
                db = torch.max(gt_box[3] - pos_shifts[:, :, 3], 0)
                deltas = torch.stack((dl, dt, dr, db), dim=-1)
                level_box_regr_pos_target = deltas / 4.0 / stride
                level_pos_box_ap_weight = torch.max(dl, dr) * torch.min(dt, db) / torch.max(dl, dr) / torch.max(dt,
                                                                                                                    db)
                level_pos_box_soft_weight = level_pos_box_ap_weight * level_box_meta_select_weight
                level_box_soft_weight = F.pad(level_pos_box_soft_weight, neg_pad, "constant", 1.0)
                level_pos_box_regr_mask = torch.ones(pos_y2_ - pos_y1_, pos_x2_ - pos_x1_)
                level_box_regr_mask = F.pad(level_pos_box_regr_mask, neg_pad, "constant")
                level_box_regr_target = F.pad(level_box_regr_pos_target,
                                               torch.cat((neg_pad, torch.Tensor([[0, 0]])), 0), "constant")
                level_box_cls_target = torch.cat([level_box_cls_target, level_box_soft_weight[..., None],
                                                  level_box_regr_mask[..., None]], -1)
                level_box_regr_target = torch.cat([level_box_regr_target, level_box_soft_weight[..., None],
                                                   level_box_regr_mask[..., None]], -1)
                level_box_pos_area = (dl + dr) * (dt + db)
                level_box_area = F.pad(level_box_pos_area, neg_pad, "constant", 1e7)
                return level_box_cls_target, level_box_regr_target, level_box_area

            batch_size = gt_boxes.shape[0]
            level_cls_target = []
            level_regr_target = []
            level_area = []
            for i in range(batch_size):
                l_cls_target, l_regr_target, l_area = build_single_gt_box_sapd_target(pos_x1[i], pos_y1[i], pos_x2[i], pos_y2[i], 
                                                    gt_boxes[i], gt_labels[i], level_meta_select_weight[i])
                level_cls_target.append(l_cls_target)
                level_regr_target.append(l_regr_target)
                level_area.append(l_area)
            level_cls_target = torch.FloatTensor(level_cls_target)
            level_regr_target = torch.FloatTensor(level_regr_target)
            level_area = torch.FloatTensor(level_area)
                
            level_min_area_box_indices = torch.argmin(level_area, 0)
            level_min_area_box_indices = torch.reshape(level_min_area_box_indices, (-1,))
            # (1, fw)
            locs_x = torch.range(0, fw)
            # (1, fh)
            locs_y = torch.range(0, fh)
            # (fh, fw), (fh, fw)
            locs_yy, locs_xx = torch.meshgrid(locs_y, locs_x)
            locs_xx = torch.reshape(locs_xx, (-1,))
            locs_yy = torch.reshape(locs_yy, (-1,))
            # (fh * fw, 3)
            level_indices = torch.stack((level_min_area_box_indices, locs_yy, locs_xx), dim=-1)
            level_cls_target = level_cls_target[level_indices]
            level_regr_target = level_regr_target[level_indices]

            cls_target = torch.cat([cls_target, level_cls_target], 0)
            regr_target = torch.cat([regr_target, level_regr_target], 0)
        return [cls_target, regr_target]

    def do_not_have_gt_boxes():
        fa = torch.prod(fm_shapes, -1)
        fa_sum = torch.sum(fa)
        cls_target = torch.zeros(fa_sum, num_classes)
        regr_target = torch.zeros(fa_sum, 4)
        weight = torch.ones(fa_sum, 1)
        mask = torch.zeros(fa_sum, 1)
        cls_target = torch.cat([cls_target, weight, mask], -1)
        regr_target = torch.cat([regr_target, weight, mask], -1)
        return [cls_target, regr_target]

    cls_target = None
    regr_target = None

    if (torch.equal(torch.size(gt_boxes), 0)):
        cls_target, regr_target = do_not_have_gt_boxes()
    else:
        cls_target, regr_target = do_have_gt_boxes()

    return [cls_target, regr_target]

class SAPDTarget(nn.Module):

    def __init__(self):
        super(SAPDTarget, self).__init__()

    def forward(self, batch_gt_boxes, batch_meta_select_weight, fm_shapes, num_classes=80, strides=(8, 16, 32, 64, 128), shrink_ratio=0.2):
        batch_size = batch_gt_boxes.shape[0]
        cls_targets = []
        regr_targets = []
        for i in range(batch_size):
            cls_target, regr_target = build_sapd_target(batch_gt_boxes[i], batch_meta_select_weight[i],
                            fm_shapes, num_classes, strides, shrink_ratio)
            cls_targets.append(cls_target)
            regr_targets.append(regr_target)
        cls_targets = torch.stack(cls_targets, dim=0)
        regr_target = torch.stack(regr_targets, dim=0)
        return cls_targets, regr_targets