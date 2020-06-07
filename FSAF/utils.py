import torch
import numpy as np
import math

def clip_boxes(boxes, img):
    batch_size, num_channels, height, width = img.shape
    boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
    boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

    boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
    boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)

    return boxes

def generate_predict_boxes(anchors, regressions, mean=None, std=None):
    if mean is None:
        mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)).cuda()
    if std is None:
        std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)).cuda()

    widths  = anchors[:, :, 2] - anchors[:, :, 0]
    heights = anchors[:, :, 3] - anchors[:, :, 1]
    ctr_x   = anchors[:, :, 0] + 0.5 * widths
    ctr_y   = anchors[:, :, 1] + 0.5 * heights

    dx = regressions[:, :, 0] * std[0] + mean[0]
    dy = regressions[:, :, 1] * std[1] + mean[1]
    dw = regressions[:, :, 2] * std[2] + mean[2]
    dh = regressions[:, :, 3] * std[3] + mean[3]

    predict_ctr_x = ctr_x + dx * widths
    predict_ctr_y = ctr_y + dy * heights
    predict_w     = torch.exp(dw) * widths
    predict_h     = torch.exp(dh) * heights

    predict_boxes_x1 = predict_ctr_x - 0.5 * predict_w
    predict_boxes_y1 = predict_ctr_y - 0.5 * predict_h
    predict_boxes_x2 = predict_ctr_x + 0.5 * predict_w
    predict_boxes_y2 = predict_ctr_y + 0.5 * predict_h

    predict_boxes = torch.stack([predict_boxes_x1, predict_boxes_y1, predict_boxes_x2, predict_boxes_y2], dim=2)

    return predict_boxes

def anchor_free_predict_boxes(regressions, strides, image_shapes):
    regressions = regressions.squeeze() * 4
    boxes = torch.zeros(0, 4).cuda()
    fa = torch.prod(image_shapes, dim=1)
    for idx, stride in enumerate(strides):
        image_shape = image_shapes[idx]
        fh = math.ceil(image_shape[0])
        fw = math.ceil(image_shape[1])
        start_index = torch.sum(fa[:idx])
        end_index = start_index + fh * fw
        start_index = start_index.int()
        end_index = end_index.int()
        regr = regressions[start_index:end_index, :]
        shift_x = (np.arange(0, image_shape[1]) + 0.5) * stride
        shift_y = (np.arange(0, image_shape[0]) + 0.5) * stride

        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        ctr_x = shift_x.ravel()
        ctr_y = shift_y.ravel()

        ctr_x = torch.from_numpy(ctr_x).cuda()
        ctr_y = torch.from_numpy(ctr_y).cuda()
        x1 = ctr_x - regr[:, 0]
        y1 = ctr_y - regr[:, 1]
        x2 = ctr_x + regr[:, 2]
        y2 = ctr_y + regr[:, 3]
        level_box = torch.stack((x1, y1, x2, y2), dim=0).T.float()

        boxes = torch.cat((boxes, level_box), dim = 0)
    return boxes

def trim_zeros_graph(boxes):
    non_zeros = torch.sum(torch.abs(boxes), dim=1).bool()
    boxes = boxes[non_zeros, :]
    return boxes, non_zeros

def prop_box_graph(boxes, scale, width, height):
    boxes_ctr_x = (boxes[:,0] + boxes[:,2]) / 2.0
    boxes_ctr_y = (boxes[:,1] + boxes[:,3]) / 2.0
    boxes_width = boxes[:,2] - boxes[:,0]
    boxes_height = boxes[:,3] - boxes[:,1]
    boxes_width = boxes_width * scale
    boxes_width = boxes_height * scale
    boxes_scale_x1 = boxes_ctr_x - 0.5 * boxes_width
    boxes_scale_x2 = boxes_ctr_x + 0.5 * boxes_width
    boxes_scale_y1 = boxes_ctr_y - 0.5 * boxes_height
    boxes_scale_y2 = boxes_ctr_y + 0.5 * boxes_height
    x1 = torch.floor(boxes_scale_x1)
    y1 = torch.floor(boxes_scale_y1)
    x2 = torch.ceil(boxes_scale_x2)
    y2 = torch.ceil(boxes_scale_y2)
    x2 = torch.clamp(x2, 1, width).int()
    y2 = torch.clamp(y2, 1, height).int()
    x1 = torch.clamp(x1, 0).int()
    y1 = torch.clamp(y1, 0).int()
    x1 = torch.min(x1, x2 - 1)
    y1 = torch.min(y1, y2 - 1)

    return x1, y1 ,x2, y2

