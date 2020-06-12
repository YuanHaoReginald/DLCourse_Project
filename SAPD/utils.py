import torch
import numpy as np

def clip_boxes(img, boxes):
    _, _, height, width = img.shape
    boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], max=width)
    boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], max=height)
    boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
    boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)
    return boxes

def regress_boxes(locations, strides, regression):
    x1 = locations[:, :, 0] - regression[:, :, 0] * 4.0 * strides[:, :]
    y1 = locations[:, :, 1] - regression[:, :, 1] * 4.0 * strides[:, :]
    x2 = locations[:, :, 0] + regression[:, :, 2] * 4.0 * strides[:, :]
    y2 = locations[:, :, 1] + regression[:, :, 3] * 4.0 * strides[:, :]
    level_box = torch.stack((x1, y1, x2, y2), dim=0).T.float()
    return level_box

def Location(pyramid_features):
    strides = (8, 16, 32, 64, 128)
    feature_shapes = [torch.size(feature)[1:3] for feature in pyramid_features]
    locations_per_feature = []
    strides_per_feature = []
    for feature_shape, stride in zip(feature_shapes, strides):
        fh = feature_shape[0]
        fw = feature_shape[1]
        shifts_x = torch.FloatTensor(torch.arange(0, fw * stride, stride))
        shifts_y = torch.FloatTensor(torch.arange(0, fh * stride, stride))
        shift_x, shift_y = torch.meshgrid(shifts_x, shifts_y)
        # (h * w, )
        shift_x = torch.reshape(shift_x, (-1,))
        # (h * w, )
        shift_y = torch.reshape(shift_y, (-1,))
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        locations_per_feature.append(locations)

        strides = torch.ones((fh, fw)) * stride
        strides = torch.reshape(strides, (-1,))
        strides_per_feature.append(strides)
    # (sum(h * w), 2)
    locations = torch.cat(locations_per_feature, 0)
    # (batch, sum(h * w), 2)
    locations = torch.unsqueeze(locations, 0)
    locations = locations.repeat(locations, (torch.size(pyramid_features[0])(0), 1, 1))
    strides = torch.cat(strides_per_feature, 0)
    strides = torch.unsqueeze(strides, -1)
    strides = torch.repeat(strides, (torch.size(pyramid_features[0])(0), 1))
    return locations, strides

