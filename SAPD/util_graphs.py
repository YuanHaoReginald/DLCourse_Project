import torch


def xyxy2cxcywh(xyxy):
    """
    Convert [x1 y1 x2 y2] box format to [xc yc w h] format.
    """
    return torch.cat((0.5 * (xyxy[:, 0:2] + xyxy[:, 2:4]), xyxy[:, 2:4] - xyxy[:, 0:2]), dim=-1)


def cxcywh2xyxy(xywh):
    """
    Convert [cx cy w y] box format to [x1 y1 x2 y2] format.
    """
    return torch.cat((xywh[:, 0:2] - 0.5 * xywh[:, 2:4], xywh[:, 0:2] + 0.5 * xywh[:, 2:4]), dim=-1)


def normalize_boxes(boxes, width, height, stride):
    # normalize:
    x1 = boxes[:, 0:1] / stride / width
    y1 = boxes[:, 1:2] / stride / height
    x2 = boxes[:, 2:3] / stride / width
    y2 = boxes[:, 3:4] / stride / height
    return torch.cat([y1, x1, y2, x2], dim=-1)


def shrink_and_normalize_boxes(boxes, width, height, stride, shrink_ratio=0.2):
    # shrink
    boxes = xyxy2cxcywh(boxes)
    boxes = torch.cat((boxes[:, :2], boxes[:, 2:] * shrink_ratio), dim=-1)
    boxes = cxcywh2xyxy(boxes)
    # normalize:
    x1 = boxes[:, 0:1] / stride / width
    y1 = boxes[:, 1:2] / stride / height
    x2 = boxes[:, 2:3] / stride / width
    y2 = boxes[:, 3:4] / stride / height
    return torch.cat([x1, y1, x2, y2], dim=-1)


def shrink_and_project_boxes(boxes, width, height, stride, shrink_ratio=0.2, keep_dims=False):
    """
    Compute proportional box coordinates.

    Box centers are fixed. Box w and h scaled by scale.
    """
    # shrink
    boxes = xyxy2cxcywh(boxes)
    boxes = torch.cat((boxes[:, :2], boxes[:, 2:] * shrink_ratio), dim=-1)
    boxes = cxcywh2xyxy(boxes)

    if keep_dims:
        x1 = torch.floor(boxes[:, 0:1] / stride)
        y1 = torch.floor(boxes[:, 1:2] / stride)
        x2 = torch.ceil(boxes[:, 2:3] / stride)
        y2 = torch.ceil(boxes[:, 3:4] / stride)
    else:
        x1 = torch.floor(boxes[:, 0] / stride)
        y1 = torch.floor(boxes[:, 1] / stride)
        x2 = torch.ceil(boxes[:, 2] / stride)
        y2 = torch.ceil(boxes[:, 3] / stride)
    width = torch.FloatTensor(width)
    height = torch.FloatTensor(height)
    x2 = torch.IntTensor(clip_by_tensor(x2, 1, width))
    y2 = torch.IntTensor(clip_by_tensor(y2, 1, height))
    x1 = torch.IntTensor(clip_by_tensor(x1, 0, torch.FloatTensor(x2) - 1))
    y1 = torch.IntTensor(clip_by_tensor(y1, 0, torch.FloatTensor(y2) - 1))
    return x1, y1, x2, y2


def trim_padding_boxes(boxes):
    """
    Often boxes are represented with matrices of shape [N, 4] and are padded with zeros.
    This removes zero boxes.

    Args:
        boxes: [N, 4] matrix of boxes.

    Returns:

    """
    box_sum = torch.sum(torch.abs(boxes), dim=1)
    non_zeros = torch.ByteTensor(box_sum > 0)
    boxes = torch.masked_select(boxes, non_zeros)
    return boxes, non_zeros



def clip_by_tensor(t,t_min,t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    t=t.float()
    t_min=t_min.float()
    t_max=t_max.float()
 
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result