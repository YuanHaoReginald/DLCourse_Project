
import numpy as np
import torch
import torch.nn as nn

def generate_anchors(base_anchor_size=16):
    ratios = np.asarray([0.5, 1, 2])
    scales = np.asarray([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    num_anchors = len(ratios) * len(scales)

    anchors = np.zeros((num_anchors, 4))
    anchors[:, 2:] = base_anchor_size * np.tile(scales, (2, len(ratios))).T

    areas = anchors[:, 2] * anchors[:, 3]

    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
    return anchors

def shift(shape, stride, anchors):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    all_anchors = all_anchors.reshape((K * A, 4))
    return all_anchors

class Anchors(nn.Module):
    def __init__(self):
        super(Anchors, self).__init__()
        self.pyramid_levels = [3, 4, 5, 6, 7]
        self.strides = [2 ** x for x in self.pyramid_levels]
        self.sizes = [2 ** (x + 2) for x in self.pyramid_levels]

    def forward(self, image):
        image_shape = image.shape[2:]
        image_shape = np.asarray(image_shape)
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]

        #all_anchors = np.zeros((0, 4)).astype(np.float32)
        all_anchors = []
        for idx, p in enumerate(self.pyramid_levels):
            anchors = generate_anchors(self.sizes[idx])
            shifted_anchors = shift(image_shapes[idx], self.strides[idx], anchors)
            shifted_anchors = np.expand_dims(shifted_anchors, axis=0)
            shifted_anchors = torch.from_numpy(shifted_anchors.astype(np.float32)).cuda()
            all_anchors.append(shifted_anchors)
        return all_anchors
