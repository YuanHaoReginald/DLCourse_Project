import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ClassificationBranch(nn.Module):
    def __init__(self, class_num, feature_num, anchor_num=9):
        super(ClassificationBranch, self).__init__()

        self.num_classes = class_num
        self.num_anchors = anchor_num

        self.conv1 = nn.Conv2d(feature_num, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)

        #self.anchor_based_output = nn.Conv2d(256, anchor_num * class_num, kernel_size=3, padding=1)
        self.anchor_free_output = nn.Conv2d(256, class_num, kernel_size=3, padding=1)

    def forward(self, inputs):
        out = self.conv1(inputs)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = F.relu(out)
        out = self.conv4(out)
        out = F.relu(out)

        #b = self.anchor_based_output(out)
        f = self.anchor_free_output(out)
        #b = torch.sigmoid(b)
        f = torch.sigmoid(f)
        #b = b.permute(0, 2, 3, 1)
        f = f.permute(0, 2, 3, 1)
        batch_size = f.shape[0]
        #b = b.contiguous().view(batch_size, -1, self.num_classes)
        f = f.contiguous().view(batch_size, -1, self.num_classes)
        return f

class RegressionBranch(nn.Module):
    def __init__(self, feature_num, anchor_num=9):
        super(RegressionBranch, self).__init__()
        self.conv1 = nn.Conv2d(feature_num, 256, 3, padding=1)

        self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)

        #self.anchor_based_output = nn.Conv2d(256, anchor_num * 4, kernel_size=3, padding=1)
        self.anchor_free_output = nn.Conv2d(256, 4, kernel_size=3, padding=1)

    def forward(self, inputs):
        out = self.conv1(inputs)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = F.relu(out)
        out = self.conv4(out)
        out = F.relu(out)

        f = self.anchor_free_output(out)
        #b = self.anchor_based_output(out)

        #b = b.permute(0, 2, 3, 1)
        f = f.permute(0, 2, 3, 1)
        batch_size = f.shape[0]
        #b = b.contiguous().view(batch_size, -1, 4)
        f = f.contiguous().view(batch_size, -1, 4)

        return f



