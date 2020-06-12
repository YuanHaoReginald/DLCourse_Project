import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import trim_zeros_graph, prop_box_graph
import numpy as np
from branch import *
from sapd_layers import *

class Net(nn.Module):
    def __init__(self, class_num=80):
        super(Net, self).__init__()
        self.classification_branch = ClassificationBranch(class_num, 256)
        self.regression_branch = RegressionBranch(256)

    def forward(self, p_feature_maps):
        #cls_anchor_based = []
        cls_anchor_free = []
        #regr_anchor_based = []
        regr_anchor_free = []
        for p_feature in p_feature_maps:
            cf = self.classification_branch(p_feature)
            rf = self.regression_branch(p_feature)
            #cls_anchor_based.append(cb)
            cls_anchor_free.append(cf)
            #regr_anchor_based.append(rb)
            regr_anchor_free.append(rf)
        return torch.cat(cls_anchor_free, dim=1), torch.cat(regr_anchor_free,dim=1)


class MetaSelectionNet(nn.Module):
    def __init__(self):
        super(MetaSelectionNet, self).__init__()
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3)
        self.fc = nn.Linear(256, 5)

    def forward(self, input):
        x = self.conv1(input)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        output = self.softmax(x)
        return output