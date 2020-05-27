import torch
import torch.nn as nn
import torch.nn.functional as F

class fsaf(nn.Module):
    def __init__(self):
        super(fsaf, self).__init__()
        self.conv1 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, 3, 1)
        self.bn4 = nn.BatchNorm2d(256)
        self.class_conv = nn.Conv2d(256, 91, 1)
        self.regress_conv = nn.Conv2d(256, 4, 1)
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        class_output = self.class_conv(x)
        class_output = F.softmax(class_output, dim=1)
        class_output.permute(0, 2, 3, 1)
        regress_output = self.regress_conv(x)
        regress_output.permute(0, 2, 3, 1)
        return class_output, regress_output