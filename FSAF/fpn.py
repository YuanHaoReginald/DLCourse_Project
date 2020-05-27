import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()
        self.features_model = torchvision.models.resnet50(pretrained=True)
        self.conv1 = nn.Conv2d(2048, 256, 1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(1024, 256, 1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(512, 256, 1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, 1)
        self.bn4 = nn.BatchNorm2d(256)
    def forward(self, inputs):
        x = self.features_model.conv1(inputs)
        x = self.features_model.bn1(x)
        x = self.features_model.relu(x)
        x = self.features_model.maxpool(x)
        c1 = self.features_model.layer1(x)
        c2 = self.features_model.layer2(c1)
        c3 = self.features_model.layer3(c2)
        c4 = self.features_model.layer4(c3)
        p1 = self.conv1(c4)
        p1 = self.bn1(p1)
        p1 = F.relu(p1)
        p2 = self.conv2(c3)
        p2 = self.bn2(p2)
        p2 = F.relu(p2)
        p2 = p2 + F.interpolate(p1, scale_factor=2)
        p3 = self.conv3(c2)
        p3 = self.bn3(p3)
        p3 = F.relu(p3)
        p3 = p3 + F.interpolate(p2, scale_factor=2)
        p4 = self.conv4(c1)
        p4 - self.bn4(p4)
        p4 = F.relu(p4)
        p3 = p3 + F.interpolate(p3, scale_factor=2)
        p1 = F.interpolate(p1, size=(256, 256))
        p2 = F.interpolate(p2, size=(256, 256))
        p3 = F.interpolate(p3, size=(256, 256))
        p4 = F.interpolate(p4, size=(256, 256))
        return p1, p2, p3, p4