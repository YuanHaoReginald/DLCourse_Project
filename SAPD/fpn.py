import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()
        self.feature_model = torchvision.models.resnet50(pretrained=True)
        self.P5_1 = nn.Conv2d(2048, 256, kernel_size=1)
        self.P5_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.P4_1 = nn.Conv2d(1024, 256, kernel_size=1)
        self.P4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.P3_1 = nn.Conv2d(512, 256, kernel_size=1)
        self.P3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.P6 = nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=1)

        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        x = self.feature_model.conv1(inputs)
        x = self.feature_model.bn1(x)
        x = self.feature_model.relu(x)
        x = self.feature_model.maxpool(x)
        x = self.feature_model.layer1(x)
        c3 = self.feature_model.layer2(x)
        c4 = self.feature_model.layer3(c3)
        c5 = self.feature_model.layer4(c4)

        p5_1 = self.P5_1(c5)
        p5_down = F.interpolate(p5_1, scale_factor=2)
        p5_2 = self.P5_2(p5_1)

        p4_1 = self.P4_1(c4)
        p4_1 = p5_down + p4_1
        p4_down = F.interpolate(p4_1, scale_factor=2)
        p4_2 = self.P4_2(p4_1)

        p3_1 = self.P3_1(c3)
        p3_1 = p3_1 + p4_down
        p3_2 = self.P3_2(p3_1)

        p6 = self.P6(c5)

        p7_1 = self.P7_1(p6)
        p7_2 = self.P7_2(p7_1)

        return p3_2, p4_2, p5_2, p6, p7_2



