import torch.nn as nn
from fpn import FPN
from fsaf import fsaf

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fpn = FPN()
        self.fsaf = fsaf()
    def forward(self, inputs):
        p1, p2, p3, p4 = self.fpn(inputs)
        c1, r1 = self.fsaf(p1)
        c2, r2 = self.fsaf(p2)
        c3, r3 = self.fsaf(p3)
        c4, r4 = self.fsaf(p4)
        return c1, c2, c3, c4, r1, r2, r3, r4