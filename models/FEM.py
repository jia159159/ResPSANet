import torch.nn as nn
from .activate_function.mish import Mish
class FEM(nn.Module):

    def __init__(self, channels, reduction=16):
        super(FEM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels//reduction, kernel_size=1, padding=0)
        self.mish = Mish()
        self.fc2 = nn.Conv2d(channels//reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.mish(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight
    
    
    