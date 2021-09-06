import torch
from torch import nn
from torch.nn import functional as F

class Flatten(nn.Module):
    def forward(self, X):
        return X.view(X.shape[0], -1)

class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel, mid_channel=None):
        super().__init__()
        if not mid_channel:
            mid_channel = out_channel
        self.doubleconv = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(mid_channel),
            nn.Conv2d(mid_channel, out_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channel),
        )
    def forward(self, X):
        return self.doubleconv(X)

class Down(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channel, out_channel)
        )

    def forward(self, X):
        return  self.down(X)

class Up(nn.Module):
    def __init__(self, in_channel, out_channel, bilinear=False): # 是用普通上采样还是bilinear上采样
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channel, out_channel, in_channel // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channel, in_channel // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channel, out_channel)

    def forward(self, X, old_out):
        up_out = self.up(X)
        diffY = old_out.shape[2] - up_out.shape[2]
        diffX = old_out.shape[3] - up_out.shape[3]

        up_out = F.pad(up_out, [diffY, diffY - diffY // 2, diffX, diffX - diffX // 2]) # 默认从最后轴向前padding
        X = torch.cat([up_out, old_out], dim=1)
        return self.conv(X)

class OutConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)

    def forward(self, X):
        return self.conv(X)