import torch
import torch.nn as nn
import torch.nn.functional as F
from Depth_Anything_V2_main.depth_anything_v2.dpt import DepthAnythingV2
import itertools
import torch.optim as optim

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Head(nn.Module):
    def __init__(self, size, in_channels=3, mid_channels=16, out_channels=3, with_linear=True):
        super(Head, self).__init__()
        self.with_linear = with_linear
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        if self.with_linear:
            self.linear = nn.Sequential(
                nn.Linear(out_channels * (size//4) * (size//4), out_channels),
                nn.ReLU(inplace=True))

    def forward(self, x):
        if self.with_linear:
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            return self.linear(x)
        else:
            return self.conv(x)

class Heads(nn.Module):
    def __init__(self, size, in_channels=3, mid_channels=16):
        super(Heads, self).__init__()
        self.size = size
        self.conv = nn.Sequential(
            DoubleConv(in_channels, mid_channels),
            DoubleConv(mid_channels, mid_channels)
        )

        self.head1 = Head(self.size, mid_channels)
        self.head2 = Head(self.size, mid_channels)
        self.head3 = Head(self.size, mid_channels, out_channels=2)

    def forward(self, x):
        x = self.conv(x)
        betaD = self.head1(x)
        betaB = self.head2(x)
        Scale = self.head3(x)
        return betaD, betaB, Scale

class DepthAnything(nn.Module):
    def __init__(self, path):
        super(DepthAnything, self).__init__()
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        encoder = 'vits'
        self.model = DepthAnythingV2(**model_configs[encoder])
        self.model.load_state_dict(torch.load(
            f'{path}/depth_anything_v2_{encoder}.pth',
            map_location='cpu'))

    def forward(self, x):
        expanded_x = F.interpolate(x, size=(280, 280), mode='bilinear', align_corners=True)

        disp = self.model(expanded_x)
        normalized = torch.zeros_like(disp)

        for i in range(disp.shape[0]):
            mask = disp[i] == 0
            non_zero_disp = disp[i][~mask]
            min_val = non_zero_disp.min()
            max_val = non_zero_disp.max()

            normalized[i][~mask] = (max_val - disp[i][~mask]) / (max_val - min_val)
            normalized[i][mask] = 1
        normalized = normalized.unsqueeze(1)
        shrunk_d = F.interpolate(normalized, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
        return shrunk_d

class MainNet(nn.Module):
    def __init__(self, device, imgSize=280, depth_anything_dir='/data/meih/project/Pycharm_tmp/Enc_with_Depth/Depth_Anything_V2_main'):
        super(MainNet, self).__init__()
        self.size = imgSize
        self.head_B = Head(self.size, in_channels=6)
        self.heads = Heads(self.size)
        self.den = DepthAnything(depth_anything_dir)

        self.head_B.to(device)
        self.heads.to(device)
        self.den.to(device)

    def forward(self, x, pre_B):
        xB = torch.cat((x, pre_B),dim=1)
        B = self.head_B(xB)
        depth = self.den(x)
        betaD, betaB, Scale = self.heads(x)
        replicated_Scale = Scale.unsqueeze(2).unsqueeze(3).repeat(1, 1, x.shape[2], x.shape[3])
        max_depth = replicated_Scale[:, 0:1, :, :]
        min_depth = replicated_Scale[:, 1:2, :, :]
        d = depth * (max_depth - min_depth) + min_depth

        return B*255, betaD, betaB, d

    def set_optimizer(self, lr=0.001):
        parameters = itertools.chain(self.head_B.parameters(), self.heads.parameters())
        self.optimizer = optim.Adam(parameters, lr=lr)

    def get_train_parameters(self, lr=0.00005):
        parameters = [
            {'params': self.head_B.parameters(), 'lr': lr},
            {'params': self.heads.parameters(), 'lr': lr}
        ]
        return parameters
