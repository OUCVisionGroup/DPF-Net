import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
from timm.models.layers import DropPath
import torch.optim as optim
import itertools
from models import MAB

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
            nn.Mish(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            #nn.ReLU(inplace=True)
            nn.Mish(),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels // 2)
            self.conv_out = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(out_channels // 2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Crop x1 if it is necessary (e.g., due to uneven sizes of input images)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)

        if hasattr(self, 'conv_out'):
            x = self.conv_out(self.conv(x))
        else:
            x = self.conv(x)

        return x


class ResBlock(nn.Module):
    def __init__(self, num_filter):
        super(ResBlock, self).__init__()
        body = []
        for i in range(2):
            body.append(nn.ReflectionPad2d(1))
            body.append(nn.Conv2d(num_filter, num_filter, kernel_size=3, padding=0))
            if i == 0:
                #body.append(nn.LeakyReLU())
                body.append(nn.Mish())
        #body.append(SELayer(num_filter))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x)
        x = res + x
        return x

class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        in_conv = []

        in_conv.append(ResBlock(in_ch))
        in_conv.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
        in_conv.append(ResBlock(out_ch))
        in_conv.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))

        self.in_conv = nn.Sequential(*in_conv)

    def forward(self, x):
        return self.in_conv(x)

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        out_conv = []

        out_conv.append(ResBlock(in_ch))
        out_conv.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
        out_conv.append(ResBlock(out_ch))
        out_conv.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))

        self.out_conv = nn.Sequential(*out_conv)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.out_conv(x)
        return self.sigmoid(x)

class XCA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)


    def forward(self, x):
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

class LGFI(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=6,
                 num_heads=8, qkv_bias=True, attn_drop=0., drop=0.):
        super().__init__()

        self.dim = dim

        self.norm_xca = LayerNorm(self.dim, eps=1e-6)

        self.gamma_xca = nn.Parameter(layer_scale_init_value * torch.ones(self.dim),
                                      requires_grad=True) if layer_scale_init_value > 0 else None
        self.xca = XCA(self.dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.norm = LayerNorm(self.dim, eps=1e-6)
        self.pwconv1 = nn.Linear(self.dim, expan_ratio * self.dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expan_ratio * self.dim, self.dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((self.dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input_ = x

        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)

        x = x + self.gamma_xca * self.xca(self.norm_xca(x))

        x = x.reshape(B, H, W, C)

        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)

        x = input_ + self.drop_path(x)

        return x

class CT_UNet(nn.Module):
    def __init__(self, input_nc=3):
        super(CT_UNet, self).__init__()
        self.input_nc = input_nc
        self.inc = InConv(3, 16)
        self.maxpool = nn.MaxPool2d(2)

        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        self.down4 = Down(128, 256)

        self.bridge = ResBlock(256)

        self.up1 = Up(256+128, 128)
        self.up2 = Up(128+64, 64)
        self.up3 = Up(64+32, 32)
        self.up4 = Up(32+16, 16)

        self.lgfi1 = LGFI(128)
        self.lgfi2 = LGFI(64)
        self.lgfi3 = LGFI(32)
        self.lgfi4 = LGFI(16)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5 = self.bridge(x5)

        x = self.up1(x5, x4)
        x = self.lgfi1(x)
        x = self.up2(x, x3)
        x = self.lgfi2(x)
        x = self.up3(x, x2)
        x = self.lgfi3(x)
        x = self.up4(x, x1)
        x = self.lgfi4(x)

        return x