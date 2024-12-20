import torch
import torch.nn as nn
from torchvision.models.vgg import vgg16
import torch.nn.functional as F


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

class Totaloss(nn.Module):
    def __init__(self, device):
        super(Totaloss, self).__init__()
        self.device = device
        self.ssim = SSIM()
        self.ssim.to(self.device)

        vgg = vgg16(pretrained=True)
        vgg_loss = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in vgg_loss.parameters():
            param.requires_grad = False
        self.vgg = vgg_loss
        self.vgg.to(self.device)
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, raw, deg, B, betaD, betaB, depth, x_B, x_betaD, x_betaB, x_d):
        raw = raw/255.0
        deg = deg/255.0
        vgg_loss = self.mse_loss(self.vgg(raw), self.vgg(deg))
        ssim_loss = self.ssim(raw, deg).mean()
        B_loss = self.l1_loss(B, x_B) * 0.01
        betaD_loss = self.l1_loss(betaD, x_betaD)
        betaB_loss = self.l1_loss(betaB, x_betaB)
        depth_loss = self.l1_loss(depth, x_d) * 0.1

        total_loss = B_loss + betaD_loss + betaB_loss + depth_loss + ssim_loss + vgg_loss

        return total_loss * 0.1
