import torch
import torch.nn as nn
from torchvision.models.vgg import vgg16
import cv2


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
        self.mse_loss_mean = nn.MSELoss(reduction='mean')
        self.l1_loss_none = nn.L1Loss(reduction='none')

    def forward(self, raw, enc, deg, ref, label_depth):
        vgg_loss1 = self.mse_loss_mean(self.vgg(ref), self.vgg(enc))
        vgg_loss2 = self.mse_loss_mean(self.vgg(raw), self.vgg(deg))

        ssim_loss1 = self.ssim(enc, ref).mean()
        ssim_loss2 = self.ssim(raw, deg).mean()

        mask = (1 + label_depth) / 2
        l1_loss = self.l1_loss_none(enc, ref)
        l1_loss = l1_loss.sum(dim=1, keepdim=True)
        l1_loss = (l1_loss * mask.unsqueeze(1)).mean()

        L_loss = get_L_loss(enc)
        L_loss = torch.tensor(L_loss).to(self.device)

        total_loss = vgg_loss1 + vgg_loss2 + 1.5*ssim_loss1 + l1_loss + 3*L_loss + 0.5*ssim_loss2

        return total_loss*0.5

def get_L_loss(enc):
    clear_mean = [[122.42, 18.99], [124.46, 6.90], [123.81, 11.68]]
    clear_std = [[55.98, 11.27], [9.16, 4.40], [15.47, 6.41]]
    enc = (enc * 255).to('cpu', dtype=torch.uint8)
    enc = enc.numpy()

    L_loss = 0.0
    for i in range(enc.shape[0]):
        enc_temp = enc[i].transpose((1, 2, 0))
        enc_temp = cv2.cvtColor(enc_temp, cv2.COLOR_RGB2LAB)
        x_mean, x_std = cv2.meanStdDev(enc_temp)
        x_mean, x_std = x_mean.squeeze(), x_std.squeeze()

        for i in range(3):
            loss1 = (x_mean[i] - clear_mean[i][0])**2 / (2 * clear_mean[i][1]**2)
            L_loss += loss1
            loss2 = (x_std[i] - clear_std[i][0])**2 / (2 * clear_std[i][1]**2)
            L_loss += loss2

    return L_loss / 500.0

