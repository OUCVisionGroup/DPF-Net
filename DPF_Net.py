import torch
import torch.nn as nn
import torch.optim as optim
import CT_UNet
import PFGM


class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels):
        super(FeatureFusionModule, self).__init__()
        self.weight_unet = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )
        self.weight_vae = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )

    def forward(self, unet_features, vae_features):
        weights_unet = self.weight_unet(unet_features)
        weights_vae = self.weight_vae(vae_features)
        fused_features = weights_unet * unet_features + weights_vae * vae_features

        return fused_features

class TotalNetwork(nn.Module):
    def __init__(self, device):
        super(TotalNetwork, self).__init__()
        self.device = device
        self.EncNet = CT_UNet.CT_UNet()
        self.EncNet.to(self.device)
        self.VAENet = PFGM.PFGM(256, device)
        self.VAENet.to(self.device)
        self.Fusion = FeatureFusionModule(16)
        self.Fusion.to(self.device)
        self.OutLayer = CT_UNet.OutConv(16, 3)
        self.OutLayer.to(self.device)


    def forward(self, x, B, depth_abs, beta_D, beta_B):
        Unet_feature = self.EncNet(x)
        VAE_feature = self.VAENet(x, B, depth_abs, beta_D, beta_B)

        total_feature = self.Fusion(Unet_feature, VAE_feature)
        encImg = self.OutLayer(total_feature)

        return encImg

    def set_optimizer(self, lr=0.001):
        parameters = [
            {'params': self.VAENet.parameters(), 'lr': lr},
            {'params': self.EncNet.parameters(), 'lr': lr},
            {'params': self.Fusion.parameters(), 'lr': lr},
            {'params': self.OutLayer.parameters(), 'lr': lr}
        ]
        self.optimizer = optim.Adam(parameters)

    def get_train_parameters(self, lr=0.001):
        parameters = [
            {'params': self.VAENet.parameters(), 'lr': lr},
            {'params': self.EncNet.parameters(), 'lr': lr},
            {'params': self.Fusion.parameters(), 'lr': lr},
            {'params': self.OutLayer.parameters(), 'lr': lr}
        ]
        return parameters
