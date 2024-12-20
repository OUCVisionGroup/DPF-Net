from torch.autograd import Function, Variable
import numpy as np
import torch
import torch.utils.data
from torch import nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Mish(),
            nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.conv(x)
class NearestEmbedFunc(Function):
    @staticmethod
    def forward(ctx, input, emb):
        if input.size(1) != emb.size(0):
            raise RuntimeError('invalid argument: input.size(1) ({}) must be equal to emb.size(0) ({})'.
                               format(input.size(1), emb.size(0)))

        ctx.batch_size = input.size(0)
        ctx.num_latents = int(np.prod(np.array(input.size()[2:])))
        ctx.emb_dim = emb.size(0)
        ctx.num_emb = emb.size(1)
        ctx.input_type = type(input)
        ctx.dims = list(range(len(input.size())))

        x_expanded = input.unsqueeze(-1)
        num_arbitrary_dims = len(ctx.dims) - 2
        if num_arbitrary_dims:
            emb_expanded = emb.view(
                emb.shape[0], *([1] * num_arbitrary_dims), emb.shape[1])
        else:
            emb_expanded = emb

        dist = torch.norm(x_expanded - emb_expanded, 2, 1)
        _, argmin = dist.min(-1)
        shifted_shape = [input.shape[0], *
                         list(input.shape[2:]), input.shape[1]]
        result = emb.t().index_select(0, argmin.view(-1)
                                      ).view(shifted_shape).permute(0, ctx.dims[-1], *ctx.dims[1:-1])

        ctx.save_for_backward(argmin)
        return result.contiguous(), argmin

    @staticmethod
    def backward(ctx, grad_output, argmin=None):
        grad_input = grad_emb = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output

        if ctx.needs_input_grad[1]:
            argmin, = ctx.saved_variables
            latent_indices = torch.arange(ctx.num_emb).type_as(argmin)
            idx_choices = (argmin.view(-1, 1) ==
                           latent_indices.view(1, -1)).type_as(grad_output.data)
            n_idx_choice = idx_choices.sum(0)
            n_idx_choice[n_idx_choice == 0] = 1
            idx_avg_choices = idx_choices / n_idx_choice
            grad_output = grad_output.permute(0, *ctx.dims[2:], 1).contiguous()
            grad_output = grad_output.view(
                ctx.batch_size * ctx.num_latents, ctx.emb_dim)
            grad_emb = torch.sum(grad_output.data.view(-1, ctx.emb_dim, 1) *
                                 idx_avg_choices.view(-1, 1, ctx.num_emb), 0)
        return grad_input, grad_emb, None, None

def nearest_embed(x, emb):
    return NearestEmbedFunc().apply(x, emb)

class NearestEmbed(nn.Module):
    def __init__(self, num_embeddings, embeddings_dim):
        super(NearestEmbed, self).__init__()
        self.weight = nn.Parameter(torch.rand(embeddings_dim, num_embeddings))

    def forward(self, x, weight_sg=False):
        return nearest_embed(x, self.weight.detach() if weight_sg else self.weight)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super(ResBlock, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        layers = [
            nn.Mish(),
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.Mish(),
            nn.Conv2d(mid_channels, out_channels,
                      kernel_size=1, stride=1, padding=0)
        ]
        if bn:
            layers.insert(2, nn.BatchNorm2d(out_channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convs(x)

class PFGM(nn.Module):
    def __init__(self, d, device, k=10, bn=True, vq_coef=1, commit_coef=0.5, num_channels=3, size=(3, 256, 256), **kwargs):
        super(PFGM, self).__init__()

        self.size = size
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=bn),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=bn),
            nn.BatchNorm2d(d),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(d+(d//4)*4, d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(d),
            ResBlock(d, d),
            nn.BatchNorm2d(d),
            ResBlock(d, d),
            nn.ConvTranspose2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                d, 16, kernel_size=4, stride=2, padding=1),
        )

        self.depth_params = nn.Parameter(torch.randn(3, size[1], size[2])).unsqueeze(0).to(device)
        self.betaD_params = nn.Parameter(torch.randn(3, size[1], size[2])).unsqueeze(0).to(device)
        self.betaB_params = nn.Parameter(torch.randn(3, size[1], size[2])).unsqueeze(0).to(device)
        self.B_params = nn.Parameter(torch.randn(3, size[1], size[2])).unsqueeze(0).to(device)
        self.depth_conv = nn.Sequential(DoubleConv(3, d//4), ResBlock(d//4, d//4))
        self.betaD_conv = nn.Sequential(DoubleConv(3, d//4), ResBlock(d//4, d//4))
        self.betaB_conv = nn.Sequential(DoubleConv(3, d//4), ResBlock(d//4, d//4))
        self.B_conv = nn.Sequential(DoubleConv(3, d//4), ResBlock(d//4, d//4))

        self.d = d
        self.emb_all = NearestEmbed(k, d+d//4*4)
        self.vq_coef = vq_coef
        self.commit_coef = commit_coef
        self.mse = 0
        self.vq_loss = torch.zeros(1)
        self.commit_loss = 0

        for l in self.modules():
            if isinstance(l, nn.Linear) or isinstance(l, nn.Conv2d):
                l.weight.detach().normal_(0, 0.02)
                torch.fmod(l.weight, 0.04)
                nn.init.constant_(l.bias, 0)

        self.encoder[-1].weight.detach().fill_(1 / 40)

        self.emb_all.weight.detach().normal_(0, 0.02)
        torch.fmod(self.emb_all.weight, 0.04)


    def forward(self, x, B, depth, beta_D, beta_B):
        z_e = self.encoder(x)
        f_B = B * self.B_params.repeat(x.shape[0], 1, 1, 1)
        f_B = self.B_conv(f_B)
        f_depth = depth * self.depth_params.repeat(x.shape[0], 1, 1, 1)
        f_depth = self.depth_conv(f_depth)
        f_betaD = beta_D * self.betaD_params.repeat(x.shape[0], 1, 1, 1)
        f_betaD = self.betaD_conv(f_betaD)
        f_betaB = beta_B * self.betaB_params.repeat(x.shape[0], 1, 1, 1)
        f_betaB = self.betaB_conv(f_betaB)
        sum_e = torch.cat((z_e, f_B, f_depth, f_betaD, f_betaB), dim=1)
        sum_f, _ = self.emb_all(sum_e, weight_sg=True)


        return self.decoder(sum_f)