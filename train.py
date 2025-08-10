import torch
import os
import argparse
from torch.utils.data import DataLoader
from dataset_UIEB import UIEB_Dataset
from loss import Totaloss
import DPF_Net
from DPEM import DPEM_model
from torch import optim
import torch.optim.lr_scheduler as lr_scheduler
from Depth_Anything_V2_main.depth_anything_v2.dpt import DepthAnythingV2
from datetime import datetime
start_time = str(datetime.now())[0:19].replace(' ', '-')

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--raw_image_path', type=str,
                        help='path to the folder of images',
                        default='./UIEB/train/raw')

    parser.add_argument('--load_DPEM', type=str,
                        help='path of a pretrained DPF-Net to use',
                        default='./DPEM/checkpoint/DPEM.pth')

    parser.add_argument('--depth_anything_folder', type=str,
                        help='path of a pretrained depth_anything to use',
                        default='./Depth_Anything_V2_main')

    parser.add_argument('--lr', type=float,
                        help='learning rate of the models',
                        default=0.001)

    parser.add_argument('--batch_size', type=int,
                        default=8)

    parser.add_argument('--max_epochs', type=int,
                        default=100)

    parser.add_argument('--device', type=str,
                        help='select the device to run the models on',
                        default='cuda')

    return parser.parse_args()


def train(args):
    depth_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}}
    encoder = 'vits'
    depth_anything = DepthAnythingV2(**depth_configs[encoder])
    depth_anything.load_state_dict(torch.load(f'{args.depth_anything_folder}/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(args.device).eval()

    model = DPF_Net.TotalNetwork(args.device)
    dpem = DPEM_model.MainNet(device=args.device, imgSize=256).to(args.device).eval()
    dpem.load_state_dict(torch.load(args.load_DPEM))

    lr = args.lr
    parameters = model.get_train_parameters(lr)
    parameters = parameters + dpem.get_train_parameters(lr/200.0)
    optimizer = optim.Adam(parameters)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    criterion = Totaloss(args.device)

    dataset = UIEB_Dataset(raw_images_path=args.raw_image_path, depthanything=depth_anything, device=args.device)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    num_epochs = args.max_epochs
    lowest_train_loss = 100

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        print("------Training progress : {}/{}------".format(epoch + 1, num_epochs))
        for batch_idx, (data_raw, data_ref, data_depth, BL) in enumerate(dataloader):
            optimizer.zero_grad()
            x_B, x_beta_D, x_beta_B, x_d = dpem(data_raw, BL)
            replicated_x_B = x_B.unsqueeze(2).unsqueeze(3).repeat(1, 1, data_raw.shape[2], data_raw.shape[3])
            replicated_x_beta_D = x_beta_D.unsqueeze(2).unsqueeze(3).repeat(1, 1, data_raw.shape[2], data_raw.shape[3])
            replicated_x_beta_B = x_beta_B.unsqueeze(2).unsqueeze(3).repeat(1, 1, data_raw.shape[2], data_raw.shape[3])
            channel_replica1 = x_d[:, 0:1, :, :]
            channel_replica2 = x_d[:, 0:1, :, :]
            replicated_x_d = torch.cat((x_d, channel_replica1, channel_replica2), dim=1)

            outputs = model(data_raw, replicated_x_B, replicated_x_d, replicated_x_beta_D, replicated_x_beta_B)
            x_degraded = ((outputs * 255.0) * torch.exp(-replicated_x_d * replicated_x_beta_D) +
                          replicated_x_B * (1 - torch.exp(-replicated_x_beta_B * replicated_x_d)))/255.0
            loss = criterion(data_raw, outputs, x_degraded, data_ref, data_depth)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        print("The loss of this epoch : {:.4f}\n".format(train_loss))

        if train_loss < lowest_train_loss:
            os.makedirs('./checkpoint', exist_ok=True)
            torch.save(model.state_dict(), os.path.join('./checkpoint', start_time + '-DPF_Net.pth'))
            torch.save(dpem.state_dict(), os.path.join('./checkpoint', start_time + '-DPEM_finetune.pth'))
            lowest_train_loss = train_loss


if __name__ == '__main__':
    args = parse_args()
    train(args)