import torch
import argparse
from torch.utils.data import DataLoader
from dataset import NYU_Dataset
from DPEM_model import MainNet
from loss import Totaloss
import os
from datetime import datetime
start_time = str(datetime.now())[0:19].replace(' ', '-')

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--onland_image_path', type=str,
                        help='path to the folder of NYU-Depth-V2 images',
                        default='./NYU-Depth-V2/Img')

    parser.add_argument('--underwater_image_path', type=str,
                        help='path to the folder of underwater images',
                        default='./UIEB/train/raw')

    parser.add_argument('--depth_path', type=str,
                        help='path to the folder of relative depth',
                        default='./NYU-Depth-V2/Depth')

    parser.add_argument('--depth_scale', type=str,
                        help='path to txt of the absolute depth scale',
                        default='./NYU-Depth-V2/depth_scale.txt')

    parser.add_argument('--depth_anything_folder', type=str,
                        help='path of a pretrained depth_anything to use',
                        default='./Depth_Anything_V2_main')

    parser.add_argument('--lr', type=float,
                        help='learning rate of the models',
                        default=0.0005)

    parser.add_argument('--batch_size', type=int,
                        default=8)

    parser.add_argument('--max_epochs', type=int,
                        default=100)

    parser.add_argument('--device', type=str,
                        help='select the device to run the models on',
                        default='cuda')

    return parser.parse_args()


def train(args):
    device = args.device
    model = MainNet(device=device, imgSize=256)
    model.set_optimizer(lr=args.lr)
    criterion = Totaloss(device)
    dataset = NYU_Dataset(nyu_path=args.onland_image_path, water_path=args.underwater_image_path,
                          depth_scale=args.depth_scale, depth_path=args.depth_path, device=device, Image_size=256)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    num_epochs = args.max_epochs
    lowest_train_loss = 100

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        print("------Training progress : {}/{}------".format(epoch + 1, num_epochs))
        for batch_idx, (data_onland, degraded_img, depths, B, beta_D, beta_B, pre_B) in enumerate(dataloader):
            model.optimizer.zero_grad()
            x_B, x_beta_D, x_beta_B, x_d = model(degraded_img/255.0, pre_B/255.0)
            replicated_x_B = x_B.unsqueeze(2).unsqueeze(3).repeat(1, 1, data_onland.shape[2], data_onland.shape[3])
            replicated_x_beta_D = x_beta_D.unsqueeze(2).unsqueeze(3).repeat(1, 1, data_onland.shape[2], data_onland.shape[3])
            replicated_x_beta_B = x_beta_B.unsqueeze(2).unsqueeze(3).repeat(1, 1, data_onland.shape[2], data_onland.shape[3])
            channel_replica1 = x_d[:, 0:1, :, :]
            channel_replica2 = x_d[:, 0:1, :, :]
            replicated_x_d = torch.cat((x_d, channel_replica1, channel_replica2), dim=1)
            x_degraded = (data_onland * torch.exp(-replicated_x_d * replicated_x_beta_D) +
                          replicated_x_B * (1 - torch.exp(-replicated_x_beta_B * replicated_x_d)))
            loss = criterion(degraded_img, x_degraded, B, beta_D, beta_B, depths, x_B, x_beta_D, x_beta_B, x_d)
            loss.backward()
            model.optimizer.step()
            train_loss += loss.item()
        print("The loss of this epoch : {:.4f}\n".format(train_loss))

        if train_loss < lowest_train_loss:
            os.makedirs('./checkpoint', exist_ok=True)
            torch.save(model.state_dict(), os.path.join('./checkpoint', start_time + '-DPEM.pth'))
            lowest_train_loss = train_loss


if __name__ == '__main__':
    args = parse_args()
    train(args)
