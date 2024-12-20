import torch
import os
import argparse
from torch.utils.data import DataLoader
from dataset_UIEB import UIEB_Dataset
from loss import Totaloss
from PIL import Image
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
                        default='./UIEB/test/raw')

    parser.add_argument('--load_DPF_Net', type=str,
                        help='path of a pretrained DPF-Net to use',
                        default='./checkpoint/DPF-Net.pth')

    parser.add_argument('--load_DPEM', type=str,
                        help='path of a pretrained DPF-Net to use',
                        default='./checkpoint/DPEM_finetune.pth')

    parser.add_argument('--depth_anything_folder', type=str,
                        help='path of a pretrained depth_anything to use',
                        default='./Depth_Anything_V2_main')

    parser.add_argument('--device', type=str,
                        help='select the device to run the models on',
                        default='cuda')

    return parser.parse_args()


def test(args):
    depth_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}}
    encoder = 'vits'
    depth_anything = DepthAnythingV2(**depth_configs[encoder])
    depth_anything.load_state_dict(torch.load(f'{args.depth_anything_folder}/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(args.device).eval()

    model = DPF_Net.TotalNetwork(args.device).eval()
    model.load_state_dict(torch.load(args.load_DPF_Net))
    dpem = DPEM_model.MainNet(device=args.device, imgSize=256).to(args.device).eval()
    dpem.load_state_dict(torch.load(args.load_DPEM))

    dataset = UIEB_Dataset(raw_images_path=args.raw_image_path, depthanything=depth_anything, device=args.device, isTrain=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    print('Test start')
    for batch_idx, (data_raw, data_depth, BL, file_name) in enumerate(dataloader):
        x_B, x_beta_D, x_beta_B, x_d = dpem(data_raw, BL)
        replicated_x_B = x_B.unsqueeze(2).unsqueeze(3).repeat(1, 1, data_raw.shape[2], data_raw.shape[3])
        replicated_x_beta_D = x_beta_D.unsqueeze(2).unsqueeze(3).repeat(1, 1, data_raw.shape[2], data_raw.shape[3])
        replicated_x_beta_B = x_beta_B.unsqueeze(2).unsqueeze(3).repeat(1, 1, data_raw.shape[2], data_raw.shape[3])
        channel_replica1 = x_d[:, 0:1, :, :]
        channel_replica2 = x_d[:, 0:1, :, :]
        replicated_x_d = torch.cat((x_d, channel_replica1, channel_replica2), dim=1)

        outputs = model(data_raw, replicated_x_B, replicated_x_d, replicated_x_beta_D, replicated_x_beta_B)
        enc_img = (outputs[0] * 255).to('cpu', dtype=torch.uint8).permute(1, 2, 0)
        img_save = Image.fromarray(enc_img.numpy())
        os.makedirs('./out_images', exist_ok=True)
        save_name = str(file_name)[2:-3]
        img_save.save(os.path.join('./out_images', save_name+'.jpg'))
        print("\r------Images saved successfully : {}/{}".format(batch_idx + 1, len(dataloader)), end="")
    print('\nTest completed')


if __name__ == '__main__':
    args = parse_args()
    test(args)
