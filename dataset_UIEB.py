import os
import sys

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
import glob
import random
import cv2


#==========================augmentation==========================
def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def img_rotate(img, angle, center=None, scale=1.0):
    (h, w) = img.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_img = cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
                                 borderValue=(0, 0, 0), )
    return rotated_img

def augmentation(imgs):
    hflip = random.random() < 0.5
    vflip = random.random() < 0.5
    rot = random.random() < 0.3
    angle = random.random() * 180 - 90
    if hflip:
        for i in range(len(imgs)):
            imgs[i] = cv2.flip(imgs[i], 1)
    if vflip:
        for i in range(len(imgs)):
            imgs[i] = cv2.flip(imgs[i], 0)
    if rot:
        for i in range(len(imgs)):
            imgs[i] = img_rotate(imgs[i], angle)
    return imgs
#==========================augmentation==========================

def pre_B_estimate(raw, device):
    bgl = np.zeros_like(raw)
    raw = np.transpose(raw, (2, 0, 1))

    for i in range(3):
        raw[i][raw[i] < 5] = 5
        raw[i][raw[i] > 250] = 250

    avg_B = np.mean(raw[0])
    std_B = np.std(raw[0])
    bgl_B = 1.13 * avg_B + 1.11 * std_B - 25.6

    avg_G = np.mean(raw[1])
    std_G = np.std(raw[1])
    bgl_G = 1.13 * avg_G + 1.11 * std_G - 25.6

    med_R = np.median(raw[2])
    bgl_R = 140 / (1 + 14.4 * np.exp(-0.034 * med_R))

    bgl[..., 0] = bgl_R
    bgl[..., 1] = bgl_G
    bgl[..., 2] = bgl_B

    bgl = torch.from_numpy(bgl / 255.0)
    return bgl.to(device, dtype=torch.float32).permute(2, 0, 1)

def preprocess(imgs, device, isTrain):
    '''
    imgs[0]:raw image
    imgs[1]:depth map
    imgs[2]:ref image(if exist)
    '''
    BL = pre_B_estimate(imgs[0], device)
    imgs[0] = cv2.cvtColor(imgs[0], cv2.COLOR_BGR2RGB)
    if isTrain:
        imgs[2] = cv2.cvtColor(imgs[2], cv2.COLOR_BGR2RGB)
        imgs = augmentation(imgs)
        data_raw = torch.from_numpy(imgs[0]/255.0)
        data_ref = torch.from_numpy(imgs[2]/255.0)
        data_depth = torch.from_numpy(imgs[1])
        return (data_raw.to(device, dtype=torch.float32).permute(2, 0, 1),
                data_ref.to(device, dtype=torch.float32).permute(2, 0, 1),
                data_depth.to(device, dtype=torch.float32).unsqueeze(0), BL)
    else:
        data_raw = torch.from_numpy(imgs[0] / 255.0)
        data_depth = torch.from_numpy(imgs[1])
        return (data_raw.to(device, dtype=torch.float32).permute(2, 0, 1),
                data_depth.to(device, dtype=torch.float32).unsqueeze(0), BL)

def populate_raw_list(raw_images_path):
    image_list_raw = glob.glob(raw_images_path + "/*.jpg")
    train_list = sorted(image_list_raw)

    return train_list

class UIEB_Dataset(Dataset):
    def __init__(self, raw_images_path, depthanything, device, Image_size=256, isTrain=True):
        self.raw_list = populate_raw_list(raw_images_path)
        raw_path = self.raw_list
        if isTrain:
            self.ref_list = [s.replace("raw", "ref") for s in raw_path]
        self.depthanything = depthanything
        self.size = Image_size
        self.isTrain = isTrain
        self.device = device

        print("Total images:", len(self.raw_list))

    def __getitem__(self, index):
        data_raw_path = self.raw_list[index]
        file_name = data_raw_path.split('/')[-1].split('.')[0]
        data_raw = cv2.imread(data_raw_path)
        data_raw = cv2.resize(data_raw, (self.size, self.size), interpolation=cv2.INTER_LINEAR)

        disp = self.depthanything.infer_image(data_raw)
        far_mask = (disp == 0)
        min_disp = np.min(disp[~far_mask])
        disp[far_mask] = min_disp
        data_depth = (disp.max() - disp) / ((disp.max() - disp.min()))

        if self.isTrain:
            data_ref_path = self.ref_list[index]
            data_ref = cv2.imread(data_ref_path)
            data_ref = cv2.resize(data_ref, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            data_raw, data_ref, data_depth, bl = preprocess([data_raw, data_depth, data_ref], self.device, self.isTrain)
            return data_raw, data_ref, data_depth, bl

        else:
            data_raw, data_depth, bl = preprocess([data_raw, data_depth], self.device, self.isTrain)
            return data_raw, data_depth, bl, file_name



    def __len__(self):
        return len(self.raw_list)


