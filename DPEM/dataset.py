import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import random
import cv2

random.seed(1145)

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

def augmentation(img1, img2):
    hflip = random.random() < 0.5
    vflip = random.random() < 0.5
    rot = random.random() < 0.3
    angle = random.random() * 180 - 90
    if hflip:
        img1 = cv2.flip(img1, 1)
        img2 = cv2.flip(img2, 1)
    if vflip:
        img1 = cv2.flip(img1, 0)
        img2 = cv2.flip(img2, 0)
    if rot:
        img1 = img_rotate(img1, angle)
        img2 = img_rotate(img2, angle)
    return img1, img2
#==========================augmentation==========================
def expand_dimension(factor, img_size):
    factor_r = np.full(img_size, factor[0])
    factor_g = np.full(img_size, factor[1])
    factor_b = np.full(img_size, factor[2])
    return np.stack((factor_r, factor_g, factor_b), axis=-1)

def disp_to_depth(disp, min_depth, max_depth):
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return depth

def preprocess(img1, img2, isTrain):
    if isTrain:
        img1 = np.uint8((np.asarray(img1)))
        img2 = np.uint8((np.asarray(img2)))
        img1, img2 = augmentation(img1, img2)
    else:
        img1 = np.uint8((np.asarray(img1)))
        img2 = np.uint8((np.asarray(img2)))
    return img1, img2
def populate_list(raw_images_path):
    image_list_raw = glob.glob(raw_images_path + "/*.jpg")
    train_list = sorted(image_list_raw)
    return train_list

def get_depth_scale(file_path):
    data_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 3:
                image_path = parts[0]
                max_value = round(float(parts[1]), 1)
                min_value = round(float(parts[2]), 1)
                data_dict[image_path] = [max_value, min_value]
    return data_dict

def adjust_B(b, g, num=130):
    if b > num and g > num:
        if b > g:
            g = num * g/b
            b = num
        else:
            b = num * b/g
            g = num

    elif b > num:
        b = num
    elif g > num:
        g = num

    return b, g

def get_B(bl_G, a, b):
    bl_R = bl_G * a[1]/a[2] * b[2]/b[1]
    bl_B = bl_G * a[1]/a[0] * b[0]/b[1]
    bl_B, bl_G = adjust_B(bl_B, bl_G)
    return np.array([bl_B, bl_G, bl_R])

def Jerlov(category):
    if category == 1:       # Jerlov I
        a = [0.022, 0.049, 0.341]
        b = [3.81e-3, 2.05e-3, 8.99e-4]
    if category == 2:       # Jerlov IA
        a = [0.0264, 0.0503, 0.342]
        b = [6.31e-3, 4.02e-3, 2.34e-3]
    if category == 3:       # Jerlov IB
        a = [0.0342, 0.0572, 0.349]
        b = [0.068, 0.0565, 0.045]
    if category == 4:       # Jerlov II
        a = [0.062, 0.0845, 0.375]
        b = [0.504, 0.387, 0.27]
    if category == 5:       # Jerlov III
        a = [0.124, 0.129, 0.426]
        b = [1.38, 1.06, 0.737]
    if category == 6:       # Jerlov IC
        a = [0.179, 0.121, 0.439]
        b = [0.514, 0.395, 0.274]
    if category == 7:       # Jerlov 3C
        a = [0.319, 0.187, 0.498]
        b = [1.5, 1.15, 0.8]
    if category == 8:       # Jerlov 5C
        a = [0.535, 0.277, 2.32]
        b = [1.87, 1.44, 2.87]
    if category == 9:       # Jerlov 7C
        a = [0.924, 0.47, 0.635]
        b = [3.3, 2.54, 1.77]
    if category == 10:      # Jerlov 9C
        a = [1.56, 0.826, 0.775]
        b = [4.39, 3.38, 2.35]
    return a, b

def convert_img(img, device):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img)
    return img.to(device, dtype=torch.float32).permute(2, 0, 1)
def convert_parameter(p, device):
    new_p = np.asarray(p)[[2, 1, 0]]
    new_p = torch.from_numpy(new_p)
    return new_p.to(device, dtype=torch.float32)

def img_correct(img):
    max_color = img.max()
    min_color = img.min()
    if max_color <= 255.0 and min_color >= 0.0:
        return img

    normalized_img = (img - min_color) / (max_color - min_color)
    if max_color <= 255.0 and min_color <= 0.0:
        scaled_image = normalized_img * max_color
    if max_color >= 255.0 and min_color >= 0.0:
        scaled_image = normalized_img * (255 - min_color) + min_color
    if max_color >= 255.0 and min_color <= 0.0:
        scaled_image = normalized_img * 255
    return scaled_image

def pre_B_estimate(raw, device, isThree=False):
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

    if isThree:
        bgl_B, bgl_G = adjust_B(bgl_B, bgl_G)
        return np.array([bgl_B, bgl_G, bgl_R])

    bgl[..., 0] = bgl_B
    bgl[..., 1] = bgl_G
    bgl[..., 2] = bgl_R

    bgl = cv2.cvtColor(bgl, cv2.COLOR_BGR2RGB)
    bgl = torch.from_numpy(bgl)
    return bgl.to(device, dtype=torch.float32).permute(2, 0, 1)


class NYU_Dataset(Dataset):
    def __init__(self, device, nyu_path, water_path, depth_scale, depth_path, Image_size=256, isTrain=True):
        self.onland_list = populate_list(nyu_path)
        self.depth_list = populate_list(depth_path)
        self.water_list = populate_list(water_path)
        self.depth_scale = get_depth_scale(depth_scale)
        self.size = Image_size
        self.isTrain = isTrain
        self.device = device

    def __getitem__(self, index):
        data_onland_path = self.onland_list[index]
        data_onland = cv2.imread(data_onland_path)
        data_onland = cv2.resize(data_onland, (self.size, self.size), interpolation=cv2.INTER_LINEAR)

        data_depth_path = self.depth_list[index]
        data_depth = cv2.imread(data_depth_path, cv2.COLOR_BGR2GRAY)
        data_depth = cv2.resize(data_depth, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        data_onland, data_depth = preprocess(data_onland, data_depth, self.isTrain)
        data_depth = np.float32(data_depth) / 255.0

        max_depth = self.depth_scale[data_onland_path][0]
        min_depth = self.depth_scale[data_onland_path][1]
        abs_depths = (np.max(data_depth) - data_depth + np.min(data_depth)) * (max_depth - min_depth) + min_depth

        category = np.random.randint(0, 10)
        beta_D, beta_B = Jerlov(category + 1)
        underwater_img = cv2.imread(self.water_list[np.random.randint(0, len(self.water_list))])
        B = pre_B_estimate(underwater_img, self.device, True)
        B = get_B(B[1], beta_D, beta_B)
        degraded_img = (data_onland * np.exp(-expand_dimension(beta_D, self.size) * np.expand_dims(abs_depths, axis=2)) +
                        expand_dimension(B, self.size) * (1 - np.exp(-expand_dimension(beta_B, self.size) * np.expand_dims(abs_depths, axis=2))))
        degraded_img = img_correct(degraded_img)

        data_onland = convert_img(data_onland, self.device)
        pre_B = pre_B_estimate(degraded_img.astype(np.uint8), self.device)
        degraded_img = convert_img(degraded_img.astype(np.uint8), self.device)
        beta_D = convert_parameter(beta_D, self.device)
        beta_B = convert_parameter(beta_B, self.device)
        B = convert_parameter(B, self.device)
        depths = torch.from_numpy(abs_depths).unsqueeze(0)
        depths = depths.to(self.device, dtype=torch.float32)

        return data_onland, degraded_img, depths, B, beta_D, beta_B, pre_B

    def __len__(self):
        return len(self.onland_list)


