import sys
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from tqdm import tqdm
import timm
import torch
import torch.nn as nn
from torch import optim
import pandas as pd
import numpy as np
import os

from torch.utils.data import Dataset
import cv2

from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE,
    RandomRotate90, Transpose, ShiftScaleRotate, Blur, OpticalDistortion,
    GridDistortion, HueSaturationValue, IAAAdditiveGaussianNoise, GaussNoise, 
    MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop, IAASharpen,
    IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize,
    Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_size: int):
    return Compose([
            Resize(img_size, img_size),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2,
                            sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1),
                            contrast_limit=(-0.1, 0.1), p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            CoarseDropout(p=0.5),
            Cutout(p=0.5),
            ToTensorV2(p=1.0),
        ], p=1.)


def get_valid_transforms(img_size: int):
    return Compose([
            CenterCrop(img_size, img_size, p=1.),
            Resize(img_size, img_size),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)


class ClassificationDataset(Dataset):
    def __init__(self, df, transforms):
        super().__init__()
        self.df = df
        self.transforms = transforms
        self.paths = self.df['path'].values
        self.labels = self.df['label'].values

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        image = cv2.imread(self.paths[index], cv2.IMREAD_COLOR).copy().astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        label = self.labels[index]
        if self.transforms:
            image = self.transforms(image=image)['image']
        return image, label


def get_train_val_split(config):
    data = pd.read_csv(config.csv_file)
    return data, None


class Logger(object):
    def __init__(self, logdir):
        from collections import defaultdict
        from torch.utils.tensorboard import SummaryWriter

        self.step_map = defaultdict(int)
        self.writer = SummaryWriter(logdir)

    def log(self, key, value):
        self.writer.add_scalar(key, value, self.step_map[key])
        self.step_map[key] += 1


def save_checkpoint(epoch, model, optimizer, cfg, fold):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    filename = cfg.base_name.format(fold=fold, epoch=epoch)
    checkpoint_dir = cfg.checkpoint_dir
    if checkpoint_dir[-1] != '/':
        checkpoint_dir += '/'
    torch.save(state, checkpoint_dir + filename)
