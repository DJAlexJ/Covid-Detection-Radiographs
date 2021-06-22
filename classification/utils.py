import os
import sys

import cv2
import numpy as np
import pandas as pd
import pydicom
import timm
import torch
import torch.nn as nn
from albumentations import (CLAHE, Blur, CenterCrop, CoarseDropout, Compose,
                            Cutout, Flip, GaussNoise, GridDistortion,
                            HorizontalFlip, HueSaturationValue,
                            IAAAdditiveGaussianNoise, IAAEmboss,
                            IAAPerspective, IAAPiecewiseAffine, IAASharpen,
                            MedianBlur, MotionBlur, Normalize, OneOf,
                            OpticalDistortion, RandomBrightnessContrast,
                            RandomResizedCrop, RandomRotate90, Resize,
                            ShiftScaleRotate, Transpose, VerticalFlip)
from albumentations.pytorch import ToTensorV2
from pydicom.pixel_data_handlers.util import apply_voi_lut
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch import optim
from torch.utils.data import Dataset
from tqdm import tqdm


def get_train_transforms(img_size: int):
    return Compose(
        [
            Resize(img_size, img_size),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )


def get_valid_transforms(img_size: int):
    return Compose(
        [
            Resize(img_size, img_size),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )


class ClassificationDataset(Dataset):
    def __init__(self, df, transforms, label_col):
        super().__init__()
        self.df = df
        self.transforms = transforms
        self.paths = self.df["path"].values
        self.labels = self.df[label_col].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image = (
            cv2.imread(self.paths[index], cv2.IMREAD_COLOR)
            .copy()
            .astype(np.float32)
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        label = self.labels[index]
        if self.transforms:
            image = self.transforms(image=image)["image"]
        return image, label


def get_train_val_split(data_dir, fold):
    data = pd.read_csv(data_dir)
    col = f"fold{fold}"
    return data[data[col] == "train"], data[data[col] == "val"]


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
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    filename = cfg.base_name.format(fold=fold, epoch=epoch)
    checkpoint_dir = cfg.checkpoint_dir
    torch.save(state, checkpoint_dir.format(filename=filename))


def get_roc_auc_score(actual, preds):
    raise NotImplementedError("Yet to be implemented")
    preds = nn.functional.softmax(preds, dim=1)
    preds = preds.detach().cpu().numpy()
    actual = [actual.cpu().numpy()]
    print(actual, preds)

    return roc_auc_score(actual, preds, multi_class="ovr")
