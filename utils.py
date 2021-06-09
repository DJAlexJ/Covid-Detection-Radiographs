import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader

import augmentations as aug

from dataset import CustomDataset


def get_train_file_path(image_id):
    return "./data512/train/{}.png".format(image_id)


def get_test_file_path(image_id):
    return "./data512/test/{}.png".format(image_id)


def get_train_dataset(fold_number, df_folds, train):
    return CustomDataset(
        image_ids=df_folds[df_folds['fold'] != fold_number].index.values,
        df=train,
        transforms=aug.get_train_transforms()
    )


def get_validation_dataset(fold_number, df_folds, train):
    return CustomDataset(
        image_ids=df_folds[df_folds['fold'] == fold_number].index.values,
        df=train,
        transforms=aug.get_valid_transforms()
    )


def get_train_data_loader(train_dataset, batch_size=16):
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )


def get_validation_data_loader(valid_dataset, batch_size=16):
    return DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )


def collate_fn(batch):
    return tuple(zip(*batch))


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
