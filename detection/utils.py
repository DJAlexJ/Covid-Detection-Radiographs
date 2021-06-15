import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader

import augmentations as aug

from dataset import DetectionDataset
from config import DefaultConfig


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_train_file_path(image_id):
    return f"{DefaultConfig.train_dir}{image_id}.png"


def get_test_file_path(image_id):
    return f"{DefaultConfig.test_dir}{image_id}.png"


def get_train_dataset(fold_number, df_folds, train):
    return DetectionDataset(
        image_ids=df_folds[df_folds['fold'] != fold_number].index.values,
        df=train,
        transforms=aug.get_train_transforms()
    )


def get_validation_dataset(fold_number, df_folds, train):
    return DetectionDataset(
        image_ids=df_folds[df_folds['fold'] == fold_number].index.values,
        df=train,
        transforms=aug.get_valid_transforms()
    )


def get_train_data_loader(train_dataset, batch_size=16):
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )


def get_validation_data_loader(valid_dataset, batch_size=16):
    return DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )


def collate_fn(batch):
    return tuple(zip(*batch))


def save_checkpoint(epoch, model, optimizer, cfg, fold):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    filename = cfg.base_name.format(fold=fold, epoch=epoch)
    checkpoint_dir = cfg.checkpoint_dir
    torch.save(state, checkpoint_dir.format(filename=filename))

