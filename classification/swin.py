from torch.utils.data.dataloader import DataLoader
from .config import SwinTrainConfig as cfg


import sys
import os

import numpy as np
from tqdm import tqdm
import timm
import torch
import horovod.torch as hvd

import torch.nn as nn
from torch import optim
import pandas as pd
import numpy as np
from .utils import (ClassificationDataset,
                    get_train_transforms, 
                    get_valid_transforms)
from sklearn.model_selection import StratifiedKFold


def train_one_epoch(epoch_num, model, loader, validation_ids):
    print(f'Epoch Num: {epoch_num}')
    model.train()
    for i, (images, labels) in enumerate(loader):
        images = images.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        outputs = model(images)
        
        _,pred = torch.max(outputs, dim=1)
        correct += torch.sum(pred==labels).item()
        total += labels.size(0)
        
        loss = criterion(outputs, labels)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        if i % 40 == 0:
            print(f"Acc: {correct / total}")
            print(f"Loss: {np.mean(losses)}")
    return model


if __name__ == '__main__':
    model = timm.create_model('swin_large_patch4_window12_384', pretrained=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.devices)

    # hvd.init()
    # torch.cuda.set_device(hvd.local_rank())
    # torch.set_num_threads(cfg.loader_params['num_workers'])

    num_ftrs = model.head.in_features
    model.head = nn.Linear(num_ftrs, 4)
    model.cuda()
    model.eval()

    criterion = nn.CrossEntropyLoss()
    optimizer = cfg.optimizer(model.parameters(), **cfg.optimizer_params)
    data = pd.read_csv(cfg.csv_file)
    train_data, validation_data = get_train_val_data(data)

    train_dataset = ClassificationDataset(train_data, get_train_transforms())
    train_loader = DataLoader(train_dataset, **cfg.loader_params)
    
    losses = []
    correct = 0
    total = 0
    for epoch in range(1, cfg.n_epochs + 1):
        model = train_one_epoch(epoch, model, train_loader)

    