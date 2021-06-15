import argparse
import os

import torch
import torch.nn as nn
import timm
import pandas as pd

from models import BaseModel
from trainer import ClassificationTrainer
from config import SwinTrainConfig as cfg


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device", type=str, default=str(0))
    args = parser.parse_args()
    fold = args.fold
    device = args.device
    os.environ['CUDA_VISIBLE_DEVICES'] = device

    model_name = 'swin_large_patch4_window12_384'
    model = BaseModel(model_name, num_classes=4, pretrained=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = cfg.optimizer(model.parameters(), **cfg.optimizer_params)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.85)

    trainer = ClassificationTrainer(model, optimizer, criterion,
                                    cfg, fold=fold)
    n_epochs = cfg.n_epochs
    trainer.train(n_epochs)
