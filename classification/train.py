import argparse
import os

import pandas as pd
import timm
import torch
import torch.nn as nn
from config import TrainConfig as cfg
from models import BaseModel
from torch.optim import lr_scheduler
from trainer import ClassificationTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device", type=str, default=str(0))
    args = parser.parse_args()
    fold = args.fold
    device = args.device
    os.environ["CUDA_VISIBLE_DEVICES"] = device

    # model_name = 'resnet50'
    model_name = "swin_large_patch4_window12_384"
    model = BaseModel(model_name, num_classes=4, pretrained=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = cfg.optimizer(model.parameters(), **cfg.optimizer_params)

    scheduler_params = dict(T_0=7, T_mult=1, eta_min=1e-6, last_epoch=-1)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, **scheduler_params
    )

    trainer = ClassificationTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        config=cfg,
        fold=fold,
        grad_accum=cfg.grad_accum,
    )
    n_epochs = cfg.n_epochs
    trainer.train(n_epochs)
