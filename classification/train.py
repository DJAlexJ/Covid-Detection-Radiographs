import argparse
import os

from trainer import ClassificationTrainer
from config import SwinTrainConfig as cfg
import torch
import torch.nn as nn
import timm
import pandas as pd

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device", type=str, default=str(0))
    args = parser.parse_args()
    fold = args.fold
    device = args.device
    os.environ['CUDA_VISIBLE_DEVICES'] = device

    model = 'swin_large_patch4_window12_384'
    criterion = nn.CrossEntropyLoss()
    optimizer = cfg.optimizer
    trainer = ClassificationTrainer(model, optimizer, criterion, cfg, fold)
    n_epochs = cfg.n_epochs
    trainer.train(n_epochs)
