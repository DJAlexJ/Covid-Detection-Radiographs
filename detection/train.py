import argparse
import os

from trainer import DetectionTrainer
from config import TrainGlobalConfig as cfg
from config import DefaultConfig
import torch
import torch.nn as nn
import pandas as pd
from utils import seed_everything

if __name__ == '__main__':
    seed_everything(DefaultConfig.seed)

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=str(0))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--debug", type=bool, default=False)
    args = parser.parse_args()
    device = args.device
    fold = args.fold
    debug = args.debug

    os.environ['CUDA_VISIBLE_DEVICES'] = device
    print(f"available GPU devices: {device}")

    optimizer = cfg.optimizer
    scheduler = cfg.scheduler
    trainer = DetectionTrainer(optimizer, scheduler, cfg,
                               fold=fold, pretrained=True,
                               DEBUG=debug
                               )
    n_epochs = cfg.n_epochs
    trainer.train(n_epochs)
