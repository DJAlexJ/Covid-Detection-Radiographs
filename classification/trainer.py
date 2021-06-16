import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import timm
import gc

from utils import (
    ClassificationDataset,
    get_train_transforms,
    get_valid_transforms,
    get_train_val_split,
    Logger, save_checkpoint,
    get_roc_auc_score)

from config import  DefaultConfig


class ClassificationTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim,
        criterion,
        config: DefaultConfig,
        scheduler=None,
        fold: int = 0,
        grad_accum: int = 1
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.config = config
        self.logger = Logger(config.logdir + f"/{fold}")
        self.fold = fold
        self.grad_accum = grad_accum

    def get_loader(self, config):
        train_data, val_data = get_train_val_split(config.csv_file, self.fold)
        train_dataset = ClassificationDataset(
                train_data, get_train_transforms(config.img_size)
        )
        self.train_loader = DataLoader(train_dataset, **config.loader_params)
        val_dataset = ClassificationDataset(
            val_data, get_valid_transforms(config.img_size)
        )
        self.val_loader = DataLoader(val_dataset, **config.loader_params)

    def train(self, n_epoch: int):
        self.get_loader(self.config)
        self.iter_cntr = 0
        fold = self.config.fold_num
        for i in range(1, n_epoch+1):
            self.train_one_epoch(i, self.train_loader)
            save_checkpoint(i, self.model, self.optimizer, self.config, fold)
            self.validate_one_epoch(i, self.val_loader)
            self.scheduler.step()

    def train_one_epoch(self, epoch: int, loader: DataLoader):
        print(f'Training at epoch: {epoch}') 
        self.model.train()
        self.correct = 0
        self.total = 0
        self.losses = []
        self.optimizer.zero_grad()
        tk0 = tqdm(enumerate(loader), total=len(loader))
        for i, (images, labels) in enumerate(loader):
            images = images.cuda()
            labels = labels.cuda()
            outputs = self.model(images)
            _,pred = torch.max(outputs, dim=1)
            self.correct += torch.sum(pred==labels).item()
            self.total += labels.size(0)
            loss = self.criterion(outputs, labels)
            loss = loss / self.grad_accum
            loss.backward()
            if (i+1) % self.grad_accum == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            self.losses.append(loss.item())

            if self.logger:
                self.logger.log("Loss/train", loss.item())
                self.logger.log("Acc/train", self.correct/self.total)
            tk0.set_postfix(Train_Loss=np.mean(self.losses),
                            Epoch=epoch, 
                            LR=self.optimizer.param_groups[0]['lr'])
            tk0.update(1)

    @torch.no_grad()
    def validate_one_epoch(self, epoch, loader):
        self.correct = 0
        self.total = 0
        self.losses = []
        print(f"Starting validate at epoch: {epoch}")
        self.model.eval()
        tk0 = tqdm(enumerate(loader), total=len(loader))
        for _, (images, labels) in enumerate(loader):
            images = images.cuda()
            labels = labels.cuda()
            outputs = self.model(images)
            _,pred = torch.max(outputs, dim=1)
            self.correct += torch.sum(pred==labels).item()
            self.total += labels.size(0)
            loss = self.criterion(outputs, labels)
            self.losses.append(loss.item())
            val_acc = self.correct/self.total
            # auc = get_roc_auc_score(labels, outputs)
            if self.logger:
                self.logger.log("Loss/val", loss.item())
                self.logger.log("Acc/val", val_acc)
            # self.logger.log("ROC AUC/val", auc)
            tk0.set_postfix(Val_Loss=np.mean(self.losses),
                            Epoch=epoch,
                            Val_acc=val_acc,
                            LR=self.optimizer.param_groups[0]['lr'])
            tk0.update(1)
