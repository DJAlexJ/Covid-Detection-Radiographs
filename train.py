import os
import numpy as np
import pandas as pd
from datetime import datetime
import time
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

# torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler

# torchvision
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

# sklearn
from sklearn.model_selection import StratifiedKFold

# CV
import cv2

# Albumenatations
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# from pycocotools.coco import COCO
from sklearn.model_selection import StratifiedKFold

# glob
from glob import glob

# numba
import numba
from numba import jit

from config import DefaultConfig, TrainGlobalConfig
from dataset import CustomDataset
from augmentations import get_train_transforms, get_valid_transforms
from models import FasterRCNNDetector, get_model
from metrics import *
import utils


class Fitter:
    def __init__(self, model, device, config):
        self.config = config
        self.epoch = 0

        self.base_dir = f'./{config.folder}'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        self.log_path = f'{self.base_dir}/log.txt'
        self.best_summary_loss = 10 ** 5
        self.best_score = 0

        self.model = model
        self.device = device

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        # get the configured optimizer
        if Adam:
            self.optimizer = torch.optim.Adam(self.model.parameters(), **Adam_config)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), **SGD_config)

        self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)
        self.log(f'Fitter prepared. Device is {self.device}')
        self.log(f'Fold num is {DefaultConfig.fold_num}')

    def fit(self, train_loader, validation_loader):
        for e in range(self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr}')

            t = time.time()
            summary_loss = self.train_one_epoch(train_loader)

            if e == 0:
                self.best_summary_loss = summary_loss.avg

            self.log(
                f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')
            self.save(f'{self.base_dir}/last-checkpoint.bin')

            t = time.time()
            _, eval_scores = self.validation(validation_loader)

            self.log(
                f'[RESULT]: Val. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, image_precision: {eval_scores.avg:.5f}, time: {(time.time() - t):.5f}')

            # if summary_loss.avg < self.best_summary_loss:
            if eval_scores.avg > self.best_score:
                self.best_summary_loss = summary_loss.avg
                self.best_score = eval_scores.avg
                self.model.eval()
                self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
                for path in sorted(glob(f'{self.base_dir}/best-checkpoint-*epoch.bin'))[:-3]:
                    os.remove(path)

            if self.config.validation_scheduler:
                self.scheduler.step(metrics=eval_scores.avg)
                # self.scheduler.step(metrics=summary_loss.avg)

            self.epoch += 1

    def validation(self, val_loader):
        self.model.eval()

        # model.eval() mode --> it will return boxes and scores.
        # in this part, just print train_loss
        summary_loss = AverageMeter()
        summary_loss.update(self.best_summary_loss, self.config.batch_size)

        eval_scores = EvalMeter()
        validation_image_precisions = []

        t = time.time()
        for step, (images, targets, image_ids) in enumerate(val_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Val Step {step}/{len(val_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, ' + \
                        f'val_precision: {eval_scores.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            with torch.no_grad():
                images = torch.stack(images)
                batch_size = images.shape[0]
                images = images.to(self.device).float()
                labels = [target['labels'].float() for target in targets]

                """
                In model.train() mode, model(images)  is returning losses.
                We are using model.eval() mode --> it will return boxes and scores. 
                """
                outputs = self.model(images)

                for i, image in enumerate(images):
                    gt_boxes = targets[i]['boxes'].data.cpu().numpy()
                    boxes = outputs[i]['boxes'].data.cpu().numpy()
                    scores = outputs[i]['scores'].detach().cpu().numpy()

                    preds_sorted_idx = np.argsort(scores)[::-1]
                    preds_sorted_boxes = boxes[preds_sorted_idx]

                    eval_scores.update(pred_boxes=preds_sorted_boxes, gt_boxes=gt_boxes)

        return summary_loss, eval_scores

    def train_one_epoch(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()

        t = time.time()
        for step, (images, targets, image_ids) in enumerate(train_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Train Step {step}/{len(train_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )

            images = torch.stack(images)
            images = images.to(self.device).float()
            batch_size = images.shape[0]
            boxes = [target['boxes'].to(self.device).float() for target in targets]
            labels = [target['labels'].to(self.device).float() for target in targets]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            self.optimizer.zero_grad()

            outputs = self.model(images, targets)

            loss = sum(loss for loss in outputs.values())

            loss.backward()

            summary_loss.update(loss.detach().item(), batch_size)

            self.optimizer.step()

            if self.config.step_scheduler:
                self.scheduler.step()

        return summary_loss

    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),  # 'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1

    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')


def run_training(net, fold=0, df_folds=None, train=None):
    net.to(device)

    train_dataset = utils.get_train_dataset(fold_number=fold,
                                            df_folds=df_folds,
                                            train=train
                                            )
    train_data_loader = utils.get_train_data_loader(
        train_dataset,
        batch_size=TrainGlobalConfig.batch_size
    )

    validation_dataset = utils.get_validation_dataset(fold_number=fold,
                                                      df_folds=df_folds,
                                                      train=train
                                                      )
    validation_data_loader = utils.get_validation_data_loader(
        validation_dataset,
        batch_size=TrainGlobalConfig.batch_size
    )

    fitter = Fitter(model=net, device=DefaultConfig.device, config=TrainGlobalConfig)
    fitter.fit(train_data_loader, validation_data_loader)


os.environ['CUDA_VISIBLE_DEVICES'] = '7'
device = torch.device(DefaultConfig.device) if torch.cuda.is_available() else torch.device('cpu')

Adam = False
if Adam:
    Adam_config = {"lr": 0.001, "betas": (0.9, 0.999), "eps": 1e-08}
else:
    SGD_config = {"lr": 0.001, "momentum": 0.9, "weight_decay": 0.001}

utils.seed_everything(DefaultConfig.seed)

df = pd.read_csv('./data/updated_train_labels.csv')
df['jpg_path'] = df['id'].apply(utils.get_train_file_path)
train = df.copy()

df_folds = train.copy()
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=DefaultConfig.seed)

# Готовим фолды
for n, (train_index, val_index) in enumerate(skf.split(X=df_folds.index, y=df_folds.integer_label)):
    df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = int(n)

df_folds['fold'] = df_folds['fold'].astype(int)

df_folds = df_folds.set_index('id')

net = get_model(pretrained=True)

run_training(net,
             fold=DefaultConfig.fold_num,
             df_folds=df_folds,
             train=train
)
