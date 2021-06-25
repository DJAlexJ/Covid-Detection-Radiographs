import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import pandas as pd
import numpy as np

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

from tqdm import tqdm

from models import FasterRCNNDetector, Logger, get_faster_rcnn
from config import DefaultConfig, TrainGlobalConfig as train_cfg
from metrics import calculate_map
from utils import (
    get_train_file_path, get_train_dataset,
    get_train_data_loader, get_validation_dataset,
    get_validation_data_loader, save_checkpoint,
    return_data_w_folds
)
import gc


class DetectionTrainer:
    def __init__(
            self,
            optimizer: optim,
            scheduler: optim,
            config: DefaultConfig,
            fold: int = 0,
            pretrained: bool = True,
            DEBUG: bool = True
    ):

        self.model = get_faster_rcnn(pretrained=pretrained)
        self.model.cuda()
        self.optimizer = optimizer(self.model.parameters(), **config.optimizer_params)
        self.scheduler = scheduler(self.optimizer, **config.scheduler_params)
        self.config = config
        self.logger = Logger(config.folder + f"/{fold}")
        self.fold = fold
        self.debug = DEBUG
        self.best_map = 0

    def get_loader(self):
        data = pd.read_csv(DefaultConfig.df_path)
        if self.debug:
            data = data.sample(100)

        df_folds, train = return_data_w_folds(data)
        train_dataset = get_train_dataset(fold_number=self.fold,
                                          df_folds=df_folds,
                                          train=train
                                          )
        self.train_loader = get_train_data_loader(
            train_dataset,
            batch_size=train_cfg.batch_size
        )

        validation_dataset = get_validation_dataset(fold_number=self.fold,
                                                    df_folds=df_folds,
                                                    train=train
                                                    )
        self.val_loader = get_validation_data_loader(
            validation_dataset,
            batch_size=1
        )

    def train(self, n_epoch: int):
        self.get_loader()
        fold = self.fold
        for i in range(1, n_epoch + 1):
            self.train_one_epoch(i, self.train_loader)
            mAP = self.validate_one_epoch(i, self.val_loader)
            self.scheduler.step()
            if mAP > self.best_map:
                save_checkpoint(i, self.model, self.optimizer, self.config, fold)
                self.best_map = mAP

    def train_one_epoch(self, epoch, loader):
        self.model.train()
        self.losses = []

        tk0 = tqdm(enumerate(loader), total=len(loader))
        for step, (images, targets, image_ids) in tk0:
            images = torch.stack(images).to(self.config.device).float()
            targets = [{k: v.to(self.config.device) for k, v in t.items()} for t in targets]

            outputs = self.model(images, targets)
            loss = sum(loss for loss in outputs.values())
            loss = loss / train_cfg.accum_steps
            loss.backward()
            # gradient accumulation
            if (step + 1) % train_cfg.accum_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.losses.append(loss.item() * train_cfg.accum_steps)
            tk0.set_postfix(Train_Loss=np.mean(self.losses),
                            Epoch=epoch,
                            LR=self.optimizer.param_groups[0]['lr'])

    @torch.no_grad()
    def validate_one_epoch(self, epoch, loader, nms_threshold=0.4):
        self.maps = []
        print(f"Starting validate at epoch: {epoch}")
        self.model.eval()

        tk0 = tqdm(enumerate(loader), total=len(loader))

        for step, (images, targets, image_ids) in tk0:
            images = torch.stack(images)
            images = images.to(self.config.device).float()
            gt = [target['boxes'].to(self.config.device).float() for target in targets]
              
            outputs = self.model(images)[0]
            nms_idx = torchvision.ops.nms(
                boxes=outputs["boxes"],
                scores=outputs["scores"],
                iou_threshold=nms_threshold
            )
            outputs["boxes"] = outputs["boxes"][nms_idx]
            outputs["scores"] = outputs["scores"][nms_idx]
            if outputs["boxes"].nelement() == 0:
                mAP = torch.tensor(0)
            else:
                mAP = calculate_map(
                    gt[0], outputs["boxes"], outputs["scores"],
                    thresh=DefaultConfig.iou_threshold
                )

            self.maps.append(mAP.item())

            tk0.set_postfix(Val_mAP=np.mean(self.maps),
                            Epoch=epoch,
                            LR=self.optimizer.param_groups[0]['lr'])
            
        return np.mean(self.maps)
