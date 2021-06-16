import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import pandas as pd
import numpy as np

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from models import FasterRCNNDetector, Logger, get_faster_rcnn
from metrics import EvalMeter
from config import DefaultConfig, TrainGlobalConfig
from utils import (
    get_train_file_path,
    get_train_dataset,
    get_train_data_loader,
    get_validation_dataset,
    get_validation_data_loader,
    save_checkpoint,
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
        DEBUG: bool = True,
    ):

        self.model = get_faster_rcnn(pretrained=pretrained)
        self.model.cuda()

        self.optimizer = optimizer(self.model.parameters(), **config.optimizer_params)
        self.scheduler = scheduler(self.optimizer, **config.scheduler_params)
        self.config = config
        self.logger = Logger(config.folder + f"/{fold}")
        self.fold = fold
        self.debug = DEBUG

    def get_loader(self):
        data = pd.read_csv(DefaultConfig.df_path)
        if self.debug:
            data = data.sample(100)
        data["jpg_path"] = data["id"].apply(get_train_file_path)
        data.loc[
            data.human_label.isin(["atypical", "indeterminate", "typical"]),
            "integer_label",
        ] = 1
        data.loc[data.human_label == "negative", "integer_label"] = 0
        data = data[data.human_label != "negative"]
        train = data.copy()

        df_folds = train.copy()
        skf = StratifiedKFold(
            n_splits=DefaultConfig.n_folds,
            shuffle=True,
            random_state=DefaultConfig.seed,
        )

        # Готовим фолды
        for n, (train_index, val_index) in enumerate(
            skf.split(X=df_folds.index, y=df_folds.integer_label)
        ):
            df_folds.loc[df_folds.iloc[val_index].index, "fold"] = n

        df_folds["fold"] = df_folds["fold"].astype(int)
        df_folds.set_index("id", inplace=True)

        train_dataset = get_train_dataset(
            fold_number=self.fold, df_folds=df_folds, train=train
        )
        self.train_loader = get_train_data_loader(
            train_dataset, batch_size=TrainGlobalConfig.batch_size
        )

        validation_dataset = get_validation_dataset(
            fold_number=self.fold, df_folds=df_folds, train=train
        )
        self.val_loader = get_validation_data_loader(
            validation_dataset, batch_size=TrainGlobalConfig.batch_size
        )

    def train(self, n_epoch: int):
        self.get_loader()
        fold = self.fold
        for i in range(1, n_epoch + 1):
            self.train_one_epoch(i, self.train_loader)
            save_checkpoint(i, self.model, self.optimizer, self.config, fold)
            self.validate_one_epoch(i, self.val_loader)
            self.scheduler.step()

    def train_one_epoch(self, epoch, loader):
        self.model.train()
        self.losses = []

        tk0 = tqdm(enumerate(loader), total=len(loader))
        for step, (images, targets, image_ids) in tk0:
            images = torch.stack(images).to(self.config.device).float()
            # batch_size = images.shape[0]
            # boxes = [target['boxes'].to(self.config.device).float() for target in targets]
            # labels = [target['labels'].to(self.config.device).float() for target in targets]
            targets = [
                {k: v.to(self.config.device) for k, v in t.items()} for t in targets
            ]

            self.optimizer.zero_grad()

            # if TrainGlobalConfig.model_name == 'eff_det':
            #     target_eff = dict()
            #     target_eff['bbox'] = boxes
            #     target_eff['cls'] = labels
            #     target_eff["img_scale"] = None  # torch.tensor([1.0] * batch_size, dtype=torch.float).to(self.device)
            #     target_eff[
            #         "img_size"] = None  # torch.tensor([images[0].shape[-2:]] * batch_size, dtype=torch.float).to(self.device)
            #
            #     output = self.model(images, target_eff)
            #     loss = output['loss']

            outputs = self.model(images, targets)
            loss = sum(loss for loss in outputs.values())
            loss.backward()
            self.optimizer.step()

            self.losses.append(loss.item())
            tk0.set_postfix(
                Train_Loss=np.mean(self.losses),
                Epoch=epoch,
                LR=self.optimizer.param_groups[0]["lr"],
            )
            # if self.config.step_scheduler:
            #     self.scheduler.step()

    @torch.no_grad()
    def validate_one_epoch(self, epoch, loader):
        self.losses = []
        print(f"Starting validate at epoch: {epoch}")
        self.model.train()  # оставляем train mode, чтобы считать лоссы

        tk0 = tqdm(enumerate(loader), total=len(loader))
        eval_scores = EvalMeter()

        for step, (images, targets, image_ids) in tk0:
            images = torch.stack(images)
            # batch_size = images.shape[0]
            images = images.to(self.config.device).float()
            targets = [
                {k: v.to(self.config.device) for k, v in t.items()} for t in targets
            ]
            # boxes = [target['boxes'].to(self.device).float() for target in targets]
            # labels = [target['labels'].to(self.device).float() for target in targets]

            # elif TrainGlobalConfig.model_name == 'eff_det':
            #
            #     target_eff = {}
            #     target_eff['bbox'] = boxes
            #     target_eff['cls'] = labels
            #     target_eff["img_scale"] = None # torch.tensor([1.0] * batch_size, dtype=torch.float).to(self.device)
            #     target_eff["img_size"] = None # torch.tensor([images[0].shape[-2:]] * batch_size, dtype=torch.float).to(self.device)
            #
            #     output = self.model(images, target_eff)
            #     loss = output['loss']

            outputs = self.model(images, targets)
            loss = sum(loss for loss in outputs.values())

            self.losses.append(loss.item())
            # Считаем mAP @ 0.5, но кажется, там что-то работает некорректно
            # for i, image in enumerate(images):
            #     gt_boxes = targets[i]['boxes'].data.cpu().numpy()
            #     boxes = outputs[i]['boxes'].data.cpu().numpy()
            #     scores = outputs[i]['scores'].detach().cpu().numpy()
            #
            #     preds_sorted_idx = np.argsort(scores)[::-1]
            #     preds_sorted_boxes = boxes[preds_sorted_idx]
            #
            #     eval_scores.update(pred_boxes=preds_sorted_boxes, gt_boxes=gt_boxes)

            tk0.set_postfix(
                Val_loss=np.mean(self.losses),
                Epoch=epoch,
                LR=self.optimizer.param_groups[0]["lr"],
            )
