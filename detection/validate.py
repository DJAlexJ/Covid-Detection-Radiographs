import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision

import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from typing import Dict

from metrics import calculate_map
from models import get_faster_rcnn, FasterRCNNDetector
from dataset import DetectionDataset
from config import DefaultConfig, TrainGlobalConfig as cfg
from utils import (
    seed_everything, return_data_w_folds,
    get_validation_dataset, get_validation_data_loader
)

os.environ['CUDA_VISIBLE_DEVICES'] = '7'


@torch.no_grad()
def compute_map(
        model: FasterRCNNDetector,
        val_loader: DataLoader,
        conf_threshold: float,
        nms_threshold: float = 0.4,
        iou_threshold: float = 0.5,
):
    maps = []
    print(f"Starting validating mAP @ {iou_threshold}")
    model.eval()

    tk0 = tqdm(enumerate(val_loader), total=len(val_loader))

    for step, (images, targets, _) in tk0:
        no_boxes_flg = torch.all(targets[0]["labels"].eq(0))
        images = torch.stack(images)
        images = images.to(cfg.device).float()
        gt = [target['boxes'].to(cfg.device).float() for target in targets]

        outputs = model(images)[0]
        nms_idx = torchvision.ops.nms(
            boxes=outputs["boxes"],
            scores=outputs["scores"],
            iou_threshold=nms_threshold
        )
        outputs["boxes"] = outputs["boxes"][nms_idx]
        outputs["scores"] = outputs["scores"][nms_idx]
        outputs["scores"], outputs["boxes"] = apply_threshold_for_boxes(outputs, threshold=conf_threshold)
        if outputs["boxes"].nelement() == 0 and no_boxes_flg:
            mAP = torch.tensor(1)
        elif outputs["boxes"].nelement() == 0:
            mAP = torch.tensor(0)
        else:
            mAP = calculate_map(
                gt[0], outputs["boxes"], outputs["scores"],
                thresh=iou_threshold
            )

        maps.append(mAP.item())
        tk0.set_postfix(Val_mAP=np.mean(maps))

    return np.mean(maps)


def apply_threshold_for_boxes(prediction: Dict[str, torch.Tensor], threshold: float):
    valid_scores = (prediction["scores"] > threshold)  # finding scores > thresholds
    valid_boxes = (prediction["boxes"][valid_scores])
    valid_scores = (prediction["scores"][valid_scores])
    return valid_scores, valid_boxes


if __name__ == "__main__":
    seed_everything(DefaultConfig.seed)
    model = get_faster_rcnn(
        checkpoint_path="./checkpoints/faster_rcnn_detector_0_epoch_2.pth",
        is_eval=True
    )
    data = pd.read_csv(DefaultConfig.df_path)
    df_folds, train = return_data_w_folds(data, only_pos_class=False)
    validation_dataset = get_validation_dataset(fold_number=0,
                                                df_folds=df_folds,
                                                train=train
                                                )
    val_loader = get_validation_data_loader(
        validation_dataset,
        batch_size=1
    )

    metrics = {
        "best_conf_threshold": 0.199,
        "best_nms_threshold": 0.45,
        "best_map": 0.163,
    }
    nms_range = np.arange(0.45, 0.5, 0.05)
    thr_range = np.arange(0.199, 0.21, 0.001)
    for nms_thr in nms_range:
        for conf_thr in thr_range:
            print(f"NMS: {nms_thr}, CONF: {conf_thr}")
            mean_avg_precision = compute_map(
                model=model,
                val_loader=val_loader,
                conf_threshold=conf_thr,
                nms_threshold=nms_thr,
                iou_threshold=0.5,
            )
            if mean_avg_precision > metrics["best_map"]:
                metrics["best_map"] = mean_avg_precision
                metrics["best_conf_threshold"] = conf_thr
                metrics["best_nms_threshold"] = nms_thr

    print(f"best mAP = {metrics['best_map']} "
          f"at confidence threshold = {metrics['best_conf_threshold']} "
          f"and nms threshold = {metrics['best_nms_threshold']}"
          )
