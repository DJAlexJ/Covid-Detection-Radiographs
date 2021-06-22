from typing import Any, Callable, Dict, Union

import torch
import torch.optim as optim


class DefaultConfig:
    train_dir: str = "./data512/train/"
    test_dir: str = "./data512/test/"
    df_path: str = "./data/updated_train_labels.csv"
    n_folds: int = 5
    seed: int = 2021
    num_classes: int = 4  # "negative", "typical", "indeterminate", "atypical"
    img_size: int = 512
    iou_thresholds: list = [0.5]  #
    fold_num: int = 0  # номер фолда для тренировки
    device: str = "cuda"


class TrainConfig:
    train_dir: str = "/home/almokhov/covid/train_data"
    test_dir: str = ""
    csv_file: str = "/home/almokhov/covid/classification_data.csv"
    label: str = "classification_label" # label
    n_folds: int = 5
    n_epochs: int = 160
    num_classes: int = 2
    img_size: int = 512
    device: str = "cuda"
    checkpoint_dir: str = "/home/almokhov/covid/Covid-Detection-Radiographs/classification/checkpoints_classification/{filename}"
    base_name: str = "effnet_b7_classifier_nfold_{fold}_epoch_{epoch}.pth"  # 'swin_classifier_nfold_{fold}_epoch_{epoch}.pth'
    optimizer: optim = optim.SGD
    optimizer_params: Dict[str, Any] = {"lr": 0.0003, "momentum": 0.9}
    loader_params: Dict[str, Union[int, float, Callable]] = {
        "batch_size": 2,
        "num_workers": 0,
        "shuffle": True,
    }
    logdir: str = "effnetb7_classification/"
    grad_accum: int = 12
