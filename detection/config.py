import torch
import torch.optim


class DefaultConfig:
    train_dir: str = "../data1024/train/"
    test_dir: str = "../data1024/test/"
    df_path: str = "../data/updated_train_labels.csv"
    n_folds: int = 5
    seed: int = 2021
    num_classes: int = 2  # NEW: negative, not negative, DEPRECATED: "negative", "typical", "indeterminate", "atypical"
    img_size: int = 1024
    iou_threshold: float = 0.5  # threshold for MaP @ IoU


class TrainGlobalConfig:
    model_name: str = "faster_rcnn"  # faster_rcnn or eff_det. eff_det validation is not implemented yet!!!
    num_workers: int = 4
    batch_size: int = 10
    n_epochs: int = 30
    device: str = 'cuda'
    accum_steps = 3  # steps for gradient accumulation


    img_size = DefaultConfig.img_size

    folder = "./logs_torch"  # директория для весов и логов
    base_name: str = 'faster_rcnn_detector_{fold}_epoch_{epoch}_1024.pth'
    checkpoint_dir: str = './checkpoints/{filename}'

    optimizer = torch.optim.Adam
    optimizer_params = dict(
        lr=0.0009
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
    scheduler_params = dict(
        T_0=5,
        T_mult=1,
        eta_min=1e-6,
        last_epoch=-1
    )
