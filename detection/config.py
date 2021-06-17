import torch
import torch.optim


class DefaultConfig:
    train_dir: str = "../data512/train/"
    test_dir: str = "../data512/test/"
    df_path: str = "../data/updated_train_labels.csv"
    n_folds: int = 5
    # fold_num: int = 0  # номер фолда для тренировки
    seed: int = 2021
    num_classes: int = 2  # NEW: negative, not negative, DEPRECATED: "negative", "typical", "indeterminate", "atypical"
    img_size: int = 512
    iou_thresholds: list = [0.5]  # порог для подсчета MaP @ IoU


class TrainGlobalConfig:
    model_name: str = "faster_rcnn"  # faster_rcnn or eff_det. eff_det validation is not implemented yet!!!
    num_workers: int = 4
    batch_size: int = 10
    n_epochs: int = 15
    lr: float = 0.0003
    device: str = "cuda"

    img_size = DefaultConfig.img_size

    folder = "./logs_torch"  # директория для весов и логов
    base_name: str = "faster_rcnn_detector_{fold}_epoch_{epoch}.pth"
    checkpoint_dir: str = "./checkpoints/{filename}"

    # -------------------
    verbose = True
    verbose_step = 1
    # -------------------

    optimizer = torch.optim.Adam
    optimizer_params = dict(lr=0.001, betas=(0.9, 0.999), eps=1e-08)

    # --------------------
    # step_scheduler = False  # делать scheduler.step после optimizer.step
    # validation_scheduler = False  # делать scheduler.step после validation stage loss

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
    scheduler_params = dict(T_0=4, T_mult=1, eta_min=1e-6, last_epoch=-1)
    # --------------------
