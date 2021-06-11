import torch
import torch.optim

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
    device: str = 'cuda'


class TrainGlobalConfig:
    model_name: str = "faster_rcnn" #faster_rcnn or eff_det. eff_det validation is not implemented yet!!!
    num_workers: int = 4
    batch_size: int = 10
    n_epochs: int = 40
    lr: float = 0.0003

    img_size = DefaultConfig.img_size
        
    folder = "./logs_torch"  # директория для весов и логов

    # -------------------
    verbose = True
    verbose_step = 1
    # -------------------
    
    OptimizerClass = torch.optim.Adam
    optimizer_params = dict(
        lr=0.001, 
        betas=(0.9, 0.999), 
        eps=1e-08
    )

    # --------------------
    step_scheduler = False  # делать scheduler.step после optimizer.step
    validation_scheduler = False  # делать scheduler.step после validation stage loss
    
    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='min',
        factor=0.5,
        patience=1,
        verbose=False, 
        threshold=0.0001,
        threshold_mode="abs",
        cooldown=0, 
        min_lr=1e-8,
        eps=1e-08
    )
    # --------------------