import torch
import torch.optim

class DefaultConfig:
    n_folds: int = 5
    seed: int = 2021
    num_classes: int = 4  # "negative", "typical", "indeterminate", "atypical"
    img_size: int = 512
    fold_num: int = 0
    device: str = 'cuda'


class TrainGlobalConfig:
    num_workers: int = 8
    batch_size: int = 12
    n_epochs: int = 40 #40
    lr: float = 0.0002

    img_size = DefaultConfig.img_size
        
    folder = './logs_torch' #folder_name 

    # -------------------
    verbose = True
    verbose_step = 1
    # -------------------

    # --------------------
    step_scheduler = False  # do scheduler.step after optimizer.step
    validation_scheduler = False  # do scheduler.step after validation stage loss
    
    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='min',
        factor=0.5,
        patience=1,
        verbose=False, 
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0, 
        min_lr=1e-8,
        eps=1e-08
    )
    # --------------------