import sys
sys.path.insert(0, "./mmdetection")

import os
# Check Pytorch installation
import torch
import torchvision
print(torch.__version__, torch.cuda.is_available())

# Check mmcv installation
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print(get_compiling_cuda_version())
print(get_compiler_version())

# Check MMDetection installation
from mmdet.apis import set_random_seed

# Imports
import mmdet
from mmdet.apis import set_random_seed
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector

import random
import numpy as np
from pathlib import Path

global_seed = 63


def set_seed(seed=global_seed):
    """Sets the random seeds."""
    set_random_seed(seed, deterministic=False)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed()

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from mmcv import Config


baseline_cfg_path = "../mmdetection/configs/cascade_rcnn/cascade_rcnn_r50_fpn_20e_coco.py"
# baseline_cfg_path = "../mmdetection/configs/resnest/cascade_rcnn_s50_fpn_syncbn-backbone+head_mstrain-range_1x_coco.py"
cfg = Config.fromfile(baseline_cfg_path)

# model_name = 'vfnet_r50_fpn'
model_name = 'cascade_rcnn_r50_fpn'
fold = 1

# Folder to store model logs and weight files
job_folder = f'./logs/{model_name}_fold{fold}'
cfg.work_dir = job_folder

# Set seed thus the results are more reproducible
cfg.seed = global_seed

if not os.path.exists(job_folder):
    os.makedirs(job_folder)

print("Job folder:", job_folder)

# Set the number of classes
for head in cfg.model.roi_head.bbox_head:
    head.num_classes = 1
# cfg.model.roi_head.bbox_head.num_classes = 1
# cfg.model.bbox_head.num_classes = 1

# cfg.gpu_ids = range(1)
cfg.gpu_ids = [0]

# Setting pretrained model in the init_cfg which is required
# for transfer learning as per the latest MMdetection update
cfg.model.backbone.init_cfg = dict(type='Pretrained', checkpoint='torchvision://resnet50')
cfg.model.pop('pretrained', None)

cfg.runner.max_epochs = 10  # Epochs for the runner that runs the workflow
# One Epoch takes around 18 mins
cfg.total_epochs = 10

# Learning rate of optimizers. The LR is divided by 8 since the config file is originally for 8 GPUs
cfg.optimizer.lr = 0.001

# Learning rate scheduler config used to register LrUpdater hook
cfg.lr_config = dict(
    policy='CosineAnnealing',  # The policy of scheduler, also support CosineAnnealing, Cyclic, etc. Refer to details of supported LrUpdater from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9.
    by_epoch=False,
    warmup='linear',  # The warmup policy, also support `exp` and `constant`.
    warmup_iters=500,  # The number of iterations for warmup
    warmup_ratio=0.001,  # The ratio of the starting learning rate used for warmup
    min_lr=1e-07)

# config to register logger hook
cfg.log_config.interval = 50  # Interval to print the log

# Config to set the checkpoint hook, Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation.
cfg.checkpoint_config.interval = 1  # The save interval is 1

cfg.dataset_type = 'CocoDataset'  # Dataset type, this will be used to define the dataset
cfg.classes = ("Covid_Abnormality",)

cfg.data.train.img_prefix = './data512/train'  # Prefix of image path
cfg.data.train.classes = cfg.classes
cfg.data.train.ann_file = f'./data512/train_annotations_fold{fold}.json'
cfg.data.train.type = 'CocoDataset'

cfg.data.val.img_prefix = './data512/train'  # Prefix of image path
cfg.data.val.classes = cfg.classes
cfg.data.val.ann_file = f'./data512/val_annotations_fold{fold}.json'
cfg.data.val.type = 'CocoDataset'

cfg.data.test.img_prefix = './data512/train'  # Prefix of image path
cfg.data.test.classes = cfg.classes
cfg.data.test.ann_file = f'./data512/val_annotations_fold{fold}.json'
cfg.data.test.type = 'CocoDataset'

cfg.data.samples_per_gpu = 8  # Batch size of a single GPU used in testing
cfg.data.workers_per_gpu = 8  # Worker to pre-fetch data for each single GPU

# The config to build the evaluation hook, refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/evaluation/eval_hooks.py#L7 for more details.
cfg.evaluation.metric = 'bbox'  # Metrics used during evaluation

# Set the epoch interval to perform evaluation
cfg.evaluation.interval = 1

# Set the iou threshold of the mAP calculation during evaluation
cfg.evaluation.iou_thrs = [0.5]

cfg.evaluation.save_best = 'bbox_mAP_50'

albu_train_transforms = [
    dict(type='ShiftScaleRotate', shift_limit=0.0625,
         scale_limit=0.1, rotate_limit=15, p=0.2),
    dict(type='RandomBrightnessContrast', brightness_limit=0.2,
         contrast_limit=0.2, p=0.3),
    dict(type='IAAAffine', shear=(-10.0, 10.0), p=0.4),
    dict(
        type="OneOf",
        transforms=[
            dict(type="Blur", p=1.0, blur_limit=7),
            dict(type="GaussianBlur", p=1.0, blur_limit=7),
            dict(type="MedianBlur", p=1.0, blur_limit=7),
        ],
        p=0.2,
    ),

    #     dict(type='MixUp', p=0.2, lambd=0.5),
    #     dict(type='RandomRotate90', p=0.5),
    dict(type='CLAHE', p=0.5),
    #     dict(type='InvertImg', p=0.5),
    #     dict(type='Equalize', mode='cv', p=0.4),
    #     dict(type='MedianBlur', blur_limit=3, p=0.1)
]

cfg.train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap=dict(img='image', gt_bboxes='bboxes'),
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]
cfg.test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

model = build_detector(cfg.model,
                       train_cfg=cfg.get('train_cfg'),
                       test_cfg=cfg.get('test_cfg'))
model.init_weights()

datasets = [build_dataset(cfg.data.train)]

train_detector(model, datasets[0], cfg, distributed=False, validate=True)
