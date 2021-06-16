import gc

import torch
import torchvision
from config import DefaultConfig, TrainGlobalConfig
from effdet import (DetBenchTrain, EfficientDet, create_model,
                    create_model_from_config, get_efficientdet_config)
from effdet.efficientdet import HeadNet
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import (FasterRCNN,
                                                      FastRCNNPredictor)


class Logger(object):
    def __init__(self, logdir):
        from collections import defaultdict

        from torch.utils.tensorboard import SummaryWriter

        self.step_map = defaultdict(int)
        self.writer = SummaryWriter(logdir)

    def log(self, key, value):
        self.writer.add_scalar(key, value, self.step_map[key])
        self.step_map[key] += 1


class FasterRCNNDetector(torch.nn.Module):
    def __init__(self, pretrained=False):
        super(FasterRCNNDetector, self).__init__()
        # load pre-trained model incl. head
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=pretrained, pretrained_backbone=pretrained
        )

        # get number of input features for the classifier custom head
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, DefaultConfig.num_classes
        )

    def forward(self, images, targets=None):
        return self.model(images, targets)


def get_faster_rcnn(checkpoint_path=None, pretrained=False):
    model = FasterRCNNDetector(pretrained=pretrained)

    # Load the trained weights
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])

        del checkpoint
        gc.collect()

    model = model.to(TrainGlobalConfig.device)
    return model


# def get_efficient_det(checkpoint_path=None, pretrained=False):
#     config = get_efficientdet_config('tf_efficientdet_d2')
#
#     config.image_size = [DefaultConfig.img_size, DefaultConfig.img_size]
#     #     config.norm_kwargs=dict(eps=.001, momentum=.01)
#
#     model = EfficientDet(config, pretrained_backbone=pretrained)
#     model = model.to(DefaultConfig.device)
#     #     checkpoint = torch.load('../input/efficientdet/efficientdet_d5-ef44aea8.pth')
#     #     net.load_state_dict(checkpoint)
#
#     model.reset_head(num_classes=4)
#     model.class_net = HeadNet(config, num_outputs=config.num_classes)
#
#     #     base_config = get_efficientdet_config('tf_efficientdet_d5')
#     #     base_config.image_size = (512, 512)
#
#     #     model = create_model_from_config(
#     #         base_config,
#     #         bench_task='predict',
#     #         num_classes=4,
#     #         pretrained=pretrained,
#     #         checkpoint_path=checkpoint_path
#     #     )
#     return DetBenchTrain(model, config)
