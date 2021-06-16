import timm
import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int = 4, pretrained=True):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.model = timm.create_model(model_name, pretrained=pretrained)
        if model_name.startswith("swin"):
            in_features = self.model.head.in_features
            self.model.head = nn.Linear(in_features, num_classes)
        elif model_name.startswith("resnet"):
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
        elif "efficientnet" in model_name:
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_features, num_classes)
        self.model.cuda()

    def forward(self, x):
        return self.model(x)
