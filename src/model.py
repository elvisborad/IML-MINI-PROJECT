import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


def build_model(pretrained=True, freeze_backbone=True):
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model
