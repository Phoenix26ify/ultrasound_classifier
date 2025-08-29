# This is the Model definition (EfficientNet etc.)
# Created by ShreyaM
import torch.nn as nn
from torchvision import models

def build_model(num_classes=2):
    model = models.efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(1280, num_classes)
    return model
