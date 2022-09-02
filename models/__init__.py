import torch.nn as nn
from .coralmodels import CoralizedModel
from .cornmodels import CornifiedModel
import torchvision

def get_model(model_name, classes, use_corn=False, use_coral=False):
    weights_name = model_name + "_Weights"
    weights = getattr(torchvision.models, weights_name)
    model_ft = getattr(torchvision.models, model_name.lower())(weights=weights.DEFAULT)
    if use_coral:
        model = CoralizedModel(model_name, classes)
        return model
    elif use_corn:
        model = CornifiedModel(model_name, classes)
        return model
    # model_ft = resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)

    if "resnet" in model_name.lower():
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, classes)
    if "efficientnet" in model_name.lower():
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs, classes)

    return model_ft
