import torch
import torch.nn as nn
import torchvision
from coral_pytorch.layers import CoralLayer


class CornifiedModel(nn.Module):
    def __init__(self, model_name, num_classes) -> None:
        super(CornifiedModel, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.initiate_model()

    def initiate_model(self):
        weights_name = self.model_name + "_Weights"
        weights = getattr(torchvision.models, weights_name)
        self.model = getattr(torchvision.models, self.model_name.lower())(weights=weights.DEFAULT)

        if "resnet" in self.model_name.lower():
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, self.num_classes - 1)
        elif "efficientnet" in self.model_name.lower():
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_ftrs, self.num_classes - 1)

    def forward(self, x):
        logits = self.model(x)
        return logits

if __name__ == "__main__":
    model = CornifiedModel("ResNet50", 10)

    x = torch.randn(1, 3, 224, 224)
    logits = model(x)

    print(logits)