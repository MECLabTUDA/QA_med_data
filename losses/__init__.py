import torch.nn as nn

from coral_pytorch.losses import corn_loss
from coral_pytorch.losses import coral_loss

from .focalloss import FocalLoss


def get_loss(name, **kwargs):
    if name.lower() == "crossentropy":
        loss = nn.CrossEntropyLoss(**kwargs)
    elif name.lower() == "focalloss":
        loss = FocalLoss(**kwargs)
    elif name.lower() == "corn":
        loss = corn_loss
    elif name.lower() == "coral":
        print("here")
        loss = coral_loss
    else:
        return None

    return loss