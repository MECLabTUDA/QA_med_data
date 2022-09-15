import os
import numpy as np
import torch
import SimpleITK as sitk
import mp.utils.load_restore as lr
from mp.quantifiers.QualityQuantifier import ImgQualityQuantifier
from mp.utils.hippocampus_captured import hippocampus_fully_captured
from mp.utils.create_patches import patchify as Patches

class NoiseQualityQuantifier(ImgQualityQuantifier):
    def __init__(self, artefact_given, output_features, device='cuda:0', version='0.0'):
        
        self.device=device
        self.artefact = artefact_given
        self.quality_values = [0, 0.25, 0.5, 0.75, 1]
        
        # Get model path
        path_m = os.path.join(os.environ["OPERATOR_PERSISTENT_DIR"], self.artefact, 'model_state_dict.zip')

        # Load the correct model depending on the artefact
        if self.artefact == 'spike':
            print('Densenet121 wird geladen')
            model = lr.load_model('Densenet121', path_m, True) 
        else: 
            print('Mobilenet wird geladen')
            model = lr.load_model('MobileNetV2', path_m, True)
        
        # Upload Model
        model.to(device)
        self.model = model

        super().__init__(device, version)

    def get_quality(self, x, file, device):
        r"""Get quality values for an image representing the maximum intensity of artefacts in it.

        Args:
            x (data.Instance): an instance for a 3D image, normalized so
                that all values are between 0 and 1. An instance (image) in the dataset
                follows the dimensions (channels, width, height, depth), where channels == 1
            path (string): the full path to the file representing x (only used for checking if the 
                           lung is fully captured)

        Returns:
            fully_captured (bool): a boolean if the hippocampus is fully captured. 
            yhat (float): a float that indicates the quality of the image concerning the given artefact.
        """

        # Check if the hippocampus is fully captured
        fully_captured = hippocampus_fully_captured(source_path = file, device=device)
        
        
        # Predict the label for the given Image
        model = self.model
        x = x.to(self.device)
        yhat = model(x)
        yhat = yhat.cpu().detach()
        
        # Transform one hot vector to likert value
        _, yhat = torch.max(yhat, 1)
        yhat = self.quality_values[yhat.item()]

        # Return if hippocampus is fully captured and the quality
        return fully_captured, yhat