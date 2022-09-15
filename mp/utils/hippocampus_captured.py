import os
import logging
import SimpleITK as sitk
import torch
import numpy as np
from PIL import Image

from mp.data.pytorch.transformation import centre_crop_pad_3d
from mp.utils.Unet.predict import predict_img
from mp.utils import load_restore as lr



def hippocampus_fully_captured(source_path, device, img_size=(1,35,51,35)):
    r"""Method which checks wether the hippocampus is fully captured. The first and the last slices have to be completed back. The slices in the middle must have black edges"""

    #Load the Unet
    path_m = os.path.join('UNET_STATE_DICT_1.zip')
    model = lr.load_model('UNet', path_m, True)
    model.to(device)
    logging.info('Model for hippocampus segmentation loaded!')

    #Read the image which have to be checked
    img = sitk.ReadImage(source_path)  
    img_array = sitk.GetArrayFromImage(img)
    img_array_crop = centre_crop_pad_3d(torch.from_numpy(img_array).unsqueeze_(0), img_size)[0]
    
    #Transform 3D image to 2D slices
    for i in range(img_array_crop.shape[2]): 
        slice_img = img_array_crop[:,:,i]
        slice_img_array = np.array(slice_img)
        #Bring 2D image to PIL image format
        slice_img = Image.fromarray(slice_img_array) 

        mask = predict_img(net=model, full_img=slice_img, scale_factor=1, out_threshold=0.5, device=device)
        mask = mask.cpu()
        last_slice_index = img_array_crop.shape[2] -1

        #First and last slice have to be black
        if i == 0 or i == last_slice_index:
            if np.array(mask).any():
                return False
        #All slices in the middle must have a black edge
        else:
            if np.array(mask[0]).any():#if any on the edge is white, false is returned
                return False
            if np.array(mask[-1]).any():
                return False
            if np.array(mask[:,0]).any():
                return False
            if np.array(mask[:,-1]).any():
                return False
        
    #True when the hippocampus is fully captured
    return True
