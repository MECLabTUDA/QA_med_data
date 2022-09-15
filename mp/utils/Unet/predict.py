import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import SimpleITK as sitk
import json
import nibabel as nb
from skimage.io import imread, imsave


from mp.utils.Unet.Basic_dataset import BasicDataset
from mp.utils.Unet.UNet import UNet
from mp.utils.Unet.utils import plot_img_and_mask
from mp.utils import load_restore as lr
from mp.data.pytorch.transformation import centre_crop_pad_3d
from mp.data.datasets.dataset_utils import delete_images_and_labels
from mp.utils.Unet.dice_score import dice_coeff

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
       
    return output.argmax(dim=1)[0].float()
        





def get_output_filenames(output, input):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return output or list(map(_generate_name, input))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))




#if __name__ == '__main__':
def prediction(source_path, target_path, device, img_size=(1,35,51,35), bilinear=False, scale=1, mask_threshold=0.5, no_save=False, viz=True):
    
    delete_images_and_labels(target_path)
    filenames = [x for x in os.listdir(source_path)]

    logging.info(f'Using device {device}')

    path_m = os.path.join('UNET_STATE_DICT_1.zip')
    model = lr.load_model('UNet', path_m, True)
    model.to(device)
    logging.info('Model loaded!')
    
    dice_overall = []
    for i, filename in enumerate(filenames):
        logging.info(f'\nPredicting image {filename} ...')
        
        #Read the 3D image
        img = sitk.ReadImage(os.path.join(source_path, filename,'img', 'img.nii.gz'))  
        #Read the segmentation of the image
        seg = sitk.ReadImage(os.path.join(source_path, filename, 'seg', '001.nii.gz')) 

        #Center crop image and segmentation
        img_array = sitk.GetArrayFromImage(img)
        img_array_crop = centre_crop_pad_3d(torch.from_numpy(img_array).unsqueeze_(0), img_size)[0]
        seg_array = sitk.GetArrayFromImage(seg)
        seg_array_crop = centre_crop_pad_3d(torch.from_numpy(seg_array).unsqueeze_(0), img_size)[0]
        seg_array_crop = seg_array_crop.to(device=device, dtype=torch.float32)

        masks = []
        #Divide the 3D image in 2D slices, so each slice can be used for the training of the UNet
        for i in range(img_array_crop.shape[2]): 
            slice_img = img_array_crop[:,:,i]
            slice_img_array = np.array(slice_img)
            #Change the format of the slice to PIL image
            slice_img = Image.fromarray(slice_img_array) 

            slice_seg = seg_array_crop[:,:,i]

            #predict the segmentation
            mask = predict_img(net=model, full_img=slice_img, scale_factor=scale, out_threshold=mask_threshold, device=device)
            masks.append(np.array(mask.cpu()))

            #calculate the dice coefficient
            dice = dice_coeff(slice_seg, torch.tensor(mask))
            dice_overall.append(dice)
        
        #the hilfs_img is used for the format
        hilfs_img = nb.load(os.path.join(source_path, filename,'img', 'img.nii.gz'))
        #from the 2d predictions is a 3D prediction created
        masks = torch.tensor(masks).permute((1,2,0))
        masks = np.array(masks)
        masks = nb.Nifti1Image(masks, affine=hilfs_img.affine)

        #save the image, the true segmentation and the predicted segmentation
        os.makedirs(os.path.join(target_path, filename))
        sitk.WriteImage(img, os.path.join(target_path, filename, 'img'+'.nii.gz'))
        nb.save(masks, os.path.join(target_path, filename, 'pred_mask'+'.nii.gz'))
        sitk.WriteImage(seg, os.path.join(target_path, filename, 'seg'+'.nii.gz'))
    
    dice_mean = np.mean(np.array(torch.tensor(dice_overall).cpu()))
    print('Dice_mean for test is: '+str(dice_mean))




        