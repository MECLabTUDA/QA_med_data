
import logging
import os
from os import listdir
from os.path import splitext
from pathlib import Path
from tkinter import image_types

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import SimpleITK as sitk
from time import sleep


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.ids = []
        for file in listdir(images_dir):
            if not file.startswith('.'):
                name = splitext(file)[0]
                self.ids.append(splitext(name)[0])
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        img_ndarray = np.asarray(pil_img)

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

        return img_ndarray

    @staticmethod
    def load(filename, image_type):
        img = sitk.ReadImage(os.path.join(filename)) 
        img = sitk.GetArrayFromImage(img)
        img = Image.fromarray(img)
        
        return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = self.load(mask_file[0], image_type = 'masks')
        img = self.load(img_file[0], image_type = 'imgs')

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }

