import os

import pandas as pd

import torchvision.transforms as T
import torchio as tio
import cv2
import numpy as np
import matplotlib.pyplot as plt

def convertImg(img):
    img_data = np.array(img.data)

    for b in range(img.shape[0]):
        img_loc = img_data[b]
        img_loc = (img_loc - np.min(img_loc)) / (np.max(img_loc) - np.min(img_loc))


        for slice in range(img.shape[3]):
            blur = cv2.GaussianBlur(img_loc[:,:,slice], (5, 5), 0)

            img_loc[:,:,slice] = cv2.Laplacian(blur, cv2.CV_64F)

        img_data[b] = (img_loc - np.min(img_loc)) / (np.max(img_loc) - np.min(img_loc))
        #imgplot = plt.imshow(img_data[b,:,:,0])
        #plt.show()

    img.data = img_data

    return img

class CMRxMOTION2DEval(object):
    def __init__(self, dataset_path, transforms=None, tio_transforms=None):
        self.scans = os.listdir(dataset_path)
        self.dataset_path = dataset_path

        self.transforms = transforms
        self.tio_transforms = tio_transforms
        
        self.compose_dataset()

    def compose_dataset(self):        
        samples_paths = []
        samples = []

        for scan in self.scans:
            path = os.path.join(self.dataset_path, scan)
            data = os.listdir(path)
            samples_paths.extend([os.path.join(path, d) for d in data])

        for img_fullpath in samples_paths:
            img_name = os.path.basename(img_fullpath).split(".")[0]
            img = tio.ScalarImage(img_fullpath)

            img = convertImg(img)


            pid, phase = img_name[:-3], img_name[-2:]
            subject = tio.Subject(img=img, pid=pid, phase=phase, img_name=img_name)

            samples.append(subject)

        transform = tio.Compose(self.tio_transforms)
        samples = tio.SubjectsDataset(samples, transform=transform)

        self.samples = []

        for sample in samples:

            img, pid, phase, img_name = sample['img'], sample['pid'], \
                                        sample['phase'], sample['img_name']

            for d in range(img.shape[-1]):
                img_slice = img.data[:,:,:,d]
                sliced_img = T.ToPILImage()(img_slice).convert('RGB')

                slice = {"img_slice": sliced_img, "pid": pid, 
                         "phase": phase, "img_name": img_name, "d": d,
                         "n_slices": img.shape[-1]}
                self.samples.append(slice)

        self.n_samples = len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img, pid, phase, d = sample["img_slice"], sample["pid"],\
                             sample["phase"], sample["d"]
        n_slices = sample["n_slices"]

        if self.transforms:
            img = self.transforms(img)

        return img, pid, phase, d, n_slices

    def __len__(self):
        return self.n_samples


class CMRxMOTION2D(object):
    def __init__(self, dataset_path, split=True, mode='train', pids=None, train_val_ratio=0.8, transforms=None, tio_transforms=None):
        self.sample_path = os.path.join(dataset_path, "data")
        self.label_path = os.path.join(dataset_path, "IQA.csv")
        self.split = split
        self.mode = mode
        self.pids = pids
        self.train_val_ratio = train_val_ratio

        self.csv = pd.read_csv(self.label_path)
        self.n_samples = len(self.csv)
        self.n_3d = len(self.csv)

        self.transforms = transforms
        self.tio_transforms = tio_transforms

        self.compose_dataset()

    def compose_dataset(self):
        # content = self.csv["Image"]
        
        samples = []

        if self.split:
            if self.pids:
                # Generate the corresponding filenames matching the CSV format
                # self.pids = [f"P{str(id).zfill(3)}-{s}-{edes}" for id in self.pids for s in range(1, 5) for edes in ["ED", "ES"]]
                self.pids = [f"P{str(id).zfill(3)}" for id in self.pids]
                items = []
                for d, name in enumerate(self.csv["Image"]):
                    pid, mode, _ = name.split('-')
                    if pid in self.pids:
                        if self.mode == "val" and "aug" in mode:
                            pass
                        else:
                            items.append(d)
                    else:
                        print(name)
            else:
                if self.mode == 'train':
                    min_limit = 0
                    max_limit = int(self.n_samples * self.train_val_ratio)
                elif self.mode == 'val':
                    min_limit = int(self.n_samples * self.train_val_ratio) + 1
                    max_limit = self.n_samples
                items = list(range(min_limit, max_limit))
                self.n_samples = max_limit - min_limit
        else:
            min_limit = 0
            max_limit = self.n_samples
            items = list(range(min_limit, max_limit))
            self.n_samples = max_limit - min_limit

        i = 0
        for d in items:
            # if i == 8:
            #     break
            img_name, label = self.csv["Image"][d], self.csv["Label"][d] - 1
            pid, phase = img_name[:-3], img_name[-2:]

            img_fullpath = os.path.join(self.sample_path, pid, img_name + ".nii.gz")
            img = tio.ScalarImage(img_fullpath)

            img = convertImg(img)

            segm = None
            if label != 3:
                segm_fullpath = os.path.join(self.sample_path, pid, img_name + ".nii.gz")
                segm = tio.ScalarImage(segm_fullpath)

            subject = tio.Subject(img=img, segm=segm, label=label, pid=pid, phase=phase, img_name=img_name)

            samples.append(subject)

            i += 1

        transform = tio.Compose(self.tio_transforms)
        samples = tio.SubjectsDataset(samples, transform=transform)

        self.samples = []

        for sample in samples:

            img, segm, label = sample['img'], sample['segm'], sample['label']
            pid, phase, img_name = sample['pid'], sample['phase'], sample['img_name']

            for d in range(img.shape[-1]):
                img_slice = img.data[:,:,:,d]
                sliced_img = T.ToPILImage()(img_slice).convert('RGB')
                # sliced_img = tio.Image(tensor=img_slice)
                segm_slice = None
                if label != 3:
                    segm_slice = segm.data[:,:,:,d]
                    sliced_segm = T.ToPILImage()(segm_slice)

                slice = {"img_slice": sliced_img, "segm_slice": sliced_segm, 
                         "label": label, "pid": pid, "phase": phase,
                         "img_name": img_name, "d": d, "n_slices": img.shape[-1]}
                # slice = tio.Subject(img_slice=sliced_img, segm_slice=sliced_segm, label=label)
                self.samples.append(slice)

        self.n_samples = len(self.samples)


    def __getitem__(self, idx):
        sample = self.samples[idx]
        img, label = sample["img_slice"], sample["label"]
        
        pid, phase, d = sample["pid"], sample["phase"], sample["d"]
        n_slices = sample["n_slices"]

        if self.transforms:
            img = self.transforms(img)

        return img, label, pid, phase, d, n_slices

    def __len__(self):
        return self.n_samples

