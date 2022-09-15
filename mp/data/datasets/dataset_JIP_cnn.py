# ------------------------------------------------------------------------------
# Dataset provided by JIP Tool.
# ------------------------------------------------------------------------------

# Necessary imports
import os
import torch
import random
import traceback
import mp.utils.load_restore as lr
from mp.data.datasets.dataset_cnn import CNNDataset, CNNInstance
from mp.utils.generate_labels import generate_train_labels, generate_test_labels
from mp.data.datasets.dataset_utils import delete_images_and_labels
from data_aug import augment_data_aug_blur
from data_aug import augment_data_aug_noise
from data_aug import augment_data_aug_motion
from data_aug import augment_data_aug_ghosting
from data_aug import augment_data_aug_spike
from data_aug import augmentation_segmentation, augmentation_inference

class JIPDataset(CNNDataset):
    r"""Class for the dataset provided by the JIP tool/workflow.
    """
    def __init__(self, subset=None, img_size=(1, 35, 51, 35), num_intensities=5, data_type='all', augmentation=True, data_augmented=False,
                 gpu=True, cuda=0, msg_bot=False, nr_images=20, build_dataset=False, dtype='train', noise='blur', ds_name='Task', seed=42,
                 restore=False, inference_name = None, fft_for_inference = False):
        r"""Constructor"""
        assert subset is None, "No subsets for this dataset."
        assert len(img_size) == 4, "Image size needs to be 4D --> (batch_size, depth, height, width)."
        self.img_size = img_size
        self.num_intensities = num_intensities
        self.augmentation = augmentation
        self.gpu = gpu
        self.cuda = cuda
        self.msg_bot = msg_bot
        self.data_type = data_type
        self.ds_name = ds_name
        self.nr_images = nr_images
        self.restore = restore
        self.data_augmented = data_augmented
        self.inference_name = inference_name # Name of inference instance
        self.fft_for_inference = fft_for_inference # Indicates if fft is used
        self.data_path = os.path.join(os.environ["WORKFLOW_DIR"], os.environ["OPERATOR_IN_DIR"]) # Inference Data
        self.data_dataset_path = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"], os.environ["PREPROCESSED_OPERATOR_OUT_DATA_DIR"])
        self.data_without_fft_path = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"], os.environ["PREPROCESSED_OPERATOR_OUT_DATA_WITHOUT_FFT_DIR"])
        self.train_path = os.path.join(os.environ["TRAIN_WORKFLOW_DIR"], os.environ["OPERATOR_IN_DIR"]) # Train Data
        self.train_dataset_path = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"], os.environ["PREPROCESSED_OPERATOR_OUT_TRAIN_DIR"])
        self.train_without_fft_path = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"], os.environ["PREPROCESSED_OPERATOR_OUT_TRAIN_WITHOUT_FFT_DIR"])
        self.test_path = os.path.join(os.environ["TEST_WORKFLOW_DIR"], os.environ["OPERATOR_IN_DIR"]) # Test Data
        self.test_dataset_path = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"], os.environ["PREPROCESSED_OPERATOR_OUT_TEST_DIR"])
        self.test_without_fft_path = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"], os.environ["PREPROCESSED_OPERATOR_OUT_TEST_WITHOUT_FFT_DIR"])
        self.segmentation_path = os.environ["SEGMENTATION_WORKFLOW_DIR"] # Segmentation data


        if build_dataset:
            instances = self.buildDataset(dtype, noise, seed)
            super().__init__(instances=instances, name=self.ds_name, modality='CT')

    def preprocess(self):
        r"""This function preprocesses (and augments) the input data."""
        # Delete data in directory and preprocess data.
        try:
            if self.data_type == 'inference': 
                delete_images_and_labels(self.data_dataset_path)
                delete_images_and_labels(self.data_without_fft_path)
                augmentation_inference(self.data_path, self.data_dataset_path)
                
            if self.data_type == 'train':
                if not self.restore:
                    delete_images_and_labels(self.train_dataset_path)
                    delete_images_and_labels(self.train_without_fft_path)
                    augment_data_aug_blur(self.train_path, self.train_dataset_path, img_size=(1, 35, 51, 35))
                    augment_data_aug_noise(self.train_path, self.train_dataset_path, img_size=(1, 35, 51, 35))
                    augment_data_aug_motion(self.train_path, self.train_dataset_path, self.train_without_fft_path, img_size=(1, 35, 51, 35))
                    augment_data_aug_ghosting(self.train_path, self.train_dataset_path, self.train_without_fft_path, img_size=(1, 35, 51, 35))
                    augment_data_aug_spike(self.train_path, self.train_dataset_path, self.train_without_fft_path, img_size=(1, 35, 51, 35))
                else:
                    augment_data_aug_blur(self.train_path, self.train_dataset_path, img_size=(1, 35, 51, 35))
                    augment_data_aug_noise(self.train_path, self.train_dataset_path, img_size=(1, 35, 51, 35))
                    augment_data_aug_motion(self.train_path, self.train_dataset_path, self.train_without_fft_path, img_size=(1, 35, 51, 35))
                    augment_data_aug_ghosting(self.train_path, self.train_dataset_path, self.train_without_fft_path, img_size=(1, 35, 51, 35))
                    augment_data_aug_spike(self.train_path, self.train_dataset_path, self.train_without_fft_path, img_size=(1, 35, 51, 35))
                generate_train_labels(self.num_intensities, self.train_path, self.train_dataset_path, True)
                
            if self.data_type == 'test':
                delete_images_and_labels(self.test_dataset_path)
                delete_images_and_labels(self.test_without_fft_path)
                augment_data_aug_blur(self.test_path, self.test_dataset_path, img_size=(1, 35, 51, 35))
                augment_data_aug_noise(self.test_path, self.test_dataset_path, img_size=(1, 35, 51, 35))
                augment_data_aug_motion(self.test_path, self.test_dataset_path, self.test_without_fft_path, img_size=(1, 35, 51, 35))
                augment_data_aug_ghosting(self.test_path, self.test_dataset_path, self.test_without_fft_path, img_size=(1, 35, 51, 35))
                augment_data_aug_spike(self.test_path, self.test_dataset_path, self.test_without_fft_path, img_size=(1, 35, 51, 35))
                generate_train_labels(self.num_intensities, self.test_path, self.test_dataset_path)

            if self.data_type == 'all':
                delete_images_and_labels(self.data_dataset_path)
                delete_images_and_labels(self.data_without_fft_path)
                
                if not self.restore:
                    delete_images_and_labels(self.train_dataset_path)
                    delete_images_and_labels(self.train_without_fft_path)
                    augment_data_aug_blur(self.train_path, self.train_dataset_path, img_size=(1, 35, 51, 35))
                    augment_data_aug_noise(self.train_path, self.train_dataset_path, img_size=(1, 35, 51, 35))
                    augment_data_aug_motion(self.train_path, self.train_dataset_path, self.train_without_fft_path, img_size=(1, 35, 51, 35))
                    augment_data_aug_ghosting(self.train_path, self.train_dataset_path, self.train_without_fft_path, img_size=(1, 35, 51, 35))
                    augment_data_aug_spike(self.train_path, self.train_dataset_path, self.train_without_fft_path, img_size=(1, 35, 51, 35))
                else:
                    augment_data_aug_blur(self.train_path, self.train_dataset_path, img_size=(1, 35, 51, 35))
                    augment_data_aug_noise(self.train_path, self.train_dataset_path, img_size=(1, 35, 51, 35))
                    augment_data_aug_motion(self.train_path, self.train_dataset_path, self.train_without_fft_path, img_size=(1, 35, 51, 35))
                    augment_data_aug_ghosting(self.train_path, self.train_dataset_path, self.train_without_fft_path, img_size=(1, 35, 51, 35))
                    augment_data_aug_spike(self.train_path, self.train_dataset_path, self.train_without_fft_path, img_size=(1, 35, 51, 35))
                generate_train_labels(self.num_intensities, self.train_path, self.train_dataset_path, True)

                delete_images_and_labels(self.test_dataset_path)
                delete_images_and_labels(self.test_without_fft_path)
                augment_data_aug_blur(self.test_path, self.test_dataset_path, img_size=(1, 35, 51, 35))
                augment_data_aug_noise(self.test_path, self.test_dataset_path, img_size=(1, 35, 51, 35))
                augment_data_aug_motion(self.test_path, self.test_dataset_path, self.test_without_fft_path, img_size=(1, 35, 51, 35))
                augment_data_aug_ghosting(self.test_path, self.test_dataset_path, self.test_without_fft_path, img_size=(1, 35, 51, 35))
                augment_data_aug_spike(self.test_path, self.test_dataset_path, self.test_without_fft_path, img_size=(1, 35, 51, 35))
                generate_test_labels(self.num_intensities, self.test_path, self.test_dataset_path)

            if self.data_type == 'segmentation':
                delete_images_and_labels(self.segmentation_path)
                augmentation_segmentation(self.train_path, self.segmentation_path, image_type = 'img', img_size=(1,35,51,35))
                augmentation_segmentation(self.train_path, self.segmentation_path, image_type = 'seg', img_size=(1,35,51,35))


            return True, None
        except: # catch *all* exceptions
            e = traceback.format_exc()
            return False, e

    def buildDataset(self, dtype, noise, seed):
        r"""This function builds a dataset from the preprocessed (and augmented) data based on the dtype,
            either for training, testing, inference or segmentation. The dtype is the same as self.data_type, however it can not be
            'all' in this case, since it is important to be able to distinguish to which type a scan belongs
            (train -- inference). Noise specifies which data will be included in the dataset --> only used
            for training. ds_name specifies which dataset should be build, based on its name (in foldername).
            ds_name is only necessary for dtype == 'train'.
            NOTE: The function checks, if data is in the preprocessed folder, this does not mean, that it ensures
                  that the data is also augmented! If there is only preprocessed data (i.e. resampled and centre cropped),
                  then the preprocessing step should be performed again since this process includes the augmentation
                  (only for train data needed). In such a case, data_augmented in the config file should be set to False,
                  i.e. data is not augmentated and needs to be done."""
        # Extract all images, if not already done
        if dtype == 'train':
            if not os.path.isdir(self.train_dataset_path) or not os.listdir(self.train_dataset_path):
                print("Train data needs to be preprocessed..")
                self.data_type = dtype
                self.preprocess()
            if not self.data_augmented:
                augment_data_aug_blur(self.train_path, self.train_dataset_path, img_size=(1, 35, 51, 35))
                augment_data_aug_noise(self.train_path, self.train_dataset_path, img_size=(1, 35, 51, 35))
                augment_data_aug_motion(self.train_path, self.train_dataset_path, self.train_without_fft_path, img_size=(1, 35, 51, 35))
                augment_data_aug_ghosting(self.train_path, self.train_dataset_path, self.train_without_fft_path, img_size=(1, 35, 51, 35))
                augment_data_aug_spike(self.train_path, self.train_dataset_path, self.train_without_fft_path, img_size=(1, 35, 51, 35))

        elif 'test' in dtype:
            if not os.path.isdir(self.test_dataset_path) or not os.listdir(self.test_dataset_path):
                print("Test data needs to be preprocessed..")
                self.data_type = 'test'
                self.preprocess()

        elif 'inference' in dtype:
            if not os.path.isdir(self.data_dataset_path) or not os.listdir(self.data_dataset_path):
                print("Inference data needs to be preprocessed..")
                self.data_type = dtype
                self.preprocess()
        
        elif 'segmentation' in dtype:
            self.data_type = dtype
            self.preprocess()

        # Assert if dtype is 'all'
        assert dtype != 'all', "The dataset type can not be all, it needs to be either 'train' or 'inference'!"

        # Build dataset based on dtype
        if dtype == 'inference':
            # Build instances, dataset without labels! Dataset can be build with or without fft depending on fft_for_inference
            instances = list()
            print()
            
            msg = 'Creating inference dataset from images: '
            #msg += str(num + 1) + ' of ' + str(len(study_names)) + '.'
            print (msg, end = '\r')
            if self.fft_for_inference:
                # Build dataset with fft
                instances.append(CNNInstance(
                    x_path = os.path.join(self.data_dataset_path, self.inference_name, 'img', 'fft.nii.gz'),
                    y_label = torch.tensor(1),
                    name = self.inference_name,
                    group_id = None
                    ))
            else:
                # Build dataset without fft
                instances.append(CNNInstance(
                    x_path = os.path.join(self.data_dataset_path, self.inference_name, 'img', 'img.nii.gz'),
                    y_label = torch.tensor(1),
                    name = self.inference_name,
                    group_id = None
                    ))

        if dtype == 'train':        
            # Foldernames are patient_id
            study_names = [x for x in os.listdir(self.train_dataset_path) if '._' not in x]

            # Load labels and build one hot vector
            labels = lr.load_json(self.train_dataset_path, 'labels.json')
            one_hot = torch.nn.functional.one_hot(torch.arange(0, self.num_intensities), num_classes=self.num_intensities)

            # Load labels for selecting data equally distributed
            swapped_labels = lr.load_json(self.train_dataset_path, 'labels_swapped.json')

            # Build instances list
            instances = list()
            print()


            study_names = _get_equally_distributed_names(study_names, swapped_labels, self.ds_name, noise, self.nr_images, self.num_intensities, seed)
            # Initialize list for group_ids 
            group_ids = []
            # Build instances
            for num, name in enumerate(study_names):
                msg = 'Creating dataset from images: '
                msg += str(num + 1) + ' of ' + str(len(study_names)) + '.'
                print (msg, end = '\r')
                group_name = '_'.join(name.split('_')[:-1])

                # Extract group_id
                if group_name not in group_ids:
                    group_ids.append(group_name)
                group_id = group_ids.index(group_name)

                instances.append(CNNInstance(
                    x_path = os.path.join(self.train_dataset_path, name, 'img', 'img.nii.gz'),
                    y_label = one_hot[int(labels[name] * self.num_intensities) - 1],
                    name = name, group_id = group_id))




        if 'test' in dtype:
            # Foldernames are patient_id based on dtype
            
            study_names = [x for x in os.listdir(self.test_dataset_path) if '._' not in x and '.json' not in x]
       
            # Load labels and build one hot vector
            labels = lr.load_json(self.test_dataset_path, 'labels.json')
            one_hot = torch.nn.functional.one_hot(torch.arange(0, self.num_intensities), num_classes=self.num_intensities)

            # Build instances
            instances = list()
            print()

            # Build instances
            for num, name in enumerate(study_names):
                if noise in name:
                    msg = 'Creating test dataset from images: '
                    msg += str(num + 1) + ' of ' + str(len(study_names)) + '.'
                    print (msg, end = '\r')
                    a_name = name
                    instances.append(CNNInstance(
                        x_path = os.path.join(self.test_dataset_path, name, 'img', 'img.nii.gz'),
                        y_label = one_hot[int(labels[a_name] * self.num_intensities) - 1],
                        name = name,
                        group_id = None
                        ))

        if 'segmentation' in dtype:
            # Returns study_names for segmentation
            study_names = [x for x in os.listdir(self.train_dataset_path) if '._' not in x and '.json' not in x]

        return instances


def _get_equally_distributed_names(study_names, labels, ds_name, noise, nr_images, num_intensities, seed):
    r"""Extracts a list of folder names representing ds_name Dataset, based on noise and nr_images.
        An equal distribution of images will be extracted, ie. nr_images from each intensity level resulting
        in a dataset of num_intensities x nr_images foldernames."""
    # Set random seed
    random.seed(seed)
    # Select intensities equally
    ds_names = list()
    for i in range(1, num_intensities+1):
        labels_key = str(i/num_intensities) + '_' + noise
        possible_values = labels[labels_key]
        # Select only files from the current dataset with the current intensity level and where the name matches its label
        intensity_names = [x for x in possible_values if ds_name in x and x in study_names]
        # Select random names
        if len(intensity_names) > nr_images:
            ds_names.extend(random.sample(intensity_names, nr_images))
        else:
            ds_names.extend(intensity_names)

    # Reset random seed
    random.seed()
    return ds_names