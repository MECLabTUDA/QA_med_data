# ------------------------------------------------------------------------------
# Data augmentation. Only, transformations from the TorchIO library are 
# used (https://torchio.readthedocs.io/transforms/augmentation.html).
# ------------------------------------------------------------------------------

# Imports
import torchio as tio
import SimpleITK as sitk
import os
import torch
import numpy as np 
from mp.data.pytorch.transformation import centre_crop_pad_3d

# Perfom augmentation on dataset
def augment_image_in_four_intensities(image, noise, mean_val):
    r"""This function takes an image and augments it in 4 intensities for one of 5 artefacts:
        - Blurring
        - Ghosting
        - Motion
        - (Gaussian) Noise
        - Spike
    """

    # Define augmentation methods
    if noise == 'blur':
        blur2 = random_blur(std=0.4)
        blur3 = random_blur(std=0.75)
        blur4 = random_blur(std=1)
        blur5 = random_blur(std=1.75)
        return blur2(image), blur3(image), blur4(image), blur5(image)


    if noise == 'ghosting':
        ghosting2 = random_ghosting(num_ghosts = (1,1), intensity=(0.55,0.55))
        ghosting3 = random_ghosting(num_ghosts = (2,2), intensity=(0.62,0.62))
        ghosting4 = random_ghosting(num_ghosts = (3,3), intensity=(0.71,0.71))
        ghosting5 = random_ghosting(num_ghosts = (4,4), intensity=(0.8,0.8))
        return ghosting2(image), ghosting3(image), ghosting4(image), ghosting5(image)

    if noise == 'motion':
        motion2 = random_motion(degrees=(2,2), translation=(1,1), num_transforms=1)
        motion3 = random_motion(degrees=(2.5,2.5), translation=(1.2,1.2), num_transforms=2)
        motion4 = random_motion(degrees=(3,3), translation=(1.4,1.4), num_transforms=3)
        motion5 = random_motion(degrees=(4,4), translation=(1.6,1.6), num_transforms=3)
        return motion2(image), motion3(image), motion4(image), motion5(image)

    # Mean is set at the mean value of the image and the standard deviation also depends on the mean value of the image.
    # The mean value of the images differs strongly between different images and the noise is added with absolut values. 
    # If you don't correlate the input parameters of random_noise with the mean value you receive very different 
    # impressions of the intensity of the noiseon different images. 
    if noise == 'noise':
        noise2 = random_noise(mean=mean_val,std=0.02*mean_val)
        noise3 = random_noise(mean=mean_val,std=0.04*mean_val)
        noise4 = random_noise(mean=mean_val,std=0.06*mean_val)
        noise5 = random_noise(mean=mean_val,std=0.08*mean_val)
        return noise2(image), noise3(image), noise4(image), noise5(image)

    if noise == 'spike':
        spike2 = random_spike(num_spikes=(2,2), intensity=(0.028, 0.028))
        spike3 = random_spike(num_spikes=(3,3), intensity=(0.042, 0.042))
        spike4 = random_spike(num_spikes=(4,4), intensity=(0.055, 0.055))
        spike5 = random_spike(num_spikes=(5,5), intensity=(0.07, 0.07))
        return spike2(image), spike3(image), spike4(image), spike5(image)


# Intensity Functions for data Augmentation

def random_blur(std):
    r"""Blur an image using a random-sized Gaussian filter.
    - std: Tuple (a,b) to compute the standard deviations (𝜎1,𝜎2,𝜎3)
        of the Gaussian kernels used to blur the image along each axis.
    - p: Probability that this transform will be applied.
    - seed: Seed for torch random number generator.
    - keys: Mandatory if the input is a Python dictionary.
        The transform will be applied only to the data in each key."""
    
    blur = tio.RandomBlur(std=std)
    return blur

def random_ghosting(num_ghosts, intensity):
    r"""Add random MRI ghosting artifact.
    - num_ghosts: Number of ‘ghosts’ n in the image.
    - axes: Axis along which the ghosts will be created. If axes is a
        tuple, the axis will be randomly chosen from the passed values.
        Anatomical labels may also be used.
    - intensity: Positive number representing the artifact strength s
        with respect to the maximum of the k-space. If 0, the ghosts
        will not be visible.
    - restore: Number between 0 and 1 indicating how much of the k-space
        center should be restored after removing the planes that generate
        the artifact.
    - p: Probability that this transform will be applied.
    - seed: Seed for torch random number generator.
    - keys: Mandatory if the input is a Python dictionary.
        The transform will be applied only to the data in each key."""
    
    ghosting = tio.RandomGhosting(num_ghosts=num_ghosts, intensity=intensity)
    return ghosting

def random_motion(degrees=10, translation=10, num_transforms=2):
    r"""Add random MRI motion artifact.
    - degrees: Tuple (a, b) defining the rotation range in degrees of
        the simulated movements.
    - translation: Tuple (a,b) defining the translation in mm
        of the simulated movements.
    - num_transforms: Number of simulated movements. Larger values generate
        more distorted images.
    - image_interpolation: 'nearest' can be used for quick experimentation as
        it is very fast, but produces relatively poor results. 'linear',
        default in TorchIO, is usually a good compromise between image
        quality and speed to be used for data augmentation during training.
        Methods such as 'bspline' or 'lanczos' generate high-quality
        results, but are generally slower. They can be used to obtain
        optimal resampling results during offline data preprocessing.
    - p: Probability that this transform will be applied.
    - seed: Seed for torch random number generator.
    - keys: Mandatory if the input is a Python dictionary.
        The transform will be applied only to the data in each key."""
    
    motion = tio.RandomMotion(degrees, translation, num_transforms)
    return motion

def random_noise(mean, std):
    r"""Add random Gaussian noise.
    - mean: Mean μ of the Gaussian distribution from which the noise is sampled.
    - std: Standard deviation σ of the Gaussian distribution from which
        the noise is sampled.
    - p: Probability that this transform will be applied.
    - seed: Seed for torch random number generator.
    - keys: Mandatory if the input is a Python dictionary.
        The transform will be applied only to the data in each key."""
    
    noise = tio.RandomNoise(mean, std)
    return noise

def random_spike(num_spikes, intensity):
    r"""Add random MRI spike artifacts.
    - num_spikes: Number of spikes n present in k-space. Larger values generate
        more distorted images.
    - intensity: Ratio r between the spike intensity and the maximum of
        the spectrum. Larger values generate more distorted images.
    - p: Probability that this transform will be applied.
    - seed: Seed for torch random number generator.
    - keys: Mandatory if the input is a Python dictionary.
        The transform will be applied only to the data in each key."""
    
    spike = tio.RandomSpike(num_spikes, intensity)
    return spike


def augment_data_aug_motion(source_path, target_path, without_fft_path, img_size=(1, 35, 51, 35)):
    r"""This function augments Data for the artefact motion. 
        It saves the augmented images with and without fft, 
        because the motion classifier uses fft images.
    """

    torch.manual_seed(42)
    # Indicates if the difference between the fft of the image without artefact and the fft of the image with artifact 
    # is calculated and saved. This can be used to demonstate the differences between the ffts of different artifacts or intensities
    diff = False

    #Extract filenames
    filenames = [x for x in os.listdir(source_path)]

    # Create a path to save the differences in
    if diff:
        os.makedirs(os.path.join(target_path, 'diff_images'))

    for num, filename in enumerate(filenames):
        print('motion: ' + str(num+1) +'/' + str(len(filenames)))
        img = sitk.ReadImage(os.path.join(source_path, filename,'img', 'img.nii.gz'))
        img_array = sitk.GetArrayFromImage(img)
        # Calculate mean value of the image
        mean_val = np.mean(img_array)

        # Centercrop and pad the image without artifact
        if diff:
            img_array_crop = centre_crop_pad_3d(torch.from_numpy(img_array).unsqueeze_(0), img_size)[0]
            diff_img = []

        # Augment image
        motion2, motion3, motion4, motion5 = augment_image_in_four_intensities(img, 'motion', mean_val)

        motion = [img, motion2, motion3, motion4, motion5]
        motion_without_fft = []

        for mo in range(len(motion)):
            x_s = torch.from_numpy(sitk.GetArrayFromImage(motion[mo])).unsqueeze_(0)
            # Centercrop and pad all images to the same size
            motion[mo] = centre_crop_pad_3d(x_s, img_size)[0]
            motion_without_fft.append(motion[mo])
            
            # Calculate the difference of the ffts. 
            if diff:
                img_basic = np.fft.fftn(img_array_crop)
                fft_shift_basic = np.fft.fftshift(img_basic)
                fft_trans_abs_basic = np.abs(fft_shift_basic)
                fft_trans = np.fft.fftn(motion[mo])
                fft_shift = np.fft.fftshift(fft_trans)
                fft_trans_abs = np.abs(fft_shift)

                fftdiff = np.subtract(np.array(fft_trans_abs_basic), np.array(fft_trans_abs))
                fft_trans_log = np.log(fftdiff)
                diff_img.append(fft_trans_log) 
            
            # Transform the augmented image to fft
            fft_trans = np.fft.fftn(motion[mo])
            fft_trans_abs = np.abs(fft_trans)
            motion[mo] = fft_trans_abs

            # Can be used for visualization
            '''fft_trans = np.fft.fftn(motion[mo])
            fft_shift = np.fft.fftshift(fft_trans)
            fft_trans_abs = np.abs(fft_shift)
            fft_trans_log = np.log(fft_trans_abs)
            motion[mo] = fft_trans_log'''

        # Save images
        for ind, i in enumerate(range(5, 0, -1)):
            a_filename = filename.split('.')[0] + '_' + 'motion' + str(i)
            
            # Save fft images
            os.makedirs(os.path.join(target_path, a_filename, 'img'))
            x = sitk.GetImageFromArray(motion[ind])
            sitk.WriteImage(x, os.path.join(target_path, a_filename, 'img', 'img.nii.gz'))

            # Save imaged without fft
            os.makedirs(os.path.join(without_fft_path, a_filename, 'img'))
            x_not_fft = sitk.GetImageFromArray(motion_without_fft[ind])
            sitk.WriteImage(x_not_fft, os.path.join(without_fft_path, a_filename, 'img', 'img.nii.gz'))

            # Save the differences
            if diff:
                x = sitk.GetImageFromArray(diff_img[ind])
                sitk.WriteImage(x, os.path.join(target_path, 'diff_images', a_filename+'_diff.nii.gz'))


def augment_data_aug_blur(source_path, target_path, img_size=(1, 35, 51, 35)):
    r"""This function augments Data for the artefact blur. 
    """
    
    torch.manual_seed(42)
    #Extract filenames
    filenames = [x for x in os.listdir(source_path)]

    # Loop through filenames to augment and save every image
    for num, filename in enumerate(filenames):
        print('blur: ' + str(num+1) +'/' + str(len(filenames)))
        img = sitk.ReadImage(os.path.join(source_path, filename,'img', 'img.nii.gz'))        
        img_array = sitk.GetArrayFromImage(img)
        # Calculate mean value of the image
        mean_val = np.mean(img_array)

        # Augment image
        blur2, blur3, blur4, blur5 = augment_image_in_four_intensities(img, 'blur', mean_val)
        blur = [img, blur2, blur3, blur4, blur5]
        for bl in range(len(blur)):
            x_s = torch.from_numpy(sitk.GetArrayFromImage(blur[bl])).unsqueeze_(0)
            # Centercrop and pad all images to the same size
            blur[bl] = centre_crop_pad_3d(x_s, img_size)[0]

        # Save images
        for ind, i in enumerate(range(5, 0, -1)):
            a_filename = filename.split('.')[0] + '_' + 'blur' + str(i)
            os.makedirs(os.path.join(target_path, a_filename, 'img'))
            x = sitk.GetImageFromArray(blur[ind])
            sitk.WriteImage(x, os.path.join(target_path, a_filename, 'img', 'img.nii.gz'))


def augment_data_aug_ghosting(source_path, target_path, without_fft_path, img_size=(1, 35, 51, 35)):
    r"""This function augments Data for the artefact ghosting. 
        It saves the augmented images with and without fft, 
        because the ghosting classifier uses fft images.
    """
    
    torch.manual_seed(42)
    # Indicates if the difference between the fft of the image without artefact and the fft of the image with artifact 
    # is calculated and saved. This can be used to demonstate the differences between the ffts of different artifacts or intensities
    diff = False
    #Extract filenames
    filenames = [x for x in os.listdir(source_path)]
    # Loop through filenames to augment and save every image
    for num, filename in enumerate(filenames):
        print('ghosting: ' + str(num+1) +'/' + str(len(filenames)))
        img = sitk.ReadImage(os.path.join(source_path, filename,'img', 'img.nii.gz'))        
        img_array = sitk.GetArrayFromImage(img)
        # Calculate mean value of the image
        mean_val = np.mean(img_array)

        # Centercrop and pad the image without artifact
        if diff:
            img_array_crop = centre_crop_pad_3d(torch.from_numpy(img_array).unsqueeze_(0), img_size)[0]
            diff_img = []

        # Augment image
        ghosting2, ghosting3, ghosting4, ghosting5 = augment_image_in_four_intensities(img, 'ghosting', mean_val)
        ghosting = [img, ghosting2, ghosting3, ghosting4, ghosting5]
        ghosting_without_fft = []
        for gho in range(len(ghosting)):
            x_s = torch.from_numpy(sitk.GetArrayFromImage(ghosting[gho])).unsqueeze_(0)
            # Centercrop and pad all images to the same size
            ghosting[gho] = centre_crop_pad_3d(x_s, img_size)[0]

            ghosting_without_fft.append(ghosting[gho])

            # Calculate the difference of the ffts. 
            if diff:
                img_basic = np.fft.fftn(img_array_crop)
                fft_shift_basic = np.fft.fftshift(img_basic)
                fft_trans_abs_basic = np.abs(fft_shift_basic)
                fft_trans = np.fft.fftn(ghosting[gho])
                fft_shift = np.fft.fftshift(fft_trans)
                fft_trans_abs = np.abs(fft_shift)

                fftdiff = np.subtract(np.array(fft_trans_abs_basic), np.array(fft_trans_abs))
                fft_trans_log = np.log(fftdiff)
                diff_img.append(fft_trans_log)

            # Transform the augmented image to fft
            fft_trans = np.fft.fftn(ghosting[gho])
            fft_trans_abs = np.abs(fft_trans)
            ghosting[gho] = fft_trans_abs

        # Save images
        for ind, i in enumerate(range(5, 0, -1)):
            a_filename = filename.split('.')[0] + '_' + 'ghosting' + str(i)

            # Save fft images
            os.makedirs(os.path.join(target_path, a_filename, 'img'))
            x = sitk.GetImageFromArray(ghosting[ind])
            sitk.WriteImage(x, os.path.join(target_path, a_filename, 'img', 'img.nii.gz'))

            # Save imaged without fft
            os.makedirs(os.path.join(without_fft_path, a_filename, 'img'))
            x_not_fft = sitk.GetImageFromArray(ghosting_without_fft[ind])
            sitk.WriteImage(x_not_fft, os.path.join(without_fft_path, a_filename, 'img', 'img.nii.gz'))

            # Save the differences
            if diff:
                os.makedirs(os.path.join(target_path, 'diff_images'))
                x = sitk.GetImageFromArray(diff_img[ind])
                sitk.WriteImage(x, os.path.join(target_path, 'diff_images', a_filename+'_diff.nii.gz'))


def augment_data_aug_noise(source_path, target_path, img_size=(1, 35, 51, 35)):
    r"""This function augments Data for the artefact noise. 
    """
    
    torch.manual_seed(42)
    #Extract filenames
    filenames = [x for x in os.listdir(source_path)]
    # Loop through filenames to augment and save every image
    for num, filename in enumerate(filenames):
        print('noise: ' + str(num+1) +'/' + str(len(filenames)))
        img = sitk.ReadImage(os.path.join(source_path, filename,'img', 'img.nii.gz'))        
        img_array = sitk.GetArrayFromImage(img)
        # Calculate mean value of the image
        mean_val = np.mean(img_array)

        # Augment image
        noise2, noise3, noise4, noise5 = augment_image_in_four_intensities(img, 'noise', mean_val)
        noise = [img, noise2, noise3, noise4, noise5]
        for no in range(len(noise)):
            x_s = torch.from_numpy(sitk.GetArrayFromImage(noise[no])).unsqueeze_(0)
            # Centercrop and pad all images to the same size
            noise[no] = centre_crop_pad_3d(x_s, img_size)[0]

        # Save images
        for ind, i in enumerate(range(5, 0, -1)):
            a_filename = filename.split('.')[0] + '_' + 'noise' + str(i)
            os.makedirs(os.path.join(target_path, a_filename, 'img'))
            x = sitk.GetImageFromArray(noise[ind])
            sitk.WriteImage(x, os.path.join(target_path, a_filename, 'img', 'img.nii.gz'))


def augment_data_aug_spike(source_path, target_path, without_fft_path, img_size=(1, 35, 51, 35)):
    r"""This function augments Data for the artefact spike. 
        It saves the augmented images with and without fft, 
        because the spike classifier uses fft images.
    """
    
    torch.manual_seed(42)
    # Indicates if the difference between the fft of the image without artefact and the fft of the image with artifact 
    # is calculated and saved. This can be used to demonstate the differences between the ffts of different artifacts or intensities
    diff = False

    #Extract filenames
    filenames = [x for x in os.listdir(source_path)]
    # Loop through filenames to augment and save every image
    for num, filename in enumerate(filenames):
        print('spike: ' + str(num+1) +'/' + str(len(filenames)))
        

        img = sitk.ReadImage(os.path.join(source_path, filename,'img', 'img.nii.gz'))        
        img_array = sitk.GetArrayFromImage(img)
        # Calculate mean value of the image
        mean_val = np.mean(img_array)

        # Centercrop and pad the image without artifact
        if diff:
            img_array_crop = centre_crop_pad_3d(torch.from_numpy(img_array).unsqueeze_(0), img_size)[0]
            diff_img = []

        # Augment image
        spike2, spike3, spike4, spike5 = augment_image_in_four_intensities(img, 'spike', mean_val)
        spike = [img, spike2, spike3, spike4, spike5]
        spike_without_fft = []

        for sp in range(len(spike)):
            x_s = torch.from_numpy(sitk.GetArrayFromImage(spike[sp])).unsqueeze_(0)
            # Centercrop and pad all images to the same size
            spike[sp] = centre_crop_pad_3d(x_s, img_size)[0]
            spike_without_fft.append(spike[sp])

            # Calculate the difference of the ffts. 
            if diff:
                img_basic = np.fft.fftn(img_array_crop)
                fft_shift_basic = np.fft.fftshift(img_basic)
                fft_trans_abs_basic = np.abs(fft_shift_basic)
                fft_trans = np.fft.fftn(spike[sp])
                fft_shift = np.fft.fftshift(fft_trans)
                fft_trans_abs = np.abs(fft_shift)

                fftdiff = np.subtract(np.array(fft_trans_abs_basic), np.array(fft_trans_abs))
                fft_trans_log = np.log(fftdiff)
                diff_img.append(fft_trans_log)

            # Transform the augmented image to fft
            fft_trans = np.fft.fftn(spike[sp])
            fft_trans_abs = np.abs(fft_trans)
            spike[sp] = fft_trans_abs

        # Save images
        for ind, i in enumerate(range(5, 0, -1)):
            a_filename = filename.split('.')[0] + '_' + 'spike' + str(i)

            # Save fft images
            os.makedirs(os.path.join(target_path, a_filename, 'img'))
            x = sitk.GetImageFromArray(spike[ind])
            sitk.WriteImage(x, os.path.join(target_path, a_filename, 'img', 'img.nii.gz'))
            
            # Save imaged without fft
            os.makedirs(os.path.join(without_fft_path, a_filename, 'img'))
            x_not_fft = sitk.GetImageFromArray(spike_without_fft[ind])
            sitk.WriteImage(x_not_fft, os.path.join(without_fft_path, a_filename, 'img', 'img.nii.gz'))

            # Save the differences
            if diff:
                os.makedirs(os.path.join(target_path, 'diff_images'))
                x = sitk.GetImageFromArray(diff_img[ind])
                sitk.WriteImage(x, os.path.join(target_path, 'diff_images', a_filename+'_diff.nii.gz'))





def augmentation_segmentation(source_path, target_path, image_type, img_size=(1, 35, 51, 35)):
    r"""This function augments Data for the segmentation. 
        It brings data to the expected format for the segmentation UNet.
        Image_type is 'img' or 'seg'.
    """

    filenames = [x for x in os.listdir(source_path)]
    for num, filename in enumerate(filenames):
        print('segmentation: ' + str(num+1) +'/' + str(len(filenames)))

        if image_type == 'img':
            img = sitk.ReadImage(os.path.join(source_path, filename,'img', 'img.nii.gz'))        
            img_array = sitk.GetArrayFromImage(img)
            # Centercrop and pad all images to the same size
            img_array_crop = centre_crop_pad_3d(torch.from_numpy(img_array).unsqueeze_(0), img_size)[0]
            for i in range(img_array_crop.shape[2]):
                slice = img_array_crop[:,:,i]
                x = sitk.GetImageFromArray(slice)
                sitk.WriteImage(x, os.path.join(target_path, 'data', 'imgs', filename+'_'+str(i)+'.nii.gz'))
            
        elif image_type == 'seg':
            img = sitk.ReadImage(os.path.join(source_path, filename,'seg', '001.nii.gz'))        
            img_array = sitk.GetArrayFromImage(img)
            img_array = img_array.astype(np.int32)
            # Centercrop and pad all segmentations to the same size
            img_array_crop = centre_crop_pad_3d(torch.from_numpy(img_array).unsqueeze_(0), img_size)[0]
            for i in range(img_array_crop.shape[2]):
                slice = img_array_crop[:,:,i]
                x = sitk.GetImageFromArray(slice)
                sitk.WriteImage(x, os.path.join(target_path, 'data', 'masks', filename+'_'+str(i)+'.nii.gz'))


# Augments Data for inference
def augmentation_inference(source_path, target_path, img_size=(1, 35, 51, 35)):
    r"""This function augments Data for inference. 
    """

    filenames = [x for x in os.listdir(source_path)]
    for num, filename in enumerate(filenames):
        print('inference augmentation: ' + str(num+1) +'/' + str(len(filenames)))
        img = sitk.ReadImage(os.path.join(source_path, filename,'img', 'img.nii.gz')) 
        img_array = sitk.GetArrayFromImage(img)
        
        # Centercrop and pad all images to the same size
        img_array_crop = centre_crop_pad_3d(torch.from_numpy(img_array).unsqueeze_(0), img_size)[0]

        # Transform the image to fft
        fft_trans = np.fft.fftn(img_array_crop)
        fft_trans_abs = np.abs(fft_trans)
        img_fft = fft_trans_abs

        img_no_fft = sitk.GetImageFromArray(img_array_crop)
        img_with_fft = sitk.GetImageFromArray(img_fft)

        # Save the fft image and the image without fft
        os.makedirs(os.path.join(target_path, filename, 'img'))
        sitk.WriteImage(img_no_fft, os.path.join(target_path, filename, 'img', 'img.nii.gz'))
        sitk.WriteImage(img_with_fft, os.path.join(target_path, filename, 'img', 'fft.nii.gz'))

