import os 
import nibabel as nib
import torch
import numpy as np

location = r"path_in"
output_folder = r"path_out"

slices = False

loss_list = {}

patch_size = 256

for file_name in os.listdir(location): 
    if ".nii.gz" in file_name and "label" in file_name: 
        prediction_name = os.path.join(location, file_name) 
        image_name = prediction_name.replace("-label", "")
        
        prediction_data = nib.load(prediction_name).get_fdata()[:,:,:]
        image_Data = nib.load(image_name).get_fdata()[:,:,:,0]

        if slices:
            # 2D
            for pos_z in range(image_Data.shape[2]):
                # median of slice
                if prediction_data.shape[2]-1 < pos_z:
                    continue
                indices_LV = np.argwhere((prediction_data[:,:,pos_z] == 1))
                if len(indices_LV) == 0:
                    continue
                pos_x = np.median(indices_LV[:,0])
                pos_y = np.median(indices_LV[:,1])


                pos_x_left = int(min(max(pos_x - patch_size/2, 0), image_Data.shape[0]-patch_size))
                pos_x_right = pos_x_left + patch_size
                pos_y_bottom = int(min(max(pos_y - patch_size/2, 0), image_Data.shape[1]-patch_size))
                pos_y_top = pos_y_bottom + patch_size

                patch_name = os.path.basename(image_name)[:-7] + "_" + str(pos_z) + ".nii.gz"

                patch = image_Data[pos_x_left:pos_x_right, pos_y_bottom:pos_y_top, pos_z]
                nib_image = nib.Nifti1Image(patch, np.eye(4))
                nib.save(nib_image, os.path.join(output_folder, patch_name))
        else:
            # 3D
            indices_LV = np.argwhere((prediction_data[:,:,:] == 1))
            if len(indices_LV) == 0:
                continue
            pos_x = np.median(indices_LV[:,0])
            pos_y = np.median(indices_LV[:,1])

            pos_x_left = int(min(max(pos_x - patch_size/2, 0), image_Data.shape[0]-patch_size))
            pos_x_right = pos_x_left + patch_size
            pos_y_bottom = int(min(max(pos_y - patch_size/2, 0), image_Data.shape[1]-patch_size))
            pos_y_top = pos_y_bottom + patch_size

            patch = image_Data[pos_x_left:pos_x_right, pos_y_bottom:pos_y_top, :]
            nib_image = nib.Nifti1Image(patch, np.eye(4))

            print("OUTSAVE: " + str(os.path.join(output_folder, os.path.basename(image_name))))
            nib.save(nib_image, os.path.join(output_folder, os.path.basename(image_name)))

