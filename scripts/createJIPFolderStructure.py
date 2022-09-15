import os
import shutil
from paths import storage_data_path, JIP_dir
import dataset_utils as du
from JIP import config

def copyFilesGC(datasetName, filenames, datasetDirectory, JIPDirectory):
    r"""Helper function for createJIPFolderStructure that copies all files from datasetDirectory
        to JIPDirectory."""
    print('Copying files from {} into {}: '.format(datasetDirectory, JIPDirectory))
    # Loop through filenames and copy files into JIPDirectory
    for num, fname in enumerate(filenames):
        msg = str(num + 1) + ' of ' + str(len(filenames)) + '.'
        print (msg, end = '\r')
        if not '_seg' in fname:
            foldername = datasetName + '_' + fname.split('.')[0].replace('_ct', '')
            currJIPDirectory = os.path.join(JIPDirectory, foldername, 'img')
            # Create directories if not existing
            if not os.path.isdir(currJIPDirectory):
                os.makedirs(currJIPDirectory)
            # Copy ct scan
            shutil.copy(os.path.join(datasetDirectory, fname), os.path.join(currJIPDirectory, 'img.nii.gz'))

        else:
            foldername = datasetName + '_' + fname.split('.')[0].replace('_seg', '')
            currJIPDirectory = os.path.join(JIPDirectory, foldername, 'seg')
            # Create directories if not existing
            if not os.path.isdir(currJIPDirectory):
                os.makedirs(currJIPDirectory)
            # Copy corresponding segmentation
            shutil.copy(os.path.join(datasetDirectory, fname), os.path.join(currJIPDirectory, '001.nii.gz'))

def copyFilesTrStructure(datasetName, filenames, datasetDirectory, JIPDirectory, take_prefix=False):
    r"""Helper function for createJIPFolderStructure that copies all files from datasetDirectory
        to JIPDirectory. Structure from Deacthlon dataset."""
    datasetSegDirectory = datasetDirectory.replace('imagesTr', 'labelsTr')
    print('Copying files from {} into {}: '.format(datasetDirectory, JIPDirectory))
    # Loop through filenames and copy files into JIPDirectory
    for num, fname in enumerate(filenames):
        msg = str(num + 1) + ' of ' + str(len(filenames)) + '.'
        print (msg, end = '\r')
        foldername = datasetName.replace('Test', '').replace('Train', '') + '_' + fname.split('.')[0].replace('_0000', '')
        currJIPDirectory = os.path.join(JIPDirectory, foldername, 'img')
        # Create directories if not existing
        if not os.path.isdir(currJIPDirectory):
            os.makedirs(currJIPDirectory)
        # Copy ct scan
        shutil.copy(os.path.join(datasetDirectory, fname), os.path.join(currJIPDirectory, 'img.nii.gz'))

        currJIPDirectory = os.path.join(JIPDirectory, foldername, 'seg')
        # Create directories if not existing
        if not os.path.isdir(currJIPDirectory):
            os.makedirs(currJIPDirectory)
        # Copy corresponding segmentation
        if take_prefix:
            shutil.copy(os.path.join(datasetSegDirectory, fname.replace('_0000', '')), os.path.join(currJIPDirectory, '001.nii.gz'))
        else:
            shutil.copy(os.path.join(datasetSegDirectory, fname), os.path.join(currJIPDirectory, '001.nii.gz'))

def createJIPFolderStructure(datasetName, datasetDirectory, JIPDirectory):
    r"""Based on the datasetName, this function copies the nifti files (CT scans and segmentations)
        from the provided datasetDirectory into the JIPDirectory considering the JIP directory
        structure. Each folder will have a special name (based on filename) and includes the datasetName.
        Further this function only selects the training datasets, since validation and test datasets usually
        do not have segmentations for the scans. It also assumes that only one segmentation exists per scan.
        NOTE: The filenames will be renamed to either img.nii or 00X.nii, based on their type (scan/seg),
        however the filename is included in the foldername where both files are saved (including datasetName).
        Currently the files are saved as nii.gz instead of .nii!
        JIP directory structure:
        /
        |---WORKFLOW_DIR
            |---OPERATOR_IN_DIR ==> JIPDirectory
            |   |---0001 ==> datasetName_filename
            |   |   |---img
            |   |   |   |---img.nii ==> ct scan
            |   |   |---seg
            |   |       |---001.nii ==> segmentation
            |   |---0002
            |   |   |---img
            |   |   |   |---img.nii ==> ct scan
            |   |   |---seg
            |   |       |---001.nii ==> segmentation
            |   |---...
            |---OPERATOR_OUT_DIR
            |---OPERATOR_TEMP_DIR
    """


    if config['dataset_name'] in datasetName: 
        datasetDirectory = os.path.join(datasetDirectory, 'imagesTr')
        # Filenames have the form 'XXX_0000.nii.gz'
        filenames = [x for x in os.listdir(datasetDirectory) if 'DS_Store' not in x and '._' not in x]
        filenames.sort()
        # Copy files to JIP Directory
        copyFilesTrStructure(datasetName, filenames, datasetDirectory, JIPDirectory, True)

if __name__ == '__main__':
    # Define environments. They are not necessary for this script, since the data is already in this format on JIP
    # , but for the work on the server this step is necessary.
    os.environ["WORKFLOW_DIR"] = os.path.join(JIP_dir, 'data_dirs')
    os.environ["OPERATOR_IN_DIR"] = "input"
    os.environ["OPERATOR_OUT_DIR"] = "output"
    os.environ["OPERATOR_TEMP_DIR"] = "temp"
    os.environ["OPERATOR_PERSISTENT_DIR"] = os.path.join(JIP_dir, 'data_dirs', 'persistent')
    input_dir = os.path.join(os.environ["WORKFLOW_DIR"], os.environ["OPERATOR_IN_DIR"])
    
    datasetName = 'Task99_HarP' 
    datasetDirectory = du.get_original_data_path(datasetName)
    JIPDirectory = input_dir # --> JIP_dir/data_dirs/input (use predefined environment for this)
    createJIPFolderStructure(datasetName, datasetDirectory, JIPDirectory)

    