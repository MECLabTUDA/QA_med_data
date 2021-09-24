import numpy as np
from mp.utils.Iterators import Component_Iterator
from skimage.measure import label, regionprops
import os 
import json 
import torch
import torchio
from mp.utils.intensities import sample_intensities
from sklearn.mixture import GaussianMixture

# 2 functions to get slice_dice (which is the same as segmentation smoothness)
# sometimes also called dice_score as a feature
def get_array_of_dicescores(seg): 
    '''computes the array of dicescores for the given segmentation,
    it is assumed, that the label of interest is 1 and all other labels are 0.
    More in detail : For every two consecutive slices, the dice score is computed
    throughout the image. In the case of both images being black (no label 1), the dicescore 
    is set to one.

    Args:
        seg (torch.Tensor): the segmentation

    Returns (ndarray): array of dicescores (one for every two consecutive slices in the comp)
    '''
    shape = np.shape(seg)
    nr_slices = shape[0]
    arr_of_dicescores = np.array([])
    
    first_slide = seg[0, :, :]
    first_ones = torch.sum(first_slide)
    for i in range(nr_slices-1):
        second_slide = seg[i+1,:,:]
        second_ones = torch.sum(second_slide)
        intersection = torch.dot(torch.flatten(first_slide),
                                torch.flatten(second_slide))
        # if two consecutive slices are black, set dice score to one
        # leads to ignoring the score
        if not(first_ones+second_ones == 0):
            dice_score = 2*intersection / (first_ones+second_ones)
        else:
            dice_score = 1
        # if two consecutive slides are identical (having dicescore = 1), it is assumed, that they are copied
        # or completly black and thus dont count towards the overall Dicescore
        if not dice_score == 1:
            arr_of_dicescores = np.append(
                arr_of_dicescores, np.array([dice_score]))
        # update index
        first_slide = second_slide
        first_ones = second_ones
    return arr_of_dicescores

def get_dice_averages(img,seg,props):
    '''Computes the average dice score for a connected component of the 
    given img-seg pair. 
    STILL COMPUTED BUT NOT USED ANYMORE: Also computes the average differences between the dice scores 
    and computes that average, because it was observed, that in bad artificial bad segmentations,
    these dice scores had a more rough graph, then good segmentations, thus it is used as feature.
    
    Args: 
        img (torch.Tensor): the image
        seg (torch.Tensor): its segmentation
        props (dict[str->object]): a regionprops dictionary, c.f. skimage-> regionprops
        
    Returns (list(floats)): a list of two values, the avg dice score and the avg dice score difference'''
    min_row, min_col, min_sl, max_row, max_col, max_sl,  = props.bbox
    cut_seg = seg[min_row:max_row,min_col:max_col,min_sl:max_sl]
    arr_of_dicescores = get_array_of_dicescores(cut_seg)

    # compute the average value of dice score
    dice_avg_value = np.average(arr_of_dicescores)

    # compute the average value of dice score changes between slides
    # check for connected component, that is only on one slice
    if len(arr_of_dicescores) < 10:
        dice_diff_avg_value = 1 
    else:
        dice_diff = np.diff(arr_of_dicescores)
        dice_diff_abs = np.absolute(dice_diff)
        dice_diff_avg_value = np.average(dice_diff_abs)

    return [dice_avg_value,dice_diff_avg_value]

#function to compute the so Intesity Mode (see paper)
def mean_var_big_comp(img,seg):
    ''' Computes the mean and variance of the united intensity values 
    of the 4 biggest connected components in the image
    Args:
        img(ndarray or torch tensor): the image 
        seg(ndarray or torch tensor): a segmentation of any tissue in the image 
        
    Returns(float,float): the mean and variance of the sampled intensity values'''
    labeled_image, _ = label(seg, return_num=True)
    props = regionprops(labeled_image)
    props = sorted(props ,reverse=True, key =lambda dict:dict['area'])
    ints = sample_intensities(img,seg,props[0],number=5000)
    dens = GaussianMixture(n_components=1).fit(np.reshape(ints, newshape=(-1,1)))
    mean = dens.means_[0,0]
    var = dens.covariances_[0,0,0]
    return mean, var

# function to compute the percentage of segmented tissue within the lung
def segmentation_in_lung(seg,lung_seg):
    return np.dot(torch.flatten(seg),torch.flatten(lung_seg))/torch.sum(seg)

class Feature_extractor():
    '''A class for extracting feature of img-seg pairs and get arrays of features

    Args: 
        feature (list(str)): Every string in the list is for one feature: 
            (-density_distance : computes the distance between the densities of the img-seg
                pair and the precomputed density )
            -dice_scores : computes the avg dice scores of the components and the avg 
                difference of dice scores # segmentation smoothness ( see paper) 
            -connected components : the number of connected components in the seg
            -gauss params : the mean of a gaussian distribution fitted on sampled intensity values 
                of the 4 biggest components 
            -seg in lung : the percentage of the segmented (as unfectious tissue) area, that lies 
                within the segmentation of the lung 
    '''
    def __init__(self, features=['dice_scores','connected_components','gauss_params','seg_in_lung']):
        self.features = features
        self.nr_features = len(features)

    def get_feature(self,feature,img,seg,lung_seg):
        '''Extracts the given feature for the given img-seg pair

        Args: 
            feature (str): The feature to be extracted 
            img (ndarray): the image 
            seg (ndarray): The corresponding mask
            lung_seg (ndarray): The segmentation of the lung in the image 

        Returns (object): depending on the feature: 
            dice_scores -> (ndarray with two entries): array with two entries, the dice averages and dice_diff averages 
            connected_components -> (integer): The number of connected components
        '''
        component_iterator = Component_Iterator(img,seg)
        original_threshhold = component_iterator.threshold
        if feature == 'dice_scores':
            dice_metrices = component_iterator.iterate(get_dice_averages)
            if not dice_metrices:
                print('Image only has very small components')
                component_iterator.threshold = 0
                dice_metrices = component_iterator.iterate(get_dice_averages)
                component_iterator.threshold = original_threshhold
            if not dice_metrices:
                print('Image has no usable components, no reliable computations can be made for dice')
                return 1
            dice_metrices = np.array(dice_metrices)
            dice_metrices = np.mean(dice_metrices,0)
            return dice_metrices[0]
        if feature == 'connected_components':
            _,number_components = label(seg,return_num=True,connectivity=3)
            return number_components
        if feature == 'gauss_params':
            mean,_ = mean_var_big_comp(img,seg)
            return mean
        if feature == 'seg_in_lung':
            dice_seg_lung = segmentation_in_lung(seg,lung_seg)
            return float(dice_seg_lung)

    def compute_features_id(self,id,features='all'):
        '''Computes all features for the img-seg and img-pred pairs (if existing)
        and saves them in the preprocessed_dir/.../id/...
        Args:
            id (str): the id of the patient to compute the features for
            features (str or list(str)): either all or a list of features to compute
        '''
        #get the features to extract
        if features == 'all':
            features = self.features

        #get id path depending on global mode 
        if not os.environ["INFERENCE_OR_TRAIN"] == 'train':
            id_path = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"],os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR"],id)
        else : 
            id_path = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"],os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR_TRAIN"],id)
        img_path = os.path.join(id_path,'img','img.nii.gz')
        lung_seg_path = os.path.join(id_path,'lung_seg','lung_seg.nii.gz')
        
        #get the features for the predictions
        all_pred_path = os.path.join(id_path,'pred')
        if  os.path.exists(all_pred_path):
            for model in os.listdir(all_pred_path):
                mask_path_short = os.path.join(id_path,'pred',model)
                self.save_feat_dict_from_paths(img_path,mask_path_short,lung_seg_path,features)

        #get the features for the segmentations
        seg_path_short = os.path.join(id_path,'seg')
        seg_path = os.path.join(id_path,'seg','001.nii.gz')
        img = torch.tensor(torchio.Image(img_path, type=torchio.INTENSITY).numpy())[0]
        seg = torch.tensor(torchio.Image(seg_path, type=torchio.LABEL).numpy())[0]
        lung_seg = torch.tensor(torchio.Image(lung_seg_path, type=torchio.LABEL).numpy())[0]

        feature_save_path = os.path.join(seg_path_short,'features.json')
        if os.path.exists(feature_save_path):
            with open(feature_save_path) as file:
                feat_dict = json.load(file)
        else:
            feat_dict = {}
        for feat in features:
            feat_dict[feat] = self.get_feature(feat,img,seg,lung_seg)

        #save the features in a json file 
        with open (feature_save_path,'w') as f:
            json.dump(feat_dict,f)
            
    def save_feat_dict_from_paths (self,img_path,mask_path_short,lung_seg_path,features):
        '''computes and saves the feature dict for a given img-pred pair 
        is a utility function for compute_features_id

        Args: 
            img_path (str): The path to the image 
            mask_path_short (str): the path to the folder containing the seg mask
                and where the feature dict is to be saved 
            lung_seg_path (str): The path to the segmentation of the lung 
            features (list(str)): a list of strings for the features to compute 
        '''
        mask_path = os.path.join(mask_path_short,'pred.nii.gz')

        #load image and mask and compute the feature dict
        img = torch.tensor(torchio.Image(img_path, type=torchio.INTENSITY).numpy())[0]
        mask = torch.tensor(torchio.Image(mask_path, type=torchio.LABEL).numpy())[0]
        lung_seg = torch.tensor(torchio.Image(lung_seg_path, type=torchio.LABEL).numpy())[0]

        feature_save_path = os.path.join(mask_path_short,'features.json')
        if os.path.exists(feature_save_path):
            with open(feature_save_path) as file:
                feat_dict = json.load(file)
        else:
            feat_dict = {}
        for feat in features:
            feat_dict[feat] = self.get_feature(feat,img,mask,lung_seg)

        #save the features in a json file 
        with open (feature_save_path,'w') as f:
            json.dump(feat_dict,f)

    def collect_train_data(self):
        '''goes through the train directory and collects all the feature vectors and labels
        
        Returns: (ndarray,ndarray) the features and labels '''
        
        if os.environ["INFERENCE_OR_TRAIN"] == 'train':
            all_features = []
            labels = []
            path = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"] ,os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR_TRAIN"])
            for id in os.listdir(path):
                all_pred_path = os.path.join(path,id,'pred')
                if os.path.exists(all_pred_path):
                    for model in os.listdir(all_pred_path):
                        pred_path = os.path.join(all_pred_path,model)
                        feature_path = os.path.join(pred_path,'features.json')
                        label_path = os.path.join(pred_path,'dice_score.json')
                        feature_vec = self.read_feature_vector(feature_path)
                        label = self.read_prediction_label(label_path)
                        #filter out any nans 
                        if np.isnan(np.sum(np.array(feature_vec))):
                            pass 
                        else:
                            all_features.append(feature_vec)
                            labels.append(label)
        else:
            print('This method is only for train time')
            RuntimeError
        return np.array(all_features),np.array(labels)

    def collect_train_data_split(self,save_as_vector=True):
        original_os_path = os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR_TRAIN"]
        os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR_TRAIN"] = os.path.join(original_os_path,'train')
        X_train,y_train =self.collect_train_data(save_as_vector)
        os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR_TRAIN"] = os.path.join(original_os_path,'test')
        X_test,y_test =self.collect_train_data(save_as_vector)
        os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR_TRAIN"] = original_os_path
        return X_train,X_test,y_train,y_test

    def read_feature_vector(self,feature_path):
        feature_vec = []
        with open(feature_path,'r') as file:
            feat_dict = json.load(file)
            for feature_name in self.features:
                feature = feat_dict[feature_name]
                if isinstance(feature,list): 
                    for entry in feature:
                        feature_vec.append(entry)
                else:
                    feature_vec.append(feature)
        return feature_vec

    def read_prediction_label(self,label_path):
        with open(label_path,'r') as file:
            label = json.load(file)
        return label

    def load_list_of_feature_vectors(self,flist):
        length = len(flist)
        name = flist[0]
        features = self.load_feature_vector(name)
        for i in range(1,length):
            name = flist[i]
            feature_vec = self.load_feature_vector(name)
            features = np.append(features,feature_vec)
        return features
            



        



  
    