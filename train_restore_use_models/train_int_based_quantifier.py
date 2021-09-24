import os
import numpy as np
from mp.utils.intensities import get_intensities, load_intensities
from mp.models.classification.dice_predictor import Dice_predictor
from mp.utils.feature_extractor import Feature_extractor
from mp.utils.preprocess_utility_functions import basic_preprocessing
from mp.utils.preprocess_utility_functions import extract_features_all_data,compute_all_prediction_dice_scores

def train_int_based_quantifier(preprocess=True,train_dice_pred=True,verbose=False,label=1):
    '''The function to call in order to train the quality quantifier model
    Args:
        preprocess(bool) : Whether the data needs to be preprocessed (copied, resized, etc)
        train_dice_pred(bool): Whether the dice predictor should be trained
        verbose(bool): whether output shiud be printed 
        label(int): The label of the segmented tissue which is of interest
        '''

    if os.environ["INFERENCE_OR_TRAIN"] == 'train':
        if preprocess:
            if verbose:
                print('Preprocessing data ...')
            basic_preprocessing(label=label) 
        if train_dice_pred:
            if verbose:
                print('Training dice predictor ...')
            extract_features_all_data()
            compute_all_prediction_dice_scores()
            train_dice_predictor(verbose=verbose,label=label)
    else : 
        print('This is only for Train mode')

def train_dice_predictor(model_name='standart',feature_extractor=None,data_describtion = 'all of train data',
                            model_describtion = 'LR',verbose=False,label=1,**kwargs):
    '''Trains a dice predictor model based on features extracted from image-seg pairs

    Args:
        model_name (str): The name of the model
        feature_extractor (instance of Feature_extractor class):The names of the features to be used
        data_describtion (str): a describtion of the used data
        model_describtion (str): a describtion of the used model and density model
        verbose(bool): whether the model shall print output
        label (int): the label to train on. Is used to identify which model to use 
        **kwargs :keyworded arguemtent which can be used to alter the specifics of the model to be trained 
    '''
    if not feature_extractor:
        feature_extractor = Feature_extractor()

    #initiate model
    dice_pred = Dice_predictor(features=feature_extractor.features,version=model_name,verbose=verbose,label=label)

    #Load the features 
    X_train,y_train = feature_extractor.collect_train_data()

    #train model
    dice_pred.train(X_train,y_train, data_describtion, model_describtion,**kwargs)
    
    if verbose:
        dice_pred.print_description()




