import os
import json 
from mp.quantifiers.QualityQuantifier import SegImgQualityQuantifier
from mp.models.classification.dice_predictor import Dice_predictor
from mp.utils.feature_extractor import Feature_extractor
from mp.utils.preprocess_utility_functions import basic_preprocessing, extract_features_all_data


class IntBasedQuantifier(SegImgQualityQuantifier):

    def __init__(self, label=1, version='0.0'):
        super().__init__(version)

        self.label = label

        self.feat_extr = Feature_extractor()
        self.dice_pred = Dice_predictor(label=label,version='standart')
        self.dice_pred.load()

        self.work_path = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"],os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR"])
        self.output_path = os.path.join(os.environ["WORKFLOW_DIR"],os.environ["OPERATOR_OUT_DIR"],'segmentation_quality_metrics.json')

    def preprocess_data(self):
        '''copies the data from the input dir into preprocessed scaled dir 
        and extracts the features of the data'''
        os.environ["INFERENCE_OR_TRAIN"] = 'inference'
        basic_preprocessing(label=self.label)
        extract_features_all_data()

    def get_quality(self):
        r"""Get quality values for a segmentation mask, optionally together with
        an image, according to one or more metrics. This method should be
        overwritten.

        Returns (dict[str -> dict[str -> float]]): a dictionary that contains a dict with metrices for every id given
        """
        self.preprocess_data()
        output_dict=dict()
        for id in os.listdir(self.work_path):
            id_dict = dict()
            path_to_features = os.path.join(self.work_path,id,'seg','features.json')
            feature_vec = self.feat_extr.read_feature_vector(path_to_features)
            prediction = self.dice_pred.predict([feature_vec])[0]
            id_dict['dice_pred']=prediction
            output_dict[id]=id_dict
        print(output_dict)
        
        with open(self.output_path,'w') as file:
            json.dump(output_dict,file)


        