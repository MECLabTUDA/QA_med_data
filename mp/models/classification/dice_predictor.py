import os 
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def get_class(label):
    if label < 0.2:
        return 1
    elif label < 0.4:
        return 2 
    elif label < 0.6:
        return 3
    elif label < 0.8:
        return 4
    else:
        return 5

class Dice_predictor():
    '''a simple Logistic regression model ,that predicts a dice score from a feature vector
    '''
    def __init__(self, features = [], version='', verbose=False, label=1):
        
        self.label = label
        self.model = None
        self.features = features
        
        path = os.path.join(os.environ['OPERATOR_PERSISTENT_DIR'],'dice_predictors')
        if not os.path.isdir(path):
            os.makedirs(path)

        self.scaler = None

        self.path_to_scaler = os.path.join(path,'label_{}_{}_scaler.sav'.format(self.label,version))
        self.path_to_model = os.path.join(path,
                    'label_{}_{}.sav'.format(self.label,version)) 
        self.path_to_model_descr = os.path.join(path,
                    'label_{}_{}_descr.txt'.format(self.label,version))
        self.verbose = verbose


    def load(self):
        '''loads the model with given name'''
        try:
            self.model = pickle.load(open(self.path_to_model,'rb'))
            self.scaler = pickle.load(open(self.path_to_scaler,'rb'))
        except:
            print('there is no model with this name, please it train first before loading')
            raise RuntimeError
        if self.verbose:
            self.print_description()

    def train(self,X_train,y_train,  data_descr='', model_descr='',
                **kwargs):
        '''notice that the labels have to be binned first, which is done here as well '''

        if self.verbose:
            print('training model')

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        if kwargs:
            self.model = LogisticRegression(**kwargs)
        else:
            self.model = LogisticRegression(class_weight='balanced')
        
        # bin the labels 
        y_train = [get_class(label) for label in y_train]

        self.model.fit(X_train_scaled,y_train)

        model_score = self.model.score(X_train_scaled,y_train)
        
        losses_string = 'The model has an accuracy of {} in train data'.format(model_score)

        if self.verbose:
            print(losses_string)

        with open(self.path_to_model,'wb') as saver:
            pickle.dump(self.model,saver)
        
        with open(self.path_to_scaler,'wb') as saver: 
            pickle.dump(self.scaler,saver)

        self._save_descr(data_descr,model_descr,losses_string,**kwargs)

    def predict(self, input):
        '''predicts dice scores, for given feature matrix''' 
        input = self.scaler.transform(input)
        output = self.model.predict(input)
        #since there are 5 classes and we want to map to [0,1] we need to divide by 5 
        output = [pred/5 for pred in output]
        return output

    def _save_descr(self,data_d,model_d,losses_string,**kwargs):
        with open(self.path_to_model_descr,'w') as file:
            file.write("Data describtion: \n")
            file.write(data_d)
            file.write("\n")
            file.write("Used features: \n")
            file.write('{}'.format(self.features))
            file.write("\n")
            file.write("Model describtion: \n")
            file.write(model_d)
            file.write("\n")
            file.write("train parameter : \n")
            file.write('{}'.format(kwargs))
            file.write("\n")
            file.write(losses_string)

    def print_description(self):
        with open(self.path_to_model_descr,'r') as file:
            for line in file:
                print(line)