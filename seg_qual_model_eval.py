import os
import math 
from mp.paths import JIP_dir
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

#set environmental variables
#for data_dirs folder, nothing changed compared to Simons version 
os.environ["WORKFLOW_DIR"] = os.path.join(JIP_dir, 'data_dirs')
os.environ["OPERATOR_IN_DIR"] = "input"
os.environ["OPERATOR_OUT_DIR"] = "output"
os.environ["OPERATOR_TEMP_DIR"] = "temp"
os.environ["OPERATOR_PERSISTENT_DIR"] = os.path.join(JIP_dir, 'data_dirs', 'persistent')

# preprocessing dir and subfolders 
os.environ["PREPROCESSED_WORKFLOW_DIR"] = os.path.join(JIP_dir, 'preprocessed_dirs')
os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR"] = "output_scaled"
os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR_TRAIN"] = "output_scaled_train"

#dir where train data for intensites is stored (this only needs to be trains_dirs, but since i have more 
# datasets, another subfolder is here)
os.environ["TRAIN_WORKFLOW_DIR"] = os.path.join(JIP_dir, 'train_dirs')

#ignore
##below is for christian only, used for older data structures where models are trained on
os.environ["TRAIN_WORKFLOW_DIR_GT"] = os.path.join('Covid-RACOON','All images and labels')
os.environ["TRAIN_WORKFLOW_DIR_PRED"] = os.path.join('Covid-RACOON','All predictions')
#os.environ["TRAIN_WORKFLOW_DIR_GT"] = os.path.join('gt_small')
#os.environ["TRAIN_WORKFLOW_DIR_PRED"] = os.path.join('pred_small')

#which mode is active either 'train' or 'inference' 
os.environ["INFERENCE_OR_TRAIN"] = 'train'

#ignore
# the ending of the image files in train_dir is only for older datasets
os.environ["INPUT_FILE_ENDING"] = 'nii.gz'

from mp.utils.preprocess_utility_functions import basic_preprocessing
from mp.utils.preprocess_utility_functions import extract_features_all_data,compute_all_prediction_dice_scores
from mp.utils.feature_extractor import Feature_extractor

# all steps in 1. are not used anymore
## 2. function to split data into splits depending on task
# 2.1 (SET splits for training and eval) set the filter function, such that the splits are as wanted 
def filter_feature_extr(id,model):
    '''function to decide which split a prediction should be put in.
    Is infered from the id(giving the dataset of the image) and model
    (the dataset the predicting model was trained on)
    Args:
        id(str):giving the dataset of the image
        model(str):the dataset the predicting model was trained on

    Returns(str):the split for the pair of id and model, can be other, which can be 
        category of unused data'''
    # # train data 
    # if model[:7] in ['Task740'] and id[:7] in ['Task740','Task541']:
    #     return 'train'
    # # id data
    # if model[:7] in ['Task740'] and id[:7] in ['Task741']:
    #     return 'id_challenge'
    # if model[:7] in ['Task740'] and id[:7] in ['Task542']:
    #     return 'id_inhouse'
    # # od data 
    # if model[:7] in ['Task740'] and id[:7] in ['Task200','Task201']:
    #     return 'ood_mosmed'

    # # train data 
    # if model[:7] in ['Task740'] and id[:7] in ['Task740','Task541']:
    #     return 'id_train'
    # # id data
    # if model[:7] in ['Task740'] and id[:7] in ['Task741','Task542']:
    #     return 'id_test'
    # # od data 
    # if model[:7] in ['Task740'] and id[:7] in ['Task200','Task201','Task100','Task101']:
    #     return 'ood_test'

    # train data 
    if model[:7] in ['Task740'] and id[:7] in ['Task740']:
        return 'train'
    if model[:7] in ['Task740'] and id[:7] in ['Task541']:
        return 'train'
    # id data
    if model[:7] in ['Task740'] and id[:7] in ['Task741']:
        return 'id_challenge'
    if model[:7] in ['Task740'] and id[:7] in ['Task542']:
        return 'id_inhouse'
    # od data 
    if model[:7] in ['Task740'] and id[:7] in ['Task200','Task201']:
        return 'ood_mosmed'
    if model[:7] in ['Task740'] and id[:7] in ['Task100','Task101']:
        return 'other'

    return 'other'

# 2.2 extract the feature
CLASS = False
def extract_features_train_id_od(filter,splits,used_feat=[0,1,2,3,4,5]):
    '''extracts the features from the preprocess dictionary, and returns a  list,
    in which each entry corresponds to the features/labels of one split
    Args:
        filter(function: (str,str)->str): should be modified filter_feature_extr
        splits(list(str)):the name of the splits
        used_feat(list(int)):the used features, see in main() for more depth

    Returns(list): list, in which each entry corresponds to the features/labels of one split'''
    X = []
    y = []
    paths_to_pred = []
    for i in range(len(splits)):
        X.append([])
        y.append([])
        paths_to_pred.append([])

    feat_extr = Feature_extractor()
    work_path = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"],os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR_TRAIN"])
    for id in os.listdir(work_path):
        all_pred_path = os.path.join(work_path,id,'pred')
        if os.path.exists(all_pred_path):
            for model in os.listdir(all_pred_path):
                split = filter(id,model)
                for i,s in enumerate(splits):
                    if split == s:
                        feature_path = os.path.join(all_pred_path,model,'features.json')
                        label_path = os.path.join(all_pred_path,model,'dice_score.json')
                        feat_vec = feat_extr.read_feature_vector(feature_path)
                        feat_vec = [feat_vec[index] for index in used_feat]
                        label = feat_extr.read_prediction_label(label_path)
                        if CLASS:
                            label = get_class(label)
                        if np.isnan(np.sum(np.array(feat_vec))) or feat_vec[0]>100000:
                            pass 
                        else:
                            X[i].append(feat_vec)
                            y[i].append(label)
                            paths_to_pred[i].append(os.path.join(all_pred_path,model))
                            
    return X,y,paths_to_pred

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

def get_float(classes):
    return np.array([(clas-1)/4 for clas in classes])
    
# 3. loss functions 
def l2_loss(pred,truth):
    n = len(pred)
    return (1/n)*(np.sum((pred - truth)**2))

def l1_loss(pred,truth, std = False):
    losses = np.absolute(pred-truth)
    mean = np.mean(losses)
    std = np.std(losses)
    if std:
        return mean,std
    return mean

def l1_loss_overestimation(pred,truth,std=False):
    '''computes the error, but underpredictions 
    are counted as correct predictions '''
    losses = pred-truth
    losses_pos = np.array([ max(loss,0) for loss in losses])
    mean = np.mean(losses_pos)
    std = np.std(losses_pos)
    if std:
        return mean,std
    return mean

def get_l1_losses(truth,pred):    
    return np.absolute(truth-pred)    
       
def confidence_bootstrap(X_train,y_train,X,y,svm,lr,scaler,splits):
    '''This is based on 
    https://machinelearningmastery.com/confidence-intervals-for-machine-learning/
    I simply resample from the training data and then evaluate the performance of the model
    on each boostrap'''
    from numpy.random import randint
    from numpy import percentile
    scores = [[] for _ in range(len(splits))] #scores for smv per run
    scores_lr = [[] for _ in range(len(splits))]# scores for lr per run
    for _ in range(10000): # i conudct 10000 runs 
        indices = randint(0,len(X_train),len(X_train)) #choose the indices in data to use 
        
        X_train_b = X_train[indices]
        y_train_b = y_train[indices]
        svm = SVC(class_weight='balanced')
        lr = LogisticRegression(class_weight='balanced')
        svm.fit(X_train_b,y_train_b) 
        lr.fit(X_train_b,y_train_b) 

        for i,_ in enumerate(splits):
        
            X_eval = scaler.transform(X[i])
            y_eval = y[i]

            y_svm = svm.predict(X_eval)
            y_lr = lr.predict(X_eval)

            conf_ma = conf_matrix(y_eval,y_svm)
            conf_ma_lr = conf_matrix(y_eval,y_lr)

            scores[i].append(specificity(conf_ma))
            scores_lr[i].append(specificity(conf_ma_lr))
    lower_p = 2.5
    upper_p = 97.5
    # Now searach for the 95 percent intervalls done like in the link in
    # the description 
    for i in range(len(splits)):
        print(splits[i])
        lower = max(0.0, percentile(scores[i], lower_p))
        print('%.1fth percentile = %.3f' % (lower_p, lower))
        upper = min(1.0, percentile(scores[i], upper_p))
        print('%.1fth percentile = %.3f' % (upper_p, upper))
    print()
    print('LR')
    for i in range(len(splits)):
        print(splits[i])
        lower = max(0.0, percentile(scores_lr[i], lower_p))
        print('%.1fth percentile = %.3f' % (lower_p, lower))
        upper = min(1.0, percentile(scores_lr[i], upper_p))
        print('%.1fth percentile = %.3f' % (upper_p, upper))

def conf_matrix(truth,pred): 
    
    conf_ma =  confusion_matrix(np.array(truth),np.array(pred),labels=[1,2,3,4,5])
    #print(conf_ma)
    ma = np.zeros((2,2))
    ma [0,0] = np.sum(conf_ma[:3,:3])
    ma [0,1] = np.sum(conf_ma[:3,3:])
    ma [1,0] = np.sum(conf_ma[3:,:3])
    ma [1,1] = np.sum(conf_ma[3:,3:])
    return ma

def sensitivity(ma):
    found = int(ma[0,0])
    not_f = int(ma[0,1])
    if found == 0:
        return 0
    else:
        return found/(found+not_f)

def p_value_sensitivity(ma):
    found = int(ma[0,0])
    not_f = int(ma[0,1])
    return find_p_p_test(found,not_f)

def specificity(ma):
    tn = int(ma[1,1])
    fp = int(ma[1,0])
    return tn/(tn+fp)

def p_value_specificity(ma):
    tn = int(ma[1,1])
    fp = int(ma[1,0])
    return find_p_p_test(tn,fp)

def p_test(p,y,n):
    '''takes a p value for the null hypothesis, the number of 
    successfull results and the number of unseccessfull results'''
    prob = 0
    combined = y + n 
    for times in np.arange(y,combined+1,1):
        term = p**times * (1-p)**(combined-times)
        bin = math.comb(combined,times)
        prob += term * bin
    return prob

def find_p_p_test(y,n):
    for p in np.arange(1,0,-0.01):
        # print(p,p_test(p,y,n))
        if p_test(p,y,n) <= 0.05:
            return p, p_test(p,y,n)

def micro_sens(conf_ma):
    conf_ma = np.array(conf_ma)
    tp = 0
    p = np.sum(conf_ma)
    for i in range(0,5):
        tp += conf_ma[i,i]
    return tp/p

def micro_spec(conf_ma):
    conf_ma = np.array(conf_ma)
    tn_cum = 0
    fp_cum = 0
    all = np.sum(conf_ma)
    for i in range(5):
        fp = np.sum(conf_ma[:,i])-conf_ma[i,i]
        tp = conf_ma[i,i]
        fn = np.sum(conf_ma[i,:])-conf_ma[i,i]
        tn = all - (fp+tp+fn)
        tn_cum += tn
        fp_cum += fp
    return tn_cum/(tn_cum+fp_cum)
     


# 4. plots for further analysis.
# (OPTIONAL SET the saving path for the bar plots)
def l1_loss_bins(pred,truth,split,by_sign=True):
    '''computes and plots a bar plot for the errors of a model, sorted by bins depending on the truth, 
    e.g. the error is taken for all predictions, where the ground trouth is in [0.5,0.6]. The errors 
    are divided into over and under estimation. Additionally the number of data points and the std 
    per bin is displayed
    Args:
        pred: the vector of predictions 
        truth: the vector of ground truths
        split(str): the name of the split
        by_sign(bool): whether the errors should be divided into over and underestimation, setting to 
            false is missing some normally displayed features'''
    save_path = os.path.join('storage','Results','Setup 5','all','Bins_{}'.format(split))
    if not by_sign:
        loss_by_bin = [[] for _ in range(10)]
        bins = np.array([])
        bins_std = np.array([])
        for i in range(10):
            for pre,tru in zip(pred,truth):
                if i*0.1 <= tru and (i+1)*0.1 > tru:
                    loss_by_bin[i].append(abs(pre-tru))
            if loss_by_bin[i]:
                bins = np.append(bins,np.mean(loss_by_bin[i]))
                bins_std = np.append(bins_std,np.std(loss_by_bin[i]))
            else:
                bins = np.append(bins,0)
                bins_std = np.append(bins_std,0)
        plt.bar(0.05+np.arange(10)*0.1,height = bins, width = 0.05,yerr = bins_std)
        plt.plot(0.05+np.arange(10)*0.1,[0.1 for i in range(10)],'-r')
        plt.title(split)
        plt.xlabel('bin (e.g. 0.1 to 0.2)')
        plt.ylabel('l1_error')
    else : 
        widths = [[] for _ in range(10)]
        loss_by_bin = [[[],[]] for _ in range(10)]
        bins = [[] for _ in range(10)]
        bins_std = [[] for _ in range(10)]
        for i in range(10):
            for pre,tru in zip(pred,truth):
                if i*0.1 <= tru and (i+1)*0.1 > tru:
                    err = pre-tru
                    if err >= 0:
                        loss_by_bin[i][1].append(err)
                    else:
                        loss_by_bin[i][0].append(err)
            for j in range(2):
                if loss_by_bin[i][j]:
                    bins[i].append(np.mean(loss_by_bin[i][j]))
                    bins_std[i].append(np.std(loss_by_bin[i][j]))
                else:
                    bins[i].append(0)
                    bins_std[i].append(0)
            unders = len(loss_by_bin[i][0])
            overs = len(loss_by_bin[i][1])
            total = unders+overs
            if total > 0:
                if unders/total > 0.66:
                    widths[i]=[0.05,0.03]
                elif overs/total >0.66:
                    widths[i]=[0.03,0.05]
                else: 
                    widths[i]=[0.05,0.05]
            else:
                widths[i]=[0.05,0.05]
        bins = np.array(bins)
        bins_std = np.array(bins_std)
        widths = np.array(widths)
        plt.bar(0.05+np.arange(10)*0.1,height = bins[:,0], width = widths[:,0],yerr = bins_std[:,0])
        plt.bar(0.05+np.arange(10)*0.1,height = bins[:,1], width = widths[:,1],yerr = bins_std[:,1])
        for i,x in enumerate(0.05+np.arange(10)*0.1):
            len_bin = len(loss_by_bin[i][0])
            if len_bin > 0:
                plt.text(x,bins[i,0],str(len_bin))
        for i,x in enumerate(0.05+np.arange(10)*0.1):
            len_bin = len(loss_by_bin[i][1])
            if len_bin > 0:
                plt.text(x,bins[i,1],str(len(loss_by_bin[i][1])))
        plt.plot(0.05+np.arange(10)*0.1,[0.1 for i in range(10)],'-r')
        plt.plot(0.05+np.arange(10)*0.1,[-0.1 for i in range(10)],'-r')
        plt.xlabel('y_true bin ')
        plt.ylabel('approximation error')
    plt.title(split)
    plt.savefig(save_path)
    plt.show()

# (OPTIONAL SET the saving path for the plots)
def plot_variable_influence(X_train,X,y,splits):
    '''for every feature in the data gives a scatter plot of all predicted segmentations,
    where the y axis is the dice score of the prediction and the x axis is the feature. The 
    different splits are also visualised. 
    Additionally trains a ridge regression, support vector regression and an MLP predictor 
    on the data, who are also displayed
    Args:
        X_train: the data to train the models on 
        X,y: The output of extract_features_train_id_od (so all features/labels sorted into
            groups by splits)
        splits(list): the name of the splits in a list
    '''
    #per variable 
    for i in range(len(X[0][0])):
        
        save_path = os.path.join('storage','Results','Setup 5','Variable {}'.format(i))
        plt.title('Feature {} vs dice'.format(i))

        # Plot the models predictions based on single features 
        X_train = [[X[0][k][i]] for k in range(len(X_train))]
        y_train = y[0]

        ridge = Ridge(normalize=False)
        svr = SVR()
        mlp = MLPRegressor((50,100,100,50))

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)

        ridge.fit(X_train_s,y_train)
        svr.fit(X_train_s,y_train)
        mlp.fit(X_train_s,y_train)

        x_min,x_max = np.min(X_train), np.max(X_train)
        x_span = x_max - x_min
        # take input range to plot model and scale it 
        points = np.reshape(np.arange(x_min,x_max,step=x_span/100),(-1,1))
        points_s = scaler.transform(points)

        y_ridge = ridge.predict(points_s)
        y_svr = svr.predict(points_s)
        y_mlp = mlp.predict(points_s)
               
        plt.plot(points,y_ridge, label='ridge')
        plt.plot(points,y_svr, label='svr')
        plt.plot(points,y_mlp, label='mlp')

        #plot the data points
        for j,split in enumerate(splits):
            filt_X = [[X[j][k][i]] for k in range(len(X[j]))]
            plt.scatter(filt_X,y[j],label=split)

        plt.legend(loc='upper left')
        plt.xlabel('feature')
        plt.ylabel('dice score of prediction')
        #plt.savefig(save_path)
        plt.show()

def paper_figures_by_split(X_train,X,y,splits):
    #per variable 
    #plt.rcParams.update({'font.size': 20})
    
    for i in range(len(X[0][0])):
        
        title,leg_loc,xlabel,legend_prop = figure_namesand_settings(i)
        save_path = os.path.join('storage','Results','for_paper','Variable {} _lesslabels'.format(i))
        # plt.title(title)

        for j,split in enumerate(splits):
            filt_X = [[X[j][k][i]] for k in range(len(X[j]))]
            plt.scatter(filt_X,y[j],label=split)
        if i == 0:
            pass #plt.legend(loc=leg_loc,prop = legend_prop)
        #plt.xlabel(xlabel)
        #plt.ylabel('Dice Score of segmentation')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        #plt.savefig(save_path)
        plt.show()

def figure_namesand_settings(feat_nr):
    if feat_nr == 0:
        title = 'Segmentation smoothness vs Dice Score prediction'
        leg_loc = 'upper left'
        xlabel = 'Segmentation smoothness'
        legend_prop = {'size':15}
    if feat_nr == 1:
        title = 'Number Connected Components vs Dice Score prediction'
        leg_loc = 'upper right'
        xlabel = 'Number Connected Components'
        legend_prop = {'size':8}
    if feat_nr == 2:
        title = 'Intensity Mode vs Dice Score prediction'
        leg_loc = 'upper right'
        xlabel = 'Intensity Mode'
        legend_prop = {'size':8}
    return title,leg_loc,xlabel,legend_prop
        
def find_ex_pred(X,y,paths):
    paths_ex = []
    seen_paths = ['storage\\JIP\\preprocessed_dirs\\output_scaled_train\\Task100_RadiopediaTrain_0007\\pred\\Task740_ChallengeTrainF4',
                    'storage\\JIP\\preprocessed_dirs\\output_scaled_train\\Task200_MosmedTrain_0027\\pred\\Task740_ChallengeTrainF4',
                    'storage\\JIP\\preprocessed_dirs\\output_scaled_train\\Task200_MosmedTrain_0013\\pred\\Task740_ChallengeTrainF4',
                    'storage\\JIP\\preprocessed_dirs\\output_scaled_train\\Task100_RadiopediaTrain_0008\\pred\\Task541_FrankfurtTrainF4']
    for i in range(len(X[0][0])):
        thresh = 0.4
        if i == 0:
            smallest_val = 1
            smallest_ind = [0,0]
            for j in range(len(X)):
                filt_X = np.array([[X[j][k][i]] for k in range(len(X[j]))])
                for k in range(len(filt_X)):
                    if filt_X[k] < smallest_val and y[j][k] <= thresh and paths[j][k] not in seen_paths: 
                        smallest_val = filt_X[k]
                        smallest_ind = [j,k]
            paths_ex.append(paths[smallest_ind[0]][smallest_ind[1]])
            print(smallest_val)
            print(y[smallest_ind[0]][smallest_ind[1]])
        if i == 1:
            smallest_val = 1
            smallest_ind = [0,0]
            for j in range(len(X)):
                filt_X = np.array([[X[j][k][i]] for k in range(len(X[j]))])
                for k in range(len(filt_X)):
                    if filt_X[k] > smallest_val and y[j][k] <= thresh and paths[j][k] not in paths_ex and paths[j][k] not in seen_paths:
                        smallest_val = filt_X[k]
                        smallest_ind = [j,k]
            paths_ex.append(paths[smallest_ind[0]][smallest_ind[1]])
            print(smallest_val)
            print(y[smallest_ind[0]][smallest_ind[1]])
        if i == 2:
            smallest_val = 0
            smallest_ind = [0,0]
            for j in range(len(X)):
                filt_X = np.array([[X[j][k][i]] for k in range(len(X[j]))])
                for k in range(len(filt_X)):
                    if filt_X[k] > smallest_val and y[j][k] <= thresh and paths[j][k] not in paths_ex and paths[j][k] not in seen_paths:
                        smallest_val = filt_X[k]
                        smallest_ind = [j,k]
            paths_ex.append(paths[smallest_ind[0]][smallest_ind[1]])
            print(smallest_val)
            print(y[smallest_ind[0]][smallest_ind[1]])
        if i == 3: 
            smallest_val = 1
            smallest_ind = [0,0]
            for j in range(len(X)):
                filt_X = np.array([[X[j][k][i]] for k in range(len(X[j]))])
                for k in range(len(filt_X)):
                    if filt_X[k] < smallest_val and y[j][k] <= thresh and paths[j][k] not in seen_paths: 
                        smallest_val = filt_X[k]
                        smallest_ind = [j,k]
            paths_ex.append(paths[smallest_ind[0]][smallest_ind[1]])
            print(smallest_val)
            print(y[smallest_ind[0]][smallest_ind[1]])
    print(paths_ex)
    return paths_ex

def main(used_feat=[0,1,2,3,4,5],preprocessing=True,feature_extraction=True,
            extract_dice_scores=True,model_train=True,label=1):
    '''
    Args:
        used_feat(list(ints)): a list of ints to tell which features to use for model training:
                                1: density_distance
                                2: slice dice (smoothnes of segmentation)
                                3: slice dice diff (basically the first derivation of 2 (not really used)
                                4: connected components
                                5: mean of a gaussian fitted to intensity values
                                6: variance of a gaussain fitted to intenstiy values
        preprocessing(bool): whether to copy data from a directory into the working directoy
            includes things such as resizing and scaling of images
        train_density(bool): whether to retrain the density for intensity values (time consuming)
        feature_extraction(bool): weather to extract features from the data, not needed, when features.json 
            are already in the working directory(in e.g. subfolder seg for an id)
        extract_dice_scores(bool): whether to extract the dice scores of the predictions, not needed
            when there are already dice_score.json in the working directory(in e.g. subfolder seg for an id)
        model_train(bool): whether to train and evaluate the models '''
    if preprocessing:
        basic_preprocessing(label)
    if feature_extraction:
        extract_features_all_data()
    if extract_dice_scores:
        compute_all_prediction_dice_scores()
    if model_train:

        scaler = StandardScaler()
        splits = ['train','id_challenge','id_inhouse','ood_mosmed'] #['train_gc','train_fra','gc_gc','gc_frank','gc_mosmed','gc_radio']['id_train','id_test','ood_test']['train','gc_gc','gc_frank','gc_mosmed','gc_radio']['id_train','id_test','ood_test']['other'] 
        X,y,paths_pred = extract_features_train_id_od(filter_feature_extr,splits,used_feat)

        X_train = scaler.fit_transform(X[0])
        y_train = np.array(y[0])
        # #START of experiments with classification task

# #################conf_ma class#################################################################################################################################        
#         svm = SVC(class_weight='balanced')
#         lr = LogisticRegression(class_weight='balanced')
#         models = [svm,lr]
#         model_names = ['SVM','Logistic_Regression']

#         for model, model_name in zip(models,model_names):
#             model.fit(X_train,y_train)
#             for i,split in enumerate(splits):

#                 X_eval = scaler.transform(X[i])
#                 y_eval = y[i]

#                 plot_confusion_matrix(model,X_eval,y_eval,labels=[1,2,3,4,5])
#                 plt.title('Model {} on split {}'.format(model_name,split))
#                 plt.savefig(os.path.join('storage','Results','confusion_matrixes',model_name+' '+split))
#                 plt.show()
#                 # y_pred = model.predict(X_eval)

#                 # conf_matrix = confusion_matrix(y_eval,y_pred,labels=[1,2,3,4,5])
#                 # if i != 0:
#                 #     print(model_name,split)
#                 #     print(conf_matrix)
# #################conf_ma class#################################################################################################################################  

# ##################3 feat models lr################################################################################################################################
#         lr = LogisticRegression(class_weight='balanced')
#         lr.fit(X_train,y_train)
#         for i,split in enumerate(splits):
#             X_eval = scaler.transform(X[i])
#             y_eval = y[i]
#             y_pred = lr.predict(X_eval)
#             ma = conf_matrix(y_eval,y_pred)
#             sens = sensitivity(ma)
#             spec = specificity(ma)
#             print(split)
#             print(sens, spec)
# ##################3 feat models lr################################################################################################################################
        # # confidence_bootstrap(X_train,y_train,X,y,svm,lr,scaler,splits)
        # model = svm
        # model.fit(X_train,[get_class(tru) for tru in y_train])
        # print(model)
        # for i,split in enumerate(splits):
        
        #     X_eval = scaler.transform(X[i])
        #     y_eval = y[i]
        #     y_gt = y_eval
        #     y_eval = np.array([get_class(tru) for tru in y_eval])

        #     y_pred =model.predict(X_eval)
        #     y_pred = get_float(y_pred)
            
        #     print(split)

        #     print('err :{}'.format(l1_loss(y_gt,y_pred,std=False)))
        #     print('overest errr :{}'.format(l1_loss_overestimation(y_gt,y_pred)))
        # model = lr
        # model.fit(X_train,y_train)
        # print(model)
        # for i,split in enumerate(splits):
        
        #     X_eval = scaler.transform(X[i])
        #     y_eval = y[i]

        #     y_pred =model.predict(X_eval)
        #     print(split)
        #     conf_ma = confusion_matrix(y_eval,y_pred,labels=[1,2,3,4,5])
        #     # print('sensitivity :{}'.format(micro_sens(conf_ma)))
        #     print('specificity :{}'.format(micro_spec(conf_ma)))
        # print('oberved, p-values')
        # print('LR')
        # lr.fit(X_train,y_train)
        # for i,split in enumerate(splits):
        
        #     X_eval = scaler.transform(X[i])
        #     y_eval = y[i]

        #     y_lr =lr.predict(X_eval)
        #     print(split)
        #     ma = conf_matrix(y_eval,y_lr)
        #     # print(ma)
        #     print('sensitivity :{},  {}'.format(sensitivity(ma), p_value_sensitivity(ma)))
        #     print('specificity :{},  {}'.format(specificity(ma), p_value_specificity(ma)))
        # print()
        # print('SVM')
        # svm.fit(X_train,y_train)
        # for i,split in enumerate(splits):
        
        #     X_eval = scaler.transform(X[i])
        #     y_eval = y[i]

        #     y_svm =svm.predict(X_eval)
        #     print(split)
        #     ma = conf_matrix(y_eval,y_svm)
        #     #print(ma)
        #     print('sensitivity :{},  {}'.format(sensitivity(ma), p_value_sensitivity(ma)))
        #     print('specificity :{},  {}'.format(specificity(ma), p_value_specificity(ma)))
        ##  END of classification tests 

        # #START of part for regression task
        # plot_variable_influence(X_train,X,y,splits)
        # paper_figures_by_split(X_train,X,y,splits)
        # find_ex_pred(X,y,paths_pred)

############conf_ma regression####################################################################################################       
        from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay

        ridge = Ridge(normalize=False)
        svr = SVR()
        mlp = MLPRegressor((50,100,100,50))
        models = [ridge,svr,mlp]
        model_names = ['RidgeRegression','SVR','MLP']

        for model, model_name in zip(models,model_names):
            model.fit(X_train,y_train)
            for i,split in enumerate(splits):
               
                X_eval = scaler.transform(X[i])
                y_eval = y[i]

                y_pred_cont = model.predict(X_eval)

                #In order to get a confusion matrix, we target and prediction values need to be binned:
                y_eval = [get_class(truth) for truth in y_eval]
                y_pred = [get_class(pred) for pred in y_pred_cont]
                conf_matrix = confusion_matrix(y_eval,y_pred,labels=[1,2,3,4,5])
                print(conf_matrix)
                disp = ConfusionMatrixDisplay(conf_matrix,display_labels=[1,2,3,4,5])
                disp.plot()
                plt.title('Model {} on split {}'.format(model_name,split))
                plt.savefig(os.path.join('storage','Results','confusion_matrixes',model_name+' '+split))
                plt.show()
                # print(model_name,split)
                # print(conf_matrix)
# ###################################################################################################################################
        

        # ridge.fit(X_train,y_train)
        # svr.fit(X_train,y_train)
        # mlp.fit(X_train,y_train)
        # for i,split in enumerate(splits):
        #     print()
        #     print(split)
        #     X_eval = scaler.transform(X[i])
        #     y_eval = y[i]
        #     print(split,'{:.3f} {:.3f} '.format(np.mean(y_eval),np.std(y_eval)))

            # y_ridge = ridge.predict(X_eval)
            # y_svr = svr.predict(X_eval)
            # y_mlp = mlp.predict(X_eval)
            
            # # # sens_r = sensitivity(conf_matrix([get_class(truth) for truth in y_eval],[get_class(pred) for pred in y_ridge]))
            # # # sens_s = sensitivity(conf_matrix([get_class(truth) for truth in y_eval],[get_class(pred) for pred in y_svr]))
            # # # sens_m = sensitivity(conf_matrix([get_class(truth) for truth in y_eval],[get_class(pred) for pred in y_mlp]))

            # print('minimal pred values {},{},{}'.format(np.min(y_ridge),np.min(y_svr),np.min(y_mlp)))
            # # ma_r = confusion_matrix([get_class(truth) for truth in y_eval],[get_class(pred) for pred in y_ridge],labels=[1,2,3,4,5])
            # # ma_s = confusion_matrix([get_class(truth) for truth in y_eval],[get_class(pred) for pred in y_svr],labels=[1,2,3,4,5])
            # # ma_m = confusion_matrix([get_class(truth) for truth in y_eval],[get_class(pred) for pred in y_mlp],labels=[1,2,3,4,5])

            # # # for matrix, model in zip([sens_r,sens_s,sens_m],['ridge','svr','mlp']):
            # # #     print(model)
            # # #     print('sensitivity :{}'.format(matrix))
                


        #     ridge_err,ridge_std = l1_loss(y_ridge,y_eval, std = True)
        #     svr_err, svr_std  = l1_loss(y_svr,y_eval,std = True)
        #     mlp_err, mlp_std = l1_loss(y_mlp,y_eval,std = True)

        #     ridge_err_over,ridge_std_over = l1_loss_overestimation(y_ridge,y_eval, std = True)
        #     svr_err_over, svr_std_over  = l1_loss_overestimation(y_svr,y_eval,std = True)
        #     mlp_err_over, mlp_std_over = l1_loss_overestimation(y_mlp,y_eval,std = True)

        #     # l1_loss_bins(y_svr,y_eval,split)

        #     for i,std in enumerate([ridge_std,svr_std,mlp_std]):
        #         stds_of_splits[i].append(std)
        #     for i,std in enumerate([ridge_std_over,svr_std_over,mlp_std_over]):
        #         stds_of_splits_over[i].append(std)
        #     for i,errors in enumerate([get_l1_losses(y_ridge,y_eval),get_l1_losses(y_svr,y_eval),get_l1_losses(y_mlp,y_eval)]):
        #         for err in errors:  
        #             all_errors[i].append(err)
        #     for i in range(3):
        #         all_errors_over[i] = [max(err,0) for err in all_errors[i]]

        #     #a vector that predicts the mean value of y_train, is a baseline
        #     # y_mean = np.mean(y_train)*np.ones(np.shape(y_eval))
        #     print('{}    :        ridge               svr             mlp  '.format(split))
        #     print(u"error          {:.3f} \u00B1 {:.3f},   {:.3f} \u00B1 {:.3f},  {:.3f} \u00B1 {:.3f}".format(ridge_err,ridge_std,svr_err,svr_std,mlp_err,mlp_std))
        #     print(u"error overest. {:.3f} \u00B1 {:.3f},   {:.3f} \u00B1 {:.3f},  {:.3f} \u00B1 {:.3f}".format(ridge_err_over,ridge_std_over,svr_err_over,svr_std_over,mlp_err_over,mlp_std_over))

        #     # print('Using mean of train values, has error of {} and std of {}'.format(l1_loss(y_mean,y_train,std=True)[0],l1_loss(y_mean,y_train,std=True)[1]))
        #     print()

        # print('TOTAL:                       ridge     svr      mlp  ')
        # print('mean of split stds          {:.3f},   {:.3f},  {:.3f}'.format(np.mean(stds_of_splits[0]),np.mean(stds_of_splits[1]),np.mean(stds_of_splits[2])))
        # print('std of all errors           {:.3f},   {:.3f},  {:.3f}'.format(np.std(all_errors[0]),np.std(all_errors[1]),np.std(all_errors[2])))
        # print('mean of split stds overest. {:.3f},   {:.3f},  {:.3f}'.format(np.mean(stds_of_splits_over[0]),np.mean(stds_of_splits_over[1]),np.mean(stds_of_splits_over[2])))
        # print('std of all errors overest.  {:.3f},   {:.3f},  {:.3f}'.format(np.std(all_errors_over[0]),np.std(all_errors_over[1]),np.std(all_errors_over[2])))
        # # END regression task

if __name__ == "__main__":
    #(SET all params, depending on desired train procedurey)
    main([0,1,2,3],preprocessing=False,feature_extraction=False, extract_dice_scores=False,model_train=True)


## Examples for images to show 
# for dices : copies: img:  Task100_RadiopediaTrain_0009 pred:  Task740_ChallengeTrainF4'
# for int mode : not lung tissue: img : Task200_MosmedTrain_0033 pred : Task740_ChallengeTrainF4'
# for conn comp : too muh and many: img : Task100_RadiopediaTrain_0008 pred : Task541_FrankfurtTrainF4'
#                                or img : Task100_RadiopediaTrain_0007 pred : Task541_FrankfurtTrainF4'
# seg in lung 