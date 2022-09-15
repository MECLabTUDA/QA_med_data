# Import needed libraries
import torch
import os
import numpy as np
import pandas as pd
import shutil
from mp.visualization.plot_results import plot_dataframe
import seaborn as sb

def save_results(model, noise, paths, pathr, losses_train, losses_val, accuracy_train,
                 accuracy_det_train, accuracy_val, accuracy_det_val, losses_test, accuracy_test,
                 accuracy_det_test, losses_cum_train, losses_cum_val, y_hat_table):
    r"""This function saves the results from a trained model, i.e. losses and accuracies."""
    print('Save trained model losses and accuracies..')
    torch.save(model.state_dict(), os.path.join(paths, 'model_state_dict.zip'))
    model_path = os.path.join(os.environ["OPERATOR_PERSISTENT_DIR"], noise)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    else:
        # Empty directory
        shutil.rmtree(model_path)
        os.makedirs(model_path)
    torch.save(model.state_dict(), os.path.join(model_path, 'model_state_dict.zip'))
    
    #Values for the confusion matrix
    with open(os.path.join(pathr, 'y_hat_table.txt'), 'w') as f: 
        print(y_hat_table, file=f)
    #Values for the accuracy
    with open(os.path.join(pathr, 'accuracy.txt'), 'w') as f: 
        print('\n accuracy train \n', file=f)
        print(accuracy_train, file=f)
        print('\n accuracy det train \n', file=f)
        print(accuracy_det_train, file=f)
        print('\n accuracy val \n', file=f)
        print(accuracy_val, file=f)
        print('\n accuracy det val \n', file=f)
        print(accuracy_det_train, file=f)
        print('\n accuracy test \n', file=f)
        print(accuracy_test, file=f)
        print('\n accuracy det test \n', file=f)
        print(accuracy_det_train, file=f)

    


    if losses_train is not None:
        np.save(os.path.join(pathr, 'losses_train.npy'), np.array(losses_train))
    np.save(os.path.join(pathr, 'losses_cum_train.npy'), np.array(losses_cum_train))
    if losses_val is not None:
        np.save(os.path.join(pathr, 'losses_validation.npy'), np.array(losses_val))
    np.save(os.path.join(pathr, 'losses_cum_validation.npy'), np.array(losses_cum_val))
    np.save(os.path.join(pathr, 'accuracy_train.npy'), np.array(accuracy_train))
    if accuracy_det_train is not None:
        np.save(os.path.join(pathr, 'accuracy_detailed_train.npy'), np.array(accuracy_det_train))
    np.save(os.path.join(pathr, 'accuracy_validation.npy'), np.array(accuracy_val))
    if accuracy_det_val is not None:
        np.save(os.path.join(pathr, 'accuracy_detailed_validation.npy'), np.array(accuracy_det_val))
    np.save(os.path.join(pathr, 'losses_test.npy'), np.array(losses_test))
    np.save(os.path.join(pathr, 'accuracy_test.npy'), np.array(accuracy_test))
    if accuracy_det_test is not None:
        np.save(os.path.join(pathr, 'accuracy_detailed_test.npy'), np.array(accuracy_det_test))
    plot_dataframe(pd.DataFrame(losses_cum_train, columns = ['Epoch', 'Loss']),
        save_path = pathr, save_name = 'losses_train', title = 'Losses [train dataset]',
        x_name = 'Epoch', y_name = 'Loss', ending = '.png', figsize = (10,5),
        xints = float, yints = float)
    plot_dataframe(pd.DataFrame(accuracy_train, columns = ['Epoch', 'Accuracy']),
        save_path = pathr, save_name = 'accuracy_train', title = 'Accuracy [train dataset] in %',
        x_name = 'Epoch', y_name = 'Accuracy', ending = '.png', figsize = (10,5),
        xints = float, yints = float)
    plot_dataframe(pd.DataFrame(losses_cum_val, columns = ['Epoch', 'Loss']),
        save_path = pathr, save_name = 'losses_val', title = 'Losses [validation dataset]',
        x_name = 'Epoch', y_name = 'Loss', ending = '.png', figsize = (10,5),
        xints = float, yints = float)
    plot_dataframe(pd.DataFrame(accuracy_val, columns = ['Epoch', 'Accuracy']),
        save_path = pathr, save_name = 'accuracy_val', title = 'Accuracy [validation dataset] in %',
        x_name = 'Epoch', y_name = 'Accuracy', ending = '.png', figsize = (10,5),
        xints = float, yints = float)
    plot_dataframe(pd.DataFrame(losses_test, columns = ['Batch', 'Loss']),
        save_path = pathr, save_name = 'losses_test', title = 'Losses [test dataset]',
        x_name = 'Batch', y_name = 'Loss', ending = '.png', figsize = (10,5),
        xints = float, yints = float)
    plot_dataframe(pd.DataFrame(accuracy_test, columns = ['Batch', 'Accuracy']),
        save_path = pathr, save_name = 'accuracy_test', title = 'Accuracy [test dataset] in %',
        x_name = 'Batch', y_name = 'Accuracy', ending = '.png', figsize = (10,5),
        xints = int, yints = int)

    #Heatmap for confusion matrix
    X_val = ['No', 'Very weak', 'Weak', 'Medium', 'Strong']
    y_val = ['No', 'Very weak', 'Weak', 'Medium', 'Strong']
    heat_map = sb.heatmap(np.array(y_hat_table), annot=True, xticklabels=X_val, yticklabels=y_val, cmap='Blues_r')
    heat_map.set_xlabel('Predicted Label')
    heat_map.set_ylabel('True Label')
    heat_map.set_title(str.capitalize(noise))
    figure = heat_map.get_figure()
    figure.savefig(os.path.join(pathr, 'y_hat_table.png'))



def save_only_test_results(path, losses_test, accuracy_test, accuracy_det_test):
    r"""This function saves the test results from a pretrained model, i.e. losses and accuracies."""
    print('Save pre-trained model losses and accuracies on test dataset..')
    np.save(os.path.join(path, 'losses_test.npy'), np.array(losses_test))
    np.save(os.path.join(path, 'accuracy_test.npy'), np.array(accuracy_test))
    if accuracy_det_test is not None:
        np.save(os.path.join(path, 'accuracy_detailed_test.npy'), np.array(accuracy_det_test))
    plot_dataframe(pd.DataFrame(losses_test, columns = ['Batch', 'Loss']),
        save_path = path, save_name = 'losses_test', title = 'Losses [test dataset]',
        x_name = 'Batch', y_name = 'Loss', ending = '.png', figsize = (10,5),
        xints = float, yints = float)
    plot_dataframe(pd.DataFrame(accuracy_test, columns = ['Batch', 'Accuracy']),
        save_path = path, save_name = 'accuracy_test', title = 'Accuracy [test dataset] in %',
        x_name = 'Batch', y_name = 'Accuracy', ending = '.png', figsize = (10,5),
        xints = int, yints = int)
