# Import needed libraries
import torch
import os
import shutil
import traceback
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from mp.data.data import Data
from mp.data.datasets.dataset_JIP_cnn import JIPDataset
from mp.experiments.data_splitting import split_dataset
import mp.utils.load_restore as lr
from mp.data.pytorch.pytorch_cnn_dataset import Pytorch3DQueue
from mp.models.cnn.cnn import CNN_Net3D
from mp.models.cnn.mobilenet_v2 import MobileNetV2
from mp.models.cnn.densenet import Densenet121
from mp.eval.losses.losses_cnn import LossCEL
from mp.agents.cnn_agents import NetAgent
from mp.utils.save_results import save_results, save_only_test_results
from mp.quantifiers.NoiseQualityQuantifier import NoiseQualityQuantifier

def train_model(config):
    r"""This function tries to intializes and trains a model.
        It returns True if the model was sucessfully trained and if not an error will be returned as well."""
    try:
        _CNN_initialize_and_train(config)
        return True, None
    except: # catch *all* exceptions
        e = traceback.format_exc()
        return False, e


def restore_train_model(config):
    r"""This function tries to restore and trains a model. 
        It returns True if the model was sucessfully trained and if not an error will be returned as well."""
    try:
        _CNN_restore_and_train(config)
        return True, None
    except: # catch *all* exceptions
        e = traceback.format_exc()
        return False, e


def retrain_model(config):
    r"""This function retrains a model (using transfer learning).
        It returns True if the model was sucessfully retrained and if not an error will be returned as well."""
    try:
        _CNN_retrain(config)
        return True, None
    except: # catch *all* exceptions
        e = traceback.format_exc()
        return False, e


def test_model(config):
    r"""This function tries to load a pre-trained model and tests it on the test dataset (ID or OOD).
        It returns True if the model was sucessfully tested and if not an error will be returned as well."""
    try:
        _CNN_test(config)
        return True, None
    except: # catch *all* exceptions
        e = traceback.format_exc()
        return False, e


def do_inference(config):
    r"""This function uses a pretrained model and performs inference on the dataset. It returns True
        if the inference was sucessful and if not an error will be returned as well."""
    try:
        _CNN_predict(config)
        return True, None
    except: # catch *all* exceptions
        e = traceback.format_exc()
        return False, e


# -------------------------------
# Train-Restore-Retrain approach
# -------------------------------
def _CNN_initialize_and_train(config):
    r"""This function selects random images etc. based on the config file
        and starts training the model. If everything works fine, without
        and error, the results will be saved."""

    # 1. Retrieve information from config dict
    device = config['device']
    device_name = torch.cuda.get_device_name(device)
    print('Device name: {}'.format(device_name))
    output_features = config['num_intensities']
    dataset_name = 'Task'

    # 2. Define data
    data = Data()
    JIP = JIPDataset(img_size=config['input_shape'], num_intensities=config['num_intensities'], data_type=config['data_type'],\
                     augmentation=config['augmentation'], data_augmented=config['data_augmented'], gpu=True, cuda=config['device'],\
                     msg_bot = config['msg_bot'], nr_images=config['nr_images'], build_dataset=True, dtype='train', noise=config['noise'],\
                     ds_name=dataset_name, restore=config['restore'])

    data.add_dataset(JIP)
    train_ds = (dataset_name, 'train')
    val_ds = (dataset_name, 'val')
    test_ds = (dataset_name, 'test')

    # 3. Split data and define path
    splits = dict()
    for ds_name, ds in data.datasets.items():
        splits[ds_name] = split_dataset(ds, test_ratio = config['test_ratio'], 
                          val_ratio = config['val_ratio'], nr_repetitions = 1, cross_validation = False)
    paths = os.path.join(os.environ["TRAIN_WORKFLOW_DIR"], os.environ["OPERATOR_OUT_DIR"], config['noise'], 'states')
    pathr = os.path.join(os.environ["TRAIN_WORKFLOW_DIR"], os.environ["OPERATOR_OUT_DIR"], config['noise'], 'results')
    if not os.path.exists(paths):
        os.makedirs(paths)
    else:
        # Empty directory
        shutil.rmtree(paths)
        os.makedirs(paths)
    if not os.path.exists(pathr):
        os.makedirs(pathr)
    else:
        # Empty directory
        shutil.rmtree(pathr)
        os.makedirs(pathr)

    # Save split
    if splits is not None:
        lr.save_json(splits, path = paths, name = 'data_splits')

    # 4. Create data splits for each repetition
    print('Bring data to PyTorch format..')

    # 5. Bring data to Pytorch format
    datasets = dict()
    for ds_name, ds in data.datasets.items():
        for split, data_ixs in splits[ds_name][0].items():
            if len(data_ixs) > 0: # Sometimes val indices may be an empty list
                aug = config['augment_strat'] if not('test' in split) else 'none'
                datasets[(ds_name, split)] = Pytorch3DQueue(ds, 
                    ix_lst = data_ixs, size = (1, 35, 51, 35), aug_key = aug, 
                    samples_per_volume = 15)

    # 6. Build train dataloader
    dl = DataLoader(datasets[(train_ds)], 
        batch_size = config['batch_size'], shuffle = True)
    dl_val = DataLoader(datasets[(val_ds)], 
        batch_size = config['batch_size'], shuffle = True)

    # 7. Initialize model
    if config['noise'] == 'spike':
        model = Densenet121()
    else:
        model = MobileNetV2()
    model.to(device)

    # 8. Define loss and optimizer
    loss_f = LossCEL(device = device)
    optimizer = optim.Adam(model.parameters(), lr = config['lr'],
                            weight_decay = config['weight_decay'])

    # 9. Train model
    print('Training model in batches of {}..'.format(config['batch_size']))

    agent = NetAgent(model = model, device = device, lr_decay = config['lr_decay'])
    losses_train, losses_cum_train, losses_val, losses_cum_val,\
    accuracy_train, accuracy_det_train, accuracy_val,\
    accuracy_det_val = agent.train(optimizer, loss_f, dl,
      dl_val, config['decay_rate'], config['decay_type'],
      nr_epochs = config['nr_epochs'], save_path = paths,
                 save_interval = config['save_interval'],
                             msg_bot = config['msg_bot'],
           bot_msg_interval = config['bot_msg_interval'],
                       store_data = config['store_data'])
                        
    # 10. Build test dataloader
    dl = DataLoader(datasets[(test_ds)],
            batch_size = config['batch_size'], shuffle = True)
    
    # 11. Test model
    print('Testing model in batches of {}..'.format(config['batch_size']))
    losses_test, losses_cum_test, accuracy_test, accuracy_det_test, y_hat_table = agent.test(loss_f, dl, msg_bot = config['msg_bot'])

    # 12. Save results
    save_results(model, config['noise'], paths, pathr, losses_train, losses_val, accuracy_train,
                 accuracy_det_train, accuracy_val, accuracy_det_val, losses_test, accuracy_test,
                 accuracy_det_test, losses_cum_train, losses_cum_val, y_hat_table)
    

def _CNN_restore_and_train(config):
    r"""This function loads an existing state based on the config file, trains
        the missing epochs and saves the result."""

    # 1. Retrieve information from config dict
    device = config['device']
    device_name = torch.cuda.get_device_name(device)
    print('Device name: {}'.format(device_name))
    output_features = config['num_intensities']
    dataset_name = ds_name

    # 2. Define data to restore dataset
    data = Data()
    JIP = JIPDataset(img_size=config['input_shape'], num_intensities=config['num_intensities'], data_type=config['data_type'],\
                     augmentation=config['augmentation'], data_augmented=config['data_augmented'], gpu=True, cuda=config['device'],\
                     msg_bot = config['msg_bot'], nr_images=config['nr_images'], build_dataset=True, dtype='train', noise=config['noise'],\
                     ds_name=dataset_name, restore=config['restore'])

    data.add_dataset(JIP)
    train_ds = (dataset_name, 'train')
    val_ds = (dataset_name, 'val')
    test_ds = (dataset_name, 'test')

    # 3. Restore and define path
    paths = os.path.join(os.environ["TRAIN_WORKFLOW_DIR"], os.environ["OPERATOR_OUT_DIR"], config['noise'], 'states')
    pathr = os.path.join(os.environ["TRAIN_WORKFLOW_DIR"], os.environ["OPERATOR_OUT_DIR"], config['noise'], 'results')
    splits = lr.load_json(path=paths, name='data_splits')
    print('Restored existing splits')

    # 4. Create data splits for each repetition
    print('Bring data to PyTorch format..')
    
    # 5. Bring data to Pytorch format
    datasets = dict()
    for ds_name, ds in data.datasets.items():
        for split, data_ixs in splits[ds_name][0].items():
            if len(data_ixs) > 0: # Sometimes val indicess may be an empty list
                aug = config['augment_strat'] if not('test' in split) else 'none'
                datasets[(ds_name, split)] = Pytorch3DQueue(ds, 
                    ix_lst = data_ixs, size = (1, 35, 51, 35), aug_key = aug, 
                    samples_per_volume = 15)
    
    # 6. Build train dataloader
    dl = DataLoader(datasets[(train_ds)], 
        batch_size = config['batch_size'], shuffle = True)
    dl_val = DataLoader(datasets[(val_ds)], 
        batch_size = config['batch_size'], shuffle = True)
    
    # 7. Initialize model
    model = CNN_Net3D(output_features) 
    if config['noise'] == 'spike':
        model = Densenet121()
    else:
        model = MobileNetV2()
    model.to(device)

    # 8. Define loss and optimizer
    loss_f = LossCEL(device = device)
    optimizer = optim.Adam(model.parameters(), lr = config['lr'],
                            weight_decay = config['weight_decay'])

    # 9. Train model
    state_names = [name for name in os.listdir(paths) if '.' not in name]
    state_name = state_names[0].split('_')[0]
    for idx, state in enumerate(state_names):
        state_names[idx] = int(state.split('_')[-1])
    state_names.sort()
    state_name += '_' + str(state_names[-1])

    print('Restore last saved model from epoch {}..'.format(state_name.split('_')[-1]))
    agent = NetAgent(model = model, device = device, lr_decay = config['lr_decay'])
    restored, restored_results = agent.restore_state(paths, state_name, optimizer = optimizer)
    if not restored:
        print("Desired state could not be recovered. --> Error!")
        raise FileNotFoundError

    if not config['store_data']:
        for idx, i in enumerate(restored_results):
            if i is None:
                restored_results[idx] = np.array([])

    losses_train_r, losses_cum_train_r, losses_val_r, losses_cum_val_r, accuracy_train_r,\
    accuracy_det_train_r, accuracy_val_r, accuracy_det_val_r = restored_results
    
    print('Training model in batches of {}..'.format(config['batch_size']))
    losses_train, losses_cum_train, losses_val, losses_cum_val,\
    accuracy_train, accuracy_det_train, accuracy_val,\
    accuracy_det_val = agent.train(optimizer, loss_f, dl,
       dl_val, config['decay_rate'], config['decay_type'],
                            nr_epochs=config['nr_epochs'],
             start_epoch = int(state_name.split('_')[-1]),
      save_path = paths, losses = losses_train_r.tolist(),
                 losses_cum = losses_cum_train_r.tolist(),
                       losses_val = losses_val_r.tolist(),
               losses_cum_val = losses_cum_val_r.tolist(),
                     accuracy = accuracy_train_r.tolist(),
        accuracy_detailed = accuracy_det_train_r.tolist(),
                   accuracy_val = accuracy_val_r.tolist(),
      accuracy_val_detailed = accuracy_det_val_r.tolist(),
                  save_interval = config['save_interval'],
                              msg_bot = config['msg_bot'],
            bot_msg_interval = config['bot_msg_interval'],
                        store_data = config['store_data'])
    
    # 10. Build test dataloader
    dl = DataLoader(datasets[(test_ds)],
            batch_size = config['batch_size'], shuffle = True)
    
    # 11. Test model
    print('Testing model in batches of {}..'.format(config['batch_size']))
    losses_test, _, accuracy_test, accuracy_det_test, y_hat_table = agent.test(loss_f, dl, msg_bot = config['msg_bot'], store_data = config['store_data'])

    # 12. Save results
    save_results(model, config['noise'], paths, pathr, losses_train, losses_val, accuracy_train,
                 accuracy_det_train, accuracy_val, accuracy_det_val, losses_test, accuracy_test,
                 accuracy_det_test, losses_cum_train, losses_cum_val, y_hat_table)


def _CNN_retrain(config):
    r"""This function loads a pre-trained model from presistent given the noise
        and retrains this model using tranfer learning."""

    # 1. Retrieve information from config dict
    device = config['device']
    device_name = torch.cuda.get_device_name(device)
    print('Device name: {}'.format(device_name))
    output_features = config['num_intensities']
    dataset_name = ds_name

    # 2. Define data --> Extra in JIP_dataset that loads everything from preprocessed for train!
    data = Data()
    JIP = JIPDataset(img_size=config['input_shape'], num_intensities=config['num_intensities'], data_type=config['data_type'],\
                     augmentation=config['augmentation'], data_augmented=config['data_augmented'], gpu=True, cuda=config['device'],\
                     msg_bot = config['msg_bot'], nr_images=config['nr_images'], build_dataset=True, dtype='train', noise=config['noise'],\
                     ds_name=dataset_name, restore=config['restore'])

    data.add_dataset(JIP)
    train_ds = (dataset_name, 'train')
    val_ds = (dataset_name, 'val')
    test_ds = (dataset_name, 'test')

    # 3. Split data and define path
    splits = dict()
    for ds_name, ds in data.datasets.items():
        splits[ds_name] = split_dataset(ds, test_ratio = config['test_ratio'], 
                          val_ratio = config['val_ratio'], nr_repetitions = 1, cross_validation = False)
    paths = os.path.join(os.environ["TRAIN_WORKFLOW_DIR"], os.environ["OPERATOR_OUT_DIR"], config['noise'], 'states')
    pathr = os.path.join(os.environ["TRAIN_WORKFLOW_DIR"], os.environ["OPERATOR_OUT_DIR"], config['noise'], 'results')
    if not os.path.exists(paths):
        os.makedirs(paths)
    else:
        # Empty directory
        shutil.rmtree(paths)
        os.makedirs(paths)
    if not os.path.exists(pathr):
        os.makedirs(pathr)
    else:
        # Empty directory
        shutil.rmtree(pathr)
        os.makedirs(pathr)

    # Save split
    if splits is not None:
        lr.save_json(splits, path = paths, name = 'data_splits')

    # 4. Create data splits for each repetition
    print('Bring data to PyTorch format..')

    # 5. Bring data to Pytorch format
    datasets = dict()
    for ds_name, ds in data.datasets.items():
        for split, data_ixs in splits[ds_name][0].items():
            if len(data_ixs) > 0: # Sometimes val indices may be an empty list
                aug = config['augment_strat'] if not('test' in split) else 'none'
                datasets[(ds_name, split)] = Pytorch3DQueue(ds, 
                    ix_lst = data_ixs, size = (1, 35, 51, 35), aug_key = aug, 
                    samples_per_volume = 15)

    # 6. Build train dataloader
    dl = DataLoader(datasets[(train_ds)], 
        batch_size = config['batch_size'], shuffle = True)
    dl_val = DataLoader(datasets[(val_ds)], 
        batch_size = config['batch_size'], shuffle = True)

    # 7. Load pre-trained model
    if config['noise'] == 'spike':
        model = Densenet121()
    else:
        model = MobileNetV2()
    
    state_dict = torch.load(os.path.join(os.environ["OPERATOR_PERSISTENT_DIR"], config['noise'], 'model_state_dict.zip'))
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    # 8. Define loss and optimizer
    loss_f = LossCEL(device = device)
    optimizer = optim.Adam(model.parameters(), lr = config['lr'],
                            weight_decay = config['weight_decay'])

    # 9. Train model
    print('Training model in batches of {}..'.format(config['batch_size']))

    agent = NetAgent(model = model, device = device, lr_decay = config['lr_decay'])
    losses_train, losses_cum_train, losses_val, losses_cum_val,\
    accuracy_train, accuracy_det_train, accuracy_val,\
    accuracy_det_val = agent.train(optimizer, loss_f, dl,
      dl_val, config['decay_rate'], config['decay_type'],
      nr_epochs = config['nr_epochs'], save_path = paths,
                 save_interval = config['save_interval'],
                             msg_bot = config['msg_bot'],
           bot_msg_interval = config['bot_msg_interval'],
                       store_data = config['store_data'])
                        
    # 10. Build test dataloader
    dl = DataLoader(datasets[(test_ds)],
            batch_size = config['batch_size'], shuffle = True)
    
    # 11. Test model
    print('Testing model in batches of {}..'.format(config['batch_size']))
    losses_test, _, accuracy_test, accuracy_det_test, y_hat_table = agent.test(loss_f, dl, msg_bot = config['msg_bot'], store_data = config['store_data'])

    # 12. Save results
    save_results(model, config['noise'], paths, pathr, losses_train, losses_val, accuracy_train,
                 accuracy_det_train, accuracy_val, accuracy_det_val, losses_test, accuracy_test,
                 accuracy_det_test, losses_cum_train, losses_cum_val, y_hat_table)


# -------------------------------
# Test-Inference approach
# -------------------------------
def _CNN_test(config):
    r"""This function loads an existing (pre-trained) model and makes predictions based on the test dataset.
     The test will always be performed with a batch_size of 1."""

    # 1. Retrieve information from config dict
    device = config['device']
    device_name = torch.cuda.get_device_name(device)
    print('Device name: {}'.format(device_name))
    dataset_name = config['dataset_name']

    # 2. Define data --> Extra in JIP_dataset that loads everything from preprocessed for train!
    data = Data()
    JIP = JIPDataset(img_size=config['input_shape'], num_intensities=config['num_intensities'], data_type=config['data_type'],\
                     augmentation=config['augmentation'], data_augmented=config['data_augmented'], gpu=True, cuda=config['device'],\
                     msg_bot = config['msg_bot'], nr_images=config['nr_images'], build_dataset=True, dtype=config['mode'], noise=config['noise'],\
                     ds_name=dataset_name, restore=config['restore'])
    data.add_dataset(JIP)
    test_ds = (dataset_name, 'test')

    # 3. Split data and define path
    splits = dict()
    for ds_name, ds in data.datasets.items():
        splits[ds_name] = split_dataset(ds, test_ratio = 1, val_ratio = 0, nr_repetitions = 1, cross_validation = False)
    pathr = os.path.join(os.environ["TEST_WORKFLOW_DIR"], os.environ["OPERATOR_OUT_DIR"], config['noise'], config['mode']+'_results')
    if not os.path.exists(pathr):
        os.makedirs(pathr)
    else:
        # Empty directory
        shutil.rmtree(pathr)
        os.makedirs(pathr)

    # 4. Bring data to Pytorch format
    print('Bring data to PyTorch format..')

    datasets = dict()
    for ds_name, ds in data.datasets.items():
        for split, data_ixs in splits[ds_name][0].items():
            if len(data_ixs) > 0: # Sometimes val indices may be an empty list
                aug = config['augment_strat'] if not('test' in split) else 'none'
                datasets[(ds_name, split)] = Pytorch3DQueue(ds, 
                    ix_lst = data_ixs, size = (1, 35, 51, 35), aug_key = aug, 
                    samples_per_volume = 15)

    # 5. Build test dataloader
    dl = DataLoader(datasets[(test_ds)], batch_size = 1, shuffle = True)

    # 6. Load pre-trained model
    path_m = os.path.join(os.environ["OPERATOR_PERSISTENT_DIR"], config['noise'], 'model_state_dict.zip')
    if config['noise'] == 'spike':
        model = lr.load_model('Densenet121', path_m, True)
    else:
        model = lr.load_model('MobileNetV2', path_m, True)
    model.to(device)

    # 7. Define Loss and Agent
    loss_f = LossCEL(device = device)
    agent = NetAgent(model = model, device = device, lr_decay = config['lr_decay'])

    # 8. Test model
    print('Testing model in batches of 1..')
    losses_test, _, accuracy_test, accuracy_det_test, y_hat_table = agent.test(loss_f, dl, msg_bot = config['msg_bot'], store_data = config['store_data'])
    
    # 9. Save results
    save_only_test_results(pathr, losses_test, accuracy_test, accuracy_det_test)


def _CNN_predict(config):
    r"""This function loads an existing (pre-trained) model and makes predictions based on the input file(s). The predictions are made for fft and ~fft images."""

    metrices = dict()
    data_dataset_path = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"], os.environ["PREPROCESSED_OPERATOR_OUT_DATA_DIR"])
    study_names = [x for x in os.listdir(data_dataset_path) if '._' not in x]
    #Initialize the NoiseQualityQuantifier for all artefacts
    NQQ_noise = NoiseQualityQuantifier(device=config['device'], artefact_given = 'noise', output_features=config['num_intensities'])
    NQQ_blur = NoiseQualityQuantifier(device=config['device'], artefact_given = 'blur', output_features=config['num_intensities'])
    NQQ_motion = NoiseQualityQuantifier(device=config['device'], artefact_given = 'motion', output_features=config['num_intensities'])
    NQQ_ghosting = NoiseQualityQuantifier(device=config['device'], artefact_given = 'ghosting', output_features=config['num_intensities'])
    NQQ_spike = NoiseQualityQuantifier(device=config['device'], artefact_given = 'spike', output_features=config['num_intensities'])

    for i in range(len(study_names)):
        # 1. Load data
        data = Data()
        data_fft = Data()

        #One dataset for ~fft and one for fft
        JIP = JIPDataset(img_size=config['input_shape'], num_intensities=config['num_intensities'], data_type=config['data_type'],\
                        augmentation=config['augmentation'], data_augmented=config['data_augmented'], gpu=True, cuda=config['device'],\
                        msg_bot = config['msg_bot'], nr_images=config['nr_images'], build_dataset=True, dtype='inference', noise=config['noise'],\
                        ds_name='Task', restore=config['restore'], inference_name = study_names[i], fft_for_inference = False) #NBTN JIPDataset mit einem Inferencebild
        JIP_fft = JIPDataset(img_size=config['input_shape'], num_intensities=config['num_intensities'], data_type=config['data_type'],\
                        augmentation=config['augmentation'], data_augmented=config['data_augmented'], gpu=True, cuda=config['device'],\
                        msg_bot = config['msg_bot'], nr_images=config['nr_images'], build_dataset=True, dtype='inference', noise=config['noise'],\
                        ds_name='Task', restore=config['restore'], inference_name = study_names[i], fft_for_inference = True) #NBTN JIPDataset mit einem Inferencebild
        
        data.add_dataset(JIP)
        data_fft.add_dataset(JIP_fft)
        inference_ds = (config['dataset_name'], 'test')
        # 2. Split data and define path, for fft and ~fft
        splits = dict()
        for ds_name, ds in data.datasets.items():
            splits[ds_name] = split_dataset(ds, test_ratio = 1, val_ratio = 0, nr_repetitions = 1, cross_validation = False)


        splits_fft = dict()
        for ds_name, ds in data_fft.datasets.items():
            splits_fft[ds_name] = split_dataset(ds, test_ratio = 1, val_ratio = 0, nr_repetitions = 1, cross_validation = False)
            
        # 3. Bring data to Pytorch format
        print('Bring data to PyTorch format..')
    
        datasets = dict()
        for ds_name, ds in data.datasets.items():
            for split, data_ixs in splits[ds_name][0].items():
                if len(data_ixs) > 0: # Sometimes val indices may be an empty list
                    aug = config['augment_strat'] if not('test' in split) else 'none'
                    datasets[(ds_name, split)] = Pytorch3DQueue(ds, 
                        ix_lst = data_ixs, size = (1, 35, 51, 35), aug_key = aug, 
                        samples_per_volume = 1) 

        datasets_fft = dict()
        for ds_name, ds in data_fft.datasets.items():
            for split, data_ixs in splits_fft[ds_name][0].items():
                if len(data_ixs) > 0: # Sometimes val indices may be an empty list
                    aug = config['augment_strat'] if not('test' in split) else 'none'
                    datasets_fft[(ds_name, split)] = Pytorch3DQueue(ds, 
                        ix_lst = data_ixs, size = (1, 35, 51, 35), aug_key = aug, 
                        samples_per_volume = 1) 


        #4. Initialize dataloader
        dl = DataLoader(datasets[(inference_ds)], batch_size = 1, shuffle = False)
        dl_fft = DataLoader(datasets_fft[(inference_ds)], batch_size = 1, shuffle = False)

        # 5. Calculate metrices, depending on wether the model was trained with fft images or with ~fft images, fft/~fft images are used for the prediction
        metrices_x = dict()
        file = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"], os.environ["PREPROCESSED_OPERATOR_OUT_DATA_DIR"], study_names[i], 'img', 'img.nii.gz')
        for num, (x,_) in enumerate(dl):
            
            msg = "Loading SimpleITK images and calculating metrices (doing inference): "
            print (msg, end = "\r")
            metrices_x['HFC'], metrices_x['noise'] = NQQ_noise.get_quality(x=x, file=file, device=config['device'])  
            _, metrices_x['blur'] = NQQ_blur.get_quality(x=x, file=file, device=config['device'])  

        for num, (x,_) in enumerate(dl_fft):
            #metrices_x = dict()
            msg = "Loading SimpleITK images and calculating metrices for fft (doing inference): "
            print (msg, end = "\r")
            _, metrices_x['motion'] = NQQ_motion.get_quality(x=x, file=file, device=config['device'])  
            _, metrices_x['ghosting'] = NQQ_ghosting.get_quality(x=x, file=file, device=config['device'])  
            _, metrices_x['spike'] = NQQ_spike.get_quality(x=x, file=file, device=config['device'])  

        metrices[study_names[i]] = metrices_x

    # 6. Save metrices as json
    out_dir = os.path.join('/', os.environ['WORKFLOW_DIR'], os.environ["OPERATOR_OUT_DIR"])
    lr.save_json_beautiful(metrices, out_dir, 'metrices', True)