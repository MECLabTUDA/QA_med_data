import os
import json

def generate_train_labels(num_intensities, source_path, target_path, swap_labels=True, ds_name = "Task"): #NBTN swap_label False--> true
    r"""This function generates the labels.json file that is necessary for training."""
    # Foldernames are patient_id
    print('genetrate_train_label')

    filenames = [x for x in os.listdir(source_path) if '._' not in x and ds_name in x\
                 and not 'blur' in x and not 'resolution' in x and not 'ghosting' in x and not 'motion' in x\
                 and not 'noise' in x and not 'spike' in x] 

    # Generate labels  with augmentation
    labels = dict()
    for name in filenames:
        labels[str(name)] = 5/num_intensities
        labels[str(name) + '_blur5'] = 5/num_intensities
        labels[str(name) + '_blur4'] = 4/num_intensities
        labels[str(name) + '_blur3'] = 3/num_intensities
        labels[str(name) + '_blur2'] = 2/num_intensities
        labels[str(name) + '_blur1'] = 1/num_intensities
        labels[str(name) + '_resolution5'] = 5/num_intensities
        labels[str(name) + '_resolution4'] = 4/num_intensities
        labels[str(name) + '_resolution3'] = 3/num_intensities
        labels[str(name) + '_resolution2'] = 2/num_intensities
        labels[str(name) + '_resolution1'] = 1/num_intensities
        labels[str(name) + '_ghosting5'] = 5/num_intensities
        labels[str(name) + '_ghosting4'] = 4/num_intensities
        labels[str(name) + '_ghosting3'] = 3/num_intensities
        labels[str(name) + '_ghosting2'] = 2/num_intensities
        labels[str(name) + '_ghosting1'] = 1/num_intensities
        labels[str(name) + '_motion5'] = 5/num_intensities
        labels[str(name) + '_motion4'] = 4/num_intensities
        labels[str(name) + '_motion3'] = 3/num_intensities
        labels[str(name) + '_motion2'] = 2/num_intensities
        labels[str(name) + '_motion1'] = 1/num_intensities
        labels[str(name) + '_noise5'] = 5/num_intensities
        labels[str(name) + '_noise4'] = 4/num_intensities
        labels[str(name) + '_noise3'] = 3/num_intensities
        labels[str(name) + '_noise2'] = 2/num_intensities
        labels[str(name) + '_noise1'] = 1/num_intensities
        labels[str(name) + '_spike5'] = 5/num_intensities
        labels[str(name) + '_spike4'] = 4/num_intensities
        labels[str(name) + '_spike3'] = 3/num_intensities
        labels[str(name) + '_spike2'] = 2/num_intensities
        labels[str(name) + '_spike1'] = 1/num_intensities

    
    # Save labels
    print("Saving generated labels..")
    if not os.path.isdir(target_path):
        os.makedirs(target_path)
    with open(os.path.join(target_path, 'labels.json'), 'w') as fp:
        json.dump(labels, fp, sort_keys=True, indent=4)


    # Transform labels in such a way: k:v --> v_artefact:[k] if desired
    if swap_labels:
        labels_swapped = dict()
        augmentationT = ['blur', 'noise', 'ghosting', 'spike', 'resolution', 'motion']
        intensities = [1/num_intensities, 2/num_intensities, 3/num_intensities, 4/num_intensities, 5/num_intensities]

        # Loop through labels and change k:v to v_artefact:[k]
        for k, v in labels.items():
            intensity = str(v)
            augmentation = ''.join([i for i in str(k.split('_')[-1]) if not i.isdigit()])
            key = str(intensity+'_'+augmentation)
            if key == '1.0_':   # Decathlon Data with not augmentation --> perfekt in all augmentations
                for a in augmentationT:
                    a_key = key+str(a)
                    if a_key in labels_swapped:
                        v_list = labels_swapped[a_key]
                        v_list.append(k)
                        labels_swapped[a_key] = v_list
                    else:
                        labels_swapped[a_key] = [k]
            elif key in labels_swapped:
                v_list = labels_swapped[key]
                v_list.append(k)
                labels_swapped[key] = v_list
            else:
                labels_swapped[key] = [k]

        # Add all missing v_artefacts with empty lists --> v_artefact:[]
        for i in intensities:
            for a in augmentationT:
                key = str(i)+'_'+str(a)
                if key not in labels_swapped:
                    labels_swapped[key] = list()

        # Save labels
        print("Saving swapped labels..")
        with open(os.path.join(target_path, 'labels_swapped.json'), 'w') as fp:
            json.dump(labels_swapped, fp, sort_keys=True, indent=4)


def generate_test_labels(num_intensities, source_path, target_path, ds_name ="Task"):
    r"""This function generates the labels.json file that is necessary for testing on an unseen dataset."""
    # Foldernames are patient_id
    filenames = [x for x in os.listdir(source_path) if 'DS_Store' not in x and ds_name in x\
                 and not 'blur' in x and not 'resolution' in x and not 'ghosting' in x and not 'motion' in x\
                 and not 'noise' in x and not 'spike' in x] 

    # Generate labels with augmentation
    labels = dict()
    # Save labels
    print("Saving generated labels..")
    if not os.path.isdir(target_path):
        os.makedirs(target_path)
    with open(os.path.join(target_path, 'labels.json'), 'w') as fp:
        json.dump(labels, fp, sort_keys=True, indent=4)