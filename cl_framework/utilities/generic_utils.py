import os
import random
import string
import shutil
import json
import random
import numpy as np
import torch
import pandas as pd

from copy import deepcopy


def seed_everything(seed=0):
    """Fix all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True


def experiment_folder(root_path, dev_mode, approach_name):
    if os.path.exists(os.path.join(root_path, 'exp_folder')):
        shutil.rmtree(os.path.join(root_path, 'exp_folder'), ignore_errors=True)

    if dev_mode:
        exp_folder = 'exp_folder'
    else:
        exp_folder = approach_name + '_' + ''.join(random.choices(string.ascii_letters + string.digits, k=8))

    out_path = os.path.join(root_path, exp_folder)
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    return out_path, exp_folder

def get_class_per_task(n_class_first_task, total_classes, n_task):
    assert n_class_first_task <= total_classes
    
    if n_class_first_task > -1:
        class_per_task = int((total_classes - n_class_first_task)/(n_task - 1))
        assert class_per_task > 1 
        assert n_class_first_task + (n_task - 1) * class_per_task  == total_classes
    else:
        class_per_task =  int(total_classes/ n_task)   
        assert class_per_task > 1 
        assert n_task * class_per_task == total_classes

    return class_per_task

def result_folder(out_path, name):
    if not os.path.exists(os.path.join(out_path, name)):
        os.mkdir(os.path.join(out_path, name))


def store_params(out_path, n_epoch, bs, n_task, old_reconstruction, loss_weight):
    params = {}
    params['n_epoch'] = n_epoch
    params['bs'] = bs
    params['n_task'] = n_task
    params['old_reconstruction'] = old_reconstruction
    params['loss_weight'] = loss_weight
    store_dictionary(params, out_path, name='params')


def get_task_dict(n_task, total_classes, class_per_task, n_class_first_task):
    d = {}
    l = list(range(total_classes))
    
    if n_class_first_task > - 1:
        offset = n_class_first_task
        for i in range(n_task):
            if i == 0:
                d[i] = [i for i in range(0, n_class_first_task)]
            else:
                d[i] = l[offset + (i-1)*class_per_task: offset+ (i-1)* class_per_task + class_per_task]     
            
    else:
      
        for i in range(n_task):
            d[i] = l[i*class_per_task:  i* class_per_task + class_per_task]     
               
    return d 



            


def store_dictionary(d, out_path, name):
    d={str(k): v for k, v in d.items()}
    with open(os.path.join(out_path, name+'.json'), 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=4)


def rollback_model(approach, out_path, device, name=None):
    if name is not None:
        approach.model.load_state_dict(torch.load(out_path, map_location=device))
        print("Model Loaded {}".format(out_path))
    else:
        approach.model.load_state_dict(torch.load(os.path.join(out_path,'_model.pth'), map_location=device))


def store_model(approach, out_path, name=""):
    torch.save(deepcopy(approach.model.state_dict()), os.path.join(out_path, name+"_model.pth"))


def remap_targets(train_set, test_set, total_classes):
    l = list(range(total_classes))
    l_sorted = deepcopy(l)
    random.shuffle(l)
    label_mapping = dict(zip(l_sorted, l))
    
    # remap train labels following label_mapping    
    
    for i in range(len(train_set.targets)):
        train_set.targets[i]=label_mapping[train_set.targets[i]]
    
 
    for key in train_set.class_to_idx.keys():
        train_set.class_to_idx[key] = label_mapping[train_set.class_to_idx[key]]
        
    # remap test labels following label_mapping    
    
    for i in range(len(test_set.targets)):
        test_set.targets[i]=label_mapping[test_set.targets[i]]
    
 
    for key in test_set.class_to_idx.keys():
        test_set.class_to_idx[key] = label_mapping[test_set.class_to_idx[key]]
        
        
     
    return train_set, test_set, label_mapping

def store_valid_loader(out_path, valid_loaders, store):
    if store:
        for i, loader in enumerate(valid_loaders):
            torch.save(loader, os.path.join(out_path, 'dataloader_'+str(i)+'.pth'))




#TODO to remove, right now everything i need to use is in get_task_dict_incdec
""" def get_behaviors_per_task(total_classes, n_task=6, pipeline='baseline', behaviors_to_be_removed_csv_path = None):

    if pipeline == 'baseline':
        behaviors_to_change = 0
    elif pipeline == 'decremental':
        pass


    return behaviors_to_change """


def get_task_dict_incdec(n_task, total_classes, behaviors_csv_path, pipeline, behaviors_randomize, out_path):
    #TODO: this starting_data_dict should be returned from an external file maybe
    starting_data_dict = {
    'food': [
        'eating burger', 'eating cake', 'eating carrots', 'eating chips', 'eating doughnuts',
        'eating hotdog', 'eating ice cream', 'eating spaghetti', 'eating watermelon',
        'sucking lolly', 'tasting beer', 'tasting food', 'tasting wine', 'sipping cup'
    ],
    'phone': [
        'texting', 'talking on cell phone', 'looking at phone'
    ],
    'smoking': [
        'smoking', 'smoking hookah', 'smoking pipe'
    ],
    'fatigue': [
        'sleeping', 'yawning', 'headbanging', 'headbutting', 'shaking head'
    ],
    'selfcare': [
        'scrubbing face', 'putting in contact lenses', 'putting on eyeliner', 'putting on foundation',
        'putting on lipstick', 'putting on mascara', 'brushing hair', 'brushing teeth', 'braiding hair',
        'combing hair', 'dyeing eyebrows', 'dyeing hair'
    ]
    }

    d = {}

    behaviors_dicts = []
       
    if pipeline == 'baseline':
        for i in range(n_task):
            d[i] = (len(starting_data_dict.keys()))
            behaviors_dicts.append(starting_data_dict)
    elif pipeline == 'decremental':
        
        result_folder(out_path, 'behaviors_task')

        save_path = os.path.join(out_path, 'behaviors_task')

        behaviors_to_remove = pd.read_csv(behaviors_csv_path)
        # i deepcopy just to be sure that i do not change the starting data dict if i need to reuse it
        current_behaviors_dict = deepcopy(starting_data_dict)

        # shuffling columns to do a randomized order of decremental behaviors
        # randomize all but the first row, which should not have any behaviors removed
        if behaviors_randomize == 'yes':
            for column in current_behaviors_dict:
                behaviors_to_remove[column][1:] = np.random.permutation(behaviors_to_remove[column].values[1:])

        for i, row in behaviors_to_remove.iterrows():
            d[i] = (len(starting_data_dict.keys()))

            # iterate the classes to remove them
            for idx_class in current_behaviors_dict.keys():
                # get how many behaviors to remove
                count_to_remove = row[idx_class]
                # remove the behaviors if != 0
                if count_to_remove != 0:
                    for count_idx in range(count_to_remove):
                        # get current behaviors of that class from the current dict of behaviors
                        behaviors_count = len(current_behaviors_dict[idx_class])
                        # now select randomly the index of the behavior to be removed
                        idx_to_remove = random.randint(0,behaviors_count-1)
                        # remove the item from the list
                        del current_behaviors_dict[idx_class][idx_to_remove]
            # add the new dict to the behaviors dicts
            behaviors_dicts.append(deepcopy(current_behaviors_dict))
            csv_behavior = pd.DataFrame.from_dict(current_behaviors_dict, orient='index')
            csv_behavior.to_csv(os.path.join(save_path,'task_{}'.format(i)),header=False,index=False)
    elif pipeline == 'incremental_decremental':
        result_folder(out_path, 'behaviors_task')

        save_path = os.path.join(out_path, 'behaviors_task')

        behaviors_to_substitute = pd.read_csv(behaviors_csv_path)
        # get a subset of behaviors to start with, as described in the first row of the csv
        tmp_behaviors_dict = deepcopy(starting_data_dict)

        if behaviors_randomize == 'yes':
            for column in tmp_behaviors_dict:
                behaviors_to_substitute[column][1:] = np.random.permutation(behaviors_to_substitute[column].values[1:])

        first_task_behaviors = behaviors_to_substitute.iloc[0]

        current_behaviors_dict = {}

        # get the behaviors for the first task
        for idx_class in tmp_behaviors_dict.keys():
                # get how many behaviors to remove
                count_to_get = first_task_behaviors[idx_class]
                current_behaviors_dict[idx_class] = []
                if count_to_get != 0:
                    for count_idx in range(count_to_get):
                        # get current behaviors of that class from the current dict of behaviors
                        behaviors_count = len(tmp_behaviors_dict[idx_class])
                        # now select randomly the index of the behavior to be added to the first task
                        idx_to_add = random.randint(0,behaviors_count-1)
                        current_behaviors_dict[idx_class].append(tmp_behaviors_dict[idx_class][idx_to_add])
                        del tmp_behaviors_dict[idx_class][idx_to_add]
        d[0] = (len(starting_data_dict.keys()))
        # add the first dict to the behaviors dictionaries
        behaviors_dicts.append(deepcopy(current_behaviors_dict))
        # dict with all the behaviors from the first task to be removed
        behaviors_to_remove = deepcopy(current_behaviors_dict)

        for i, row in behaviors_to_substitute.iterrows():
            # skip the first task
            if i != 0:
                d[i] = (len(starting_data_dict.keys()))

                # iterate the classes to remove them
                for idx_class in tmp_behaviors_dict.keys():
                    count_to_get = row[idx_class]
                    if count_to_get != 0:
                        for count_idx in range(count_to_get):
                            behaviors_count_remaining_to_add = len(tmp_behaviors_dict[idx_class])
                            # now select randomly the index of the behavior to be added to the task
                            idx_to_add = random.randint(0,behaviors_count_remaining_to_add-1)

                            behaviors_count_remaining_to_remove = len(behaviors_to_remove[idx_class])
                            # now select randomly the index of the behavior to be removed to the task
                            idx_to_remove = random.randint(0,behaviors_count_remaining_to_remove-1)
                            current_behaviors_dict[idx_class].append(tmp_behaviors_dict[idx_class][idx_to_add])
                            del tmp_behaviors_dict[idx_class][idx_to_add]
                            current_behaviors_dict[idx_class].remove(behaviors_to_remove[idx_class][idx_to_remove])
                            del behaviors_to_remove[idx_class][idx_to_remove]
                behaviors_dicts.append(deepcopy(current_behaviors_dict))
                csv_behavior = pd.DataFrame.from_dict(current_behaviors_dict, orient='index')
                csv_behavior.to_csv(os.path.join(save_path,'task_{}'.format(i)),header=False,index=False)

    return d, behaviors_dicts

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count