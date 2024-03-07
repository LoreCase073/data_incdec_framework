#imports to work with...
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import DataLoader
import torch
import torchvision
from torchvision import transforms
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize

from cl_framework.continual_learning.metrics.metric_evaluator_incdec import MetricEvaluatorIncDec
from cl_framework.utilities.matrix_logger import IncDecLogger
from torchmetrics import Recall


""" results_path =  """
results_path = ['../runs_trainings/no_freeze/multilabel/weighted',
                '../runs_trainings/no_freeze/decremental_multilabel/weighted',         
                '../runs_trainings/no_freeze/incremental_decremental_multilabel/weighted',
                '../runs_trainings/no_freeze/joint_incremental_multilabel/weighted',
                '../runs_trainings/no_freeze/joint_incremental_restored_multilabel/weighted/reset',

                '../runs_trainings/freeze_backbone/joint_incremental_restored_multilabel/weighted/reset',
                '../runs_trainings/freeze_backbone/joint_incremental_multilabel/weighted/new',
                '../runs_trainings/freeze_backbone/incremental_decremental_multilabel/weighted/new',
                '../runs_trainings/freeze_backbone/decremental_multilabel/weighted/new',
                '../runs_trainings/freeze_backbone/baseline_multilabel/weighted/new',]
seeds = [0,1,2]


forg_multidim_file_names = ['forg_accuracy_per_subcategory.out','forg_ap_per_subcategory.out',
                       'forg_recall_per_subcategory.out','forg_precision_per_subcategory.out']

data_dict = {
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


for res_path in results_path:
    statistics_save_path = os.path.join(res_path,'statistics')
    if not os.path.exists(statistics_save_path):
        os.mkdir(statistics_save_path)
    mean_std_over_tasks_path = os.path.join(statistics_save_path,'mean_std_forg_subcat_over_tasks')
    if not os.path.exists(mean_std_over_tasks_path):
        os.mkdir(mean_std_over_tasks_path)
    mean_over_tasks_path = os.path.join(statistics_save_path,'mean_forg_subcat_over_tasks')
    if not os.path.exists(mean_over_tasks_path):
        os.mkdir(mean_over_tasks_path)
    std_over_tasks_path = os.path.join(statistics_save_path,'std_forg_subcat_over_tasks')
    if not os.path.exists(std_over_tasks_path):
        os.mkdir(std_over_tasks_path)
    
    data = {}
    subcategories_names = None
    for file_name in forg_multidim_file_names:
        data[file_name] = []
        for idx_seed in seeds:
            seed_path = os.path.join(res_path, 'seed_' + str(idx_seed))
            
            for exp_dir in os.listdir(seed_path):
                exp_path = os.path.join(seed_path,exp_dir)
                logger_path = os.path.join(exp_path,'logger')
                file_path = os.path.join(logger_path,file_name)
                file_data = np.loadtxt(file_path,delimiter=',')
                data[file_name].append(file_data[1:])
                f = open(file_path)
                header = f.readline()
                subcategories_names = header[2:].replace('\n','').split(',')
    
    data_mean = {}
    data_std = {}
    data_string = {}
    mean_over_tasks = {}
    for file_name in forg_multidim_file_names:
        # this is done because i have the ap in 0,1 range, want to be in percentage
        if not ((file_name == 'mean_ap.out') or (file_name == 'map_weighted.out') or (file_name == 'forg_mean_ap.out') or (file_name == 'forg_acc.out') or (file_name == 'acc.out')):
            for i in range(len(data[file_name])):
                data[file_name][i] = data[file_name][i]*100
        
        # mean computed over tasks
        mean_over_tasks[file_name] = np.mean(data[file_name], axis=1)

        # now compute the mean of forgetting over the class subcategories
        
        classes_forg_subcat_mean = []

        
        for name_class in data_dict:
            class_means_all_seeds = []
            for idx_seed in seeds:
                class_indices = [subcategories_names.index(subcat) for subcat in data_dict[name_class]]
                current_subcat_values = [mean_over_tasks[file_name][idx_seed][idx] for idx in class_indices]
                class_means_all_seeds.append(np.mean(current_subcat_values, axis=0))
            classes_forg_subcat_mean.append(class_means_all_seeds)
        
        # mean and std computed over the seeds
        subcat_means_over_seeds = np.mean(classes_forg_subcat_mean, axis=1)
        subcat_std_over_seeds = np.std(classes_forg_subcat_mean, axis=1, ddof=1)
            
        np.savetxt(os.path.join(mean_over_tasks_path,file_name), np.column_stack(subcat_means_over_seeds),delimiter=',',fmt='%.3f')
        np.savetxt(os.path.join(std_over_tasks_path,file_name), np.column_stack(subcat_std_over_seeds),delimiter=',',fmt='%.3f')
        
        tmp_string_array = []
        for i in range(len(subcat_means_over_seeds)):
            
            tmp_string = "{:.1f}".format(subcat_means_over_seeds[i])+'\u00B1'+"{:.1f}".format(subcat_std_over_seeds[i])
            tmp_string_array.append(tmp_string)
            np.savetxt(os.path.join(mean_std_over_tasks_path,file_name), np.column_stack(tmp_string_array),delimiter=',',fmt='%s')

        