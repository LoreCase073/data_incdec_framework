#imports to work with...
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import DataLoader
import torch
import torchvision
from cl_framework.utilities.generic_utils import rollback_model
from cl_framework.continual_learning.models.BaseModel import BaseModel



def compute_distances(models_path):
    distances_models = []
    for idx_experiment in range(len(models_path)):
        current_experiment = models_path[idx_experiment]
        current_experiment_distances = []
        for idx_seed in range(len(current_experiment)):
            current_seed = current_experiment[idx_seed]
            current_seed_distances = []

            for idx_path in range(len(current_seed)-1):
                # get the first model
                initial_model = BaseModel('movinetA0', 'kinetics')
                initial_model.add_classification_head(5)
                initial_checkpoint = torch.load(current_seed[idx_path], map_location='cpu')
                initial_model.load_state_dict(initial_checkpoint)
                # get other model to compute the distance
                tmp_model = BaseModel('movinetA0', 'kinetics')
                tmp_model.add_classification_head(5)
                checkpoint = torch.load(current_seed[idx_path+1], map_location='cpu')
                tmp_model.load_state_dict(checkpoint)

                distances = []
                iterator_initial_model = initial_model.named_parameters()
                for name, params in tmp_model.named_parameters():
                    _,initial_params = next(iterator_initial_model)
                    distances.append(torch.dist(initial_params,params,p=2).item())
                current_seed_distances.append(distances)
            current_experiment_distances.append(current_seed_distances)
        distances_models.append(current_experiment_distances)
    
    return distances_models



def compute_distances_per_block(models_path, blocks):
    distances_models = []
    for idx_experiment in range(len(models_path)):
        current_experiment = models_path[idx_experiment]
        current_experiment_distances = []
        for idx_seed in range(len(current_experiment)):
            current_seed = current_experiment[idx_seed]
            current_seed_distances = []
            


            for idx_path in range(len(current_seed)-1):
                # get the first model
                initial_model = BaseModel('movinetA0', 'kinetics')
                initial_model.add_classification_head(5)
                initial_checkpoint = torch.load(current_seed[idx_path], map_location='cpu')
                initial_model.load_state_dict(initial_checkpoint)
                # get other model to compute the distance
                tmp_model = BaseModel('movinetA0', 'kinetics')
                tmp_model.add_classification_head(5)
                checkpoint = torch.load(current_seed[idx_path+1], map_location='cpu')
                tmp_model.load_state_dict(checkpoint)

                block_distances = []
                
                for block in blocks:
                    initial_layers = []
                    layers = []
                    iterator_initial_model = initial_model.named_parameters()
                    for name, params in tmp_model.named_parameters():
                        _,initial_params = next(iterator_initial_model)
                        if block in name:
                            initial_layers.append(initial_params.flatten())
                            layers.append(params.flatten())
                    initial_layers = torch.cat(initial_layers)
                    layers = torch.cat(layers)
                    block_distances.append(torch.dist(initial_layers,layers,p=2).item())
                current_seed_distances.append(block_distances)
            current_experiment_distances.append(current_seed_distances)
        distances_models.append(current_experiment_distances)
    
    return distances_models
        
    
def plot_distances(distances, path, names, idx_name):
    for idx_seed in range(len(distances)):
        x = [i for i in range(len(distances[idx_seed][idx_name]))]
        plt.plot(x,distances[idx_seed][idx_name], label=names)
        
    
    #plt.close()



def plot_distances_seeds(distances, distance_std, path, names, idx_name):
    
    x = [i for i in range(len(distances[idx_name]))]
    plt.plot(x,distances[idx_name], label=names)
    #plt.errorbar(x,distances[idx_name], distance_std[idx_name], label=names)

if __name__ == "__main__":
    
    experiment_paths = ["runs_trainings/from_checkpoint_sgd/no_freeze/incremental_decremental/4_4_lr",
                        "runs_trainings/from_checkpoint_sgd/freeze/incremental_decremental",
                        "runs_trainings/from_checkpoint_sgd/lwf/incremental_decremental",
                        "runs_trainings/from_checkpoint_sgd/fd/incremental_decremental",
                   ]
    
    task_model_names = [
                "best_mAP_task_0_model.pth",
                "best_mAP_task_1_model.pth",
                "best_mAP_task_2_model.pth",
                "best_mAP_task_3_model.pth",
                "best_mAP_task_4_model.pth",
                "best_mAP_task_5_model.pth",
    ]
    
    seeds = [0,1,2]
    model_paths = []
    for experiment_path in experiment_paths:
        tmp_path = []
        for idx_seed in seeds:
            seed_path = os.path.join(experiment_path, 'seed_' + str(idx_seed))
            seed_experiment_list = []
            for exp_dir in os.listdir(seed_path):
                exp_path = os.path.join(seed_path,exp_dir)
                for task_model in task_model_names:
                    path = os.path.join(exp_path,task_model)
                    seed_experiment_list.append(path)
            tmp_path.append(seed_experiment_list)
        model_paths.append(tmp_path)



    blocks = [
        "conv1",

        "blocks.b0_l0",

        "blocks.b1_l0",
        "blocks.b1_l1",
        "blocks.b1_l2",

        "blocks.b2_l0",
        "blocks.b2_l1",
        "blocks.b2_l2",

        "blocks.b3_l0",
        "blocks.b3_l1",
        "blocks.b3_l2",
        "blocks.b3_l3",

        "blocks.b4_l0",
        "blocks.b4_l1",
        "blocks.b4_l2",
        "blocks.b4_l3",

        "conv7",
        "conv9",

        #"heads",
    ]

    #distances = compute_distances(model_paths)

    

    block_distances = compute_distances_per_block(model_paths, blocks)

    block_distances_mean = np.mean(block_distances, axis=1)

    block_distances_std = np.std(block_distances, axis=1, ddof=1)

    save_path = 'statistics_to_save/weights_distances/incdec_pretrained/finetune/'

    names = ["task0_1",
             "task0_2",
             "task0_3",
             "task0_4",
             "task0_5",
             ]
    
    names_b = ["block_task0_1",
               "block_task1_2",
               "block_task2_3",
               "block_task3_4",
               "block_task4_5",
             ]

    method_name = [
        "finetuning",
        "feature_extraction",
        "lwf",
        "fd",
    ]
    
    for idx_experiment in range(len(block_distances_mean)):
        #current_experiment_distances = distances[idx_experiment]
        current_experiment_blocks_distances_mean = block_distances_mean[idx_experiment]
        current_experiment_blocks_distances_std = block_distances_std[idx_experiment]
        for idx_name in range(len(names)):
            #plot_distances(current_experiment_distances, save_path, names[idx_name], idx_experiment)
            
            plot_distances_seeds(current_experiment_blocks_distances_mean, current_experiment_blocks_distances_std, save_path, names_b[idx_name], idx_name)
        plt.legend(loc='best')
        plt.savefig(save_path + method_name[idx_experiment] + '_all.png')
        plt.close()
        