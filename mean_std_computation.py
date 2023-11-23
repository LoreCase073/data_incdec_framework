import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch
import torchvision
from torchvision import transforms

from cl_framework.dataset.dataset_utils import KineticsDataset
from tqdm import tqdm
import os
import numpy as np


def read_patches(images, type="rgb"):
    

    images = images.permute(0,2,1,3,4)
    patches = images.reshape(images.shape[0]*images.shape[1],images.shape[2], images.shape[3], images.shape[4])
    # std = sqrt(E[X^2] - (E[X])^2)
    channels_sum = torch.sum(torch.mean(patches , dim=[2,3]), dim=0)
    channels_squared_sum = torch.sum(torch.mean(patches**2, dim=[2,3]), dim=0)
    num_patches = patches.shape[0]

       
    return channels_sum, channels_squared_sum, num_patches



""" path_to_download_log = './Kinetics/Download/attempt_14/download_log.csv'
data_csv = pd.read_csv(path_to_download_log) """

train_transform = [transforms.Resize(size=(172,172)),
                           transforms.CenterCrop(172),
                transforms.ToTensor(),
                #TODO:normalize?
                #transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]
train_transform = transforms.Compose(train_transform)

data = KineticsDataset('./Kinetics', train_transform, dataset_type='train', fps=5)

loader = DataLoader(data, batch_size=10)

mean    = torch.tensor([0.0, 0.0, 0.0])
squared_mean = torch.tensor([0.0, 0.0, 0.0])
n_el = torch.tensor([0.0])

for images, _, _ in tqdm(loader):
    mean_val, squared_mean_val, num_patches = read_patches(images)

    mean += mean_val
    squared_mean += squared_mean_val
    n_el += num_patches
    

total_mean = mean / n_el
total_std  = ((squared_mean / n_el) - (total_mean ** 2)) **0.5


print(f"Mean is: {total_mean}")
print(f"Std is: {total_std}")