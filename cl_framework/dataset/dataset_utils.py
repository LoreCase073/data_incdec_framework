import sys 
import torchvision
from torchvision import transforms
import os
import matplotlib
import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import zipfile
from sklearn import preprocessing
import requests 
from io import BytesIO
from torchvision import datasets
from torch.utils.data import Dataset
from PIL import Image 
from typing import Tuple,Any 
from torch.utils.data import DataLoader
from tqdm import tqdm 
import torch

def find_classes(dir):
    if sys.version_info >= (3, 5):
        # Faster and available in Python 3.5 and above
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    else:
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def kinetics_classes(classes_csv):
    df = pd.read_csv(classes_csv)
    classes_behaviors = {}

    for _, row in df.iterrows():
        class_name = row['Class']
        subcategory = row['Subcategory']
        
        # Check if the class_name is already in the dictionary, if not, create a new entry
        if class_name not in classes_behaviors:
            classes_behaviors[class_name] = []
        
        # Add the subcategory to the corresponding class_name entry in the dictionary
        classes_behaviors[class_name].append(subcategory)

    return classes_behaviors


class KineticsDataset(Dataset):
    def __init__(self, data_path, transform, dataset_type, fps):

        #In folder_csv are place: train.csv, validation.csv, test.csv and classes.csv
        folder_csv = os.path.join(data_path,'Info')
        if dataset_type == 'train':
            self.data_csv = os.path.join(folder_csv, 'train.csv')
        elif dataset_type == 'validation':
            self.data_csv = os.path.join(folder_csv, 'validation.csv')
        elif dataset_type == 'test':
            self.data_csv = os.path.join(folder_csv, 'test.csv')
        else:
            #This only to get all the data, used for mean and std
            self.data_csv = os.path.join(folder_csv, 'tbdownloaded.csv')


        self.data_folder = os.path.join(data_path,'Videos')

        df = pd.read_csv(self.data_csv)

        self.data = []
        self.targets = []
        self.behaviors = []

        #create a mapping between classes - behaviors
        class_csv = os.path.join(folder_csv, 'classes.csv')
        self.classes_behaviors = kinetics_classes(class_csv)

        #create a index for each class -- {class: idx}
        self.class_to_idx = {key: i for i, key in enumerate(self.classes_behaviors.keys())}

        for _, row in df.iterrows():
            #replace to match how the data was called in the folder
            id_data = 'id_' + str(row['youtube_id']) + '_' + '{:06d}'.format(row['time_start']) + '_' + '{:06d}'.format(row['time_end'])
            self.data.append(id_data)

            #retrieve the class - targets from category.csv
            data_dir = os.path.join(self.data_folder, id_data)
            cat_csv_path = os.path.join(data_dir,'category.csv')
            cat_csv = pd.read_csv(cat_csv_path)
            cat_row = next(cat_csv.iterrows())[1]
            matching_class = cat_row['Category']
            #retrieve the behavior from category.csv
            self.targets.append(self.class_to_idx[matching_class])
            matching_behavior = cat_row['Sub-behavior']
            self.behaviors.append(matching_behavior)
        
        self.transform = transform

        self.fps = fps

    def get_class_to_idx(self):
        return self.class_to_idx
    
    # this is used to get the name given the data_path of a sample, done to not modify the data_loader
    def get_name_sample(self, data_path_sample):
        data_name = data_path_sample.replace(self.data_folder,'')
        if self.fps == 5:
            data_name = data_name.replace('/5fps_jpgs','')
        else:
            data_name = data_name.replace('/jpgs','')
        return data_name
    
    def __len__(self) -> int:
        return len(self.data)

    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        img_id, target, behavior = self.data[index], self.targets[index], self.behaviors[index]

        video_id_path = os.path.join(self.data_folder,img_id)
        if self.fps == 5:
            images_path = os.path.join(video_id_path,'5fps_jpgs')
        else:
            images_path = os.path.join(video_id_path,'jpgs')

        video = []
        std_video_len = self.fps*10

        tmp_len = len(os.listdir(images_path))

        
        for i in range(std_video_len):
            image_name = 'image_{:05d}.jpg'.format((i%tmp_len)+1)
            im_path = os.path.join(images_path,image_name)
            with open(im_path, 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')
                if self.transform is not None:
                        img = self.transform(img)
                video.append(img)      
        
        video = torch.stack(video,0).permute(1, 0, 2, 3)
        binarized_target = preprocessing.label_binarize([target], classes=[i for i in range(len(self.class_to_idx.keys()))])
        return video, target, binarized_target, behavior, images_path
    


class VZCDataset(Dataset):
    def __init__(self, data_path, transform, dataset_type, fps):

        #In folder_csv are place: train.csv, validation.csv, test.csv and classes.csv
        folder_csv = os.path.join(data_path,'Info')
        if dataset_type == 'train':
            self.data_csv = os.path.join(folder_csv, 'train.csv')
        elif dataset_type == 'validation':
            self.data_csv = os.path.join(folder_csv, 'validation.csv')
        elif dataset_type == 'test':
            self.data_csv = os.path.join(folder_csv, 'test.csv')


        self.data_folder = os.path.join(data_path,'Videos')

        df = pd.read_csv(self.data_csv)

        self.data = []
        self.targets = []
        
        #TODO: i guess not used for VZC case
        #self.behaviors = []

        #TODO: implement a class_to_idx dict, here i did an example
        #create a index for each class -- {class: idx}
        self.class_to_idx = {
            'nothing':0,
            'food':1,
            'phone':2,
            'cigarette':3,
        }
        

        for _, row in df.iterrows():
            #TODO: replace to match how the data was called in the folder of vzc
            # id_data = 'id_' + str(row['youtube_id']) + '_' + '{:06d}'.format(row['time_start']) + '_' + '{:06d}'.format(row['time_end'])
            id_data = str(row['id'])
            self.data.append(id_data)

            #retrieve the class - targets from category.csv
            data_dir = os.path.join(self.data_folder, id_data)
            cat_csv_path = os.path.join(data_dir,'category.csv')
            cat_csv = pd.read_csv(cat_csv_path)
            cat_row = next(cat_csv.iterrows())[1]
            matching_class = cat_row['Category']
            #retrieve the behavior from category.csv
            self.targets.append(self.class_to_idx[matching_class])
   
        self.transform = transform
        self.fps = fps

    def get_class_to_idx(self):
        return self.class_to_idx
    
    
    def __len__(self) -> int:
        return len(self.data)

    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        img_id, target = self.data[index], self.targets[index]

        video_id_path = os.path.join(self.data_folder,img_id)
        #TODO: to be modified, maybe will only be necessary a single directory control
        """
        if self.fps == 5:
            images_path = os.path.join(video_id_path,'5fps_jpgs')
        else:
        """
        images_path = os.path.join(video_id_path,'jpgs')

        video = []
        std_video_len = self.fps*16  # VZC videos are at most 16s (when everything is correct)

        tmp_len = len(os.listdir(images_path))

        
        for i in range(std_video_len):
            #TODO: to modify with jpgs names in VZC
            image_name = 'image_{:05d}.jpg'.format((i%tmp_len)+1)
            im_path = os.path.join(images_path,image_name)
            with open(im_path, 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')
                if self.transform is not None:
                        img = self.transform(img)
                video.append(img)      
        
        video = torch.stack(video,0).permute(1, 0, 2, 3)
        binarized_target = preprocessing.label_binarize([target], classes=[i for i in range(len(self.class_to_idx.keys()))])
        return video, target, binarized_target, 0, images_path


def get_train_val_images_tiny(data_path):
    if not os.path.exists(data_path):
        os.mkdir(data_path)
 
    train_dir = os.path.join(data_path, 'tiny-imagenet-200', 'train')
    test_dir = os.path.join(data_path, 'tiny-imagenet-200', 'val')
    train_dset = datasets.ImageFolder(train_dir)

    train_images = []
    train_labels = []
    for item in train_dset.imgs:
        train_images.append(item[0])
        train_labels.append(item[1])
    
    train_targets =  np.array(train_labels)
    
    test_images = []
    test_labels = []
    _, class_to_idx = find_classes(train_dir)
    imgs_path = os.path.join(test_dir, 'images')
    imgs_annotations = os.path.join(test_dir, 'val_annotations.txt')
    with open(imgs_annotations) as r:
        data_info = map(lambda s: s.split('\t'), r.readlines())
    cls_map = {line_data[0]: line_data[1] for line_data in data_info}
    for imgname in sorted(os.listdir(imgs_path)):
        if cls_map[imgname] in sorted(class_to_idx.keys()):
            path = os.path.join(imgs_path, imgname)
            test_images.append(path)
            test_labels.append(class_to_idx[cls_map[imgname]])
    
    test_targets =  np.array(test_labels)
    
    return train_images, train_targets, test_images, test_targets, class_to_idx

def get_train_val_images_imagenet_subset(data_path):
    if not os.path.exists(data_path):
        os.mkdir(data_path)
 
    
    train_dir = os.path.join(data_path, 'imagenet_subset', 'train')
    test_dir = os.path.join(data_path, 'imagenet_subset', 'val')
    train_dset = datasets.ImageFolder(train_dir)

    train_images = []
    train_labels = []
    for item in train_dset.imgs:
        train_images.append(item[0])
        train_labels.append(item[1])
    
    train_targets =  np.array(train_labels)
    _, class_to_idx = find_classes(train_dir)
    
    val_annotations = open(os.path.join(data_path, 'imagenet_subset',   'val_100.txt'), "r")
    
    val_lines = val_annotations.readlines()
    
    test_images = []
    test_labels = []
    for item in val_lines:
        splitted_item = item.rstrip()
        splitted_item  = splitted_item.split(" ")
        image_path = "/".join([splitted_item[0].split("/")[0], splitted_item[0].split("/")[2]])
        image_path = os.path.join(data_path, 'imagenet_subset', image_path)
        test_images.append(image_path)
        test_labels.append(int(splitted_item[1]))
        
    test_targets =  np.array(test_labels)
   
    return train_images, train_targets, test_images, test_targets, class_to_idx
    



class TinyImagenetDataset(Dataset):
    def __init__(self, data, targets, class_to_idx,  transform):
        self.data = data
        self.targets = targets
        self.transform = transform 
        self.class_to_idx = class_to_idx 
        
    
    def __len__(self) -> int:
        return len(self.data)

    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        img_path, target = self.data[index], self.targets[index]
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')


        if self.transform is not None:
            img = self.transform(img)
            
            
        return img, target
    

        
        
def get_dataset(dataset_type, data_path, pretrained_path=None):
    if  dataset_type == "cifar100":
        print("Loading Cifar 100")
        train_transform = [transforms.Pad(4), transforms.RandomResizedCrop(32), 
                           transforms.RandomHorizontalFlip(),transforms.ToTensor(),
                           transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]  
        train_transform = transforms.Compose(train_transform)

        test_transform = [transforms.ToTensor(),transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]  
        test_transform = transforms.Compose(test_transform)
 
  
        if dataset_type == "cifar100":
            train_set = torchvision.datasets.CIFAR100(root='./dataset/data', train=True,
                                                download=True, transform=train_transform)
            test_set = torchvision.datasets.CIFAR100(root='./dataset/data', train=False,
                                            download=True, transform=test_transform)
            #Validation is taken from train set in logic ahead in the code pipeline
            valid_set = None
            n_classes = 100
 

    elif dataset_type == "tiny-imagenet":
        # images_url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
        print("Loading Tiny Imagenet")
  
 
        train_data, train_targets, test_data, test_targets, class_to_idx = get_train_val_images_tiny(data_path)
        
        train_transform = transforms.Compose(
                                            [transforms.RandomCrop(64, padding=8), 
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                                            ])
 
        
        test_transform = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])])
        
 
        train_set = TinyImagenetDataset(train_data, train_targets, class_to_idx, train_transform)
        #Validation is taken from train set in logic ahead in the code pipeline
        valid_set = None
        test_set = TinyImagenetDataset(test_data, test_targets, class_to_idx, test_transform)
        
        n_classes = 200
        
    elif dataset_type == "imagenet-subset":
        print("Loading Imagenet Subset")
 
 
        train_data, train_targets, test_data, test_targets, class_to_idx = get_train_val_images_imagenet_subset(data_path)
        
        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                    ])
        
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
            ])
        
        train_set = TinyImagenetDataset(train_data, train_targets, class_to_idx, train_transform)
        #Validation is taken from train set in logic ahead in the code pipeline
        valid_set = None
        test_set = TinyImagenetDataset(test_data, test_targets, class_to_idx, test_transform)
        
        n_classes = 100
    elif dataset_type == "kinetics":
        
        print("Loading Kinetics")
        
        if pretrained_path == None:
            train_transform = [transforms.Resize(size=(200,200)),
                            #TODO: in futuro provare con un random crop invece che centercrop
                            transforms.RandomCrop(172),
                            transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                #added normalization factors computed on the actual dataset, training set
                                transforms.Normalize(mean=[0.4516, 0.3883, 0.3569],std=[0.2925, 0.2791, 0.2746])
                            ]
            train_transform = transforms.Compose(train_transform)

            test_transform = [transforms.Resize(size=(172,172)),
                                transforms.CenterCrop(172),
                                transforms.ToTensor(),
                                #added normalization factors computed on the actual dataset, training set
                                transforms.Normalize(mean=[0.4516, 0.3883, 0.3569],std=[0.2925, 0.2791, 0.2746])
                            ]  
            test_transform = transforms.Compose(test_transform)
        else:
            train_transform = [transforms.Resize(size=(200,200)),
                        transforms.RandomCrop(172),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],std=[0.22803, 0.22145, 0.216989])
                            ]
            train_transform = transforms.Compose(train_transform)
        
            test_transform = [transforms.Resize(size=(200,200)),
                            transforms.CenterCrop(172),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],std=[0.22803, 0.22145, 0.216989])
                            ]  
            test_transform = transforms.Compose(test_transform)


 

        #TODO: per ora fps passati sempre a 5, aggiungere logica esterna per modificarlo
        train_set = KineticsDataset(data_path, train_transform, dataset_type='train', fps=5)
        # Here validation is passed outside, separately from the train. In the future could be a subset of training
        valid_set = KineticsDataset(data_path, test_transform, dataset_type='validation', fps=5)
        test_set = KineticsDataset(data_path, test_transform, dataset_type='test', fps=5)

        #TODO: per ora aggiungo a mano, modificare da prendere dall'esterno
        n_classes = 5
    elif dataset_type == "vzc":
        
        print("Loading vzc Dataset")
        
        train_transform = [transforms.Resize(size=(200,200)),
                           transforms.RandomCrop(172),
                           transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            #TODO: normalize is at the moment with the data from Kinetics used by us, to be modified with your data
                            transforms.Normalize(mean=[0.4516, 0.3883, 0.3569],std=[0.2925, 0.2791, 0.2746])
                ]
        train_transform = transforms.Compose(train_transform)

        test_transform = [transforms.Resize(size=(172,172)),
                            transforms.CenterCrop(172),
                            transforms.ToTensor(),
                            #TODO: normalize is at the moment with the data from Kinetics used by us, to be modified with your data
                            transforms.Normalize(mean=[0.4516, 0.3883, 0.3569],std=[0.2925, 0.2791, 0.2746])
                          ]  
        test_transform = transforms.Compose(test_transform)


 

        #TODO: for now fps are set here, to be passed from outside later in implementation
        train_set = VZCDataset(data_path, train_transform, dataset_type='train', fps=5)
        # Here validation is passed outside, separately from the train. In the future could be a subset of training
        valid_set = VZCDataset(data_path, test_transform, dataset_type='validation', fps=5)
        test_set = VZCDataset(data_path, test_transform, dataset_type='test', fps=5)

        #TODO: for now set here, to be passed from outside later in implementation
        n_classes = 4
        
    
    return train_set, test_set, valid_set, n_classes

 
            
        