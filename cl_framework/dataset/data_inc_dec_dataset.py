from torch.utils.data import Subset
import numpy as np

class DataIncDecBaseline():
    def __init__(self, dataset, task_dictionary,  
                    n_task, initial_split,
                    total_classes, train=True):
        
        self.dataset = dataset
        self.train = train 
        #TODO:definire se necessito di task_dictionary e per cosa
        self.task_dictionary = task_dictionary 
        #initial_split will explain how to split the data
        #will usually be 50-50
        self.initial_split = initial_split
        #n_task will also say how much will need to split the second 50% of data
        self.n_task = n_task
        self.len_dataset = len(dataset)
        #number of classes
        self.total_classes = total_classes

    def collect(self):
        #train
        if self.train:
            train_indices_list = [[] for _ in range(self.n_task)]  # List of list, each list contains indices in the entire dataset to accumulate for the task
            #TODO: per ora suppongo di dividere per due, ma posso modificare come farlo... logica passata da fuori
            
            initial_train_split = int((self.len_dataset/self.initial_split))
            #TODO: rimuovere, forse questo non utile..
            subsequent_train_split = int((self.len_dataset)) - initial_train_split


            sub_split_indices = []

            #TODO: controllare di aver fatto i due split iniziali correttamente
            #now divide the two first splits of the dataset
            for idx_class in range(self.total_classes):
                #takes the class name from the idx
                class_name = [ c for c, idx in self.dataset.class_to_idx.items() if idx == idx_class ]

                #iterate over the behaviors for class_name
                for it_behaviors in range(self.dataset.classes_behaviors[class_name]):
                    
                    current_behavior_indices = np.where(np.array(self.dataset.behaviors) == it_behaviors)[0]

                    #create the two splits for each behavior
                    first_split_indices = current_behavior_indices[:initial_train_split]
                    second_split_indices = current_behavior_indices[initial_train_split:]
                    #add these indeces for the first task
                    train_indices_list[0].extend(list(first_split_indices))
                    sub_split_indices.extend(list(second_split_indices))

                
            #TODO: fare tutti split per ogni task, da sub_split_indices
            for i in range(self.n_task):
                pass
                #TODO: finire logica di splitting
                
                        
            
            cl_train_dataset = [Subset(self.dataset, ids)  for ids in train_indices_list]
            cl_train_sizes = [len(ids) for ids in train_indices_list]

  
            return cl_train_dataset, cl_train_sizes, None, None
        
        else:
            #test
            test_indices_list = [[] for _ in range(self.n_task)] 
            for cc in range(self.total_classes):
                test_indices  = np.where(np.array(self.dataset.targets) == cc)[0]
                for task_id, task_classes in self.task_dictionary.items():
                    if cc in task_classes:
                        test_indices_list[task_id].extend(list(test_indices))
            
            cl_test_dataset = [Subset(self.dataset, ids)  for ids in test_indices_list]
            cl_test_sizes =[len(ids) for ids in test_indices_list]

            return cl_test_dataset, cl_test_sizes, None, None