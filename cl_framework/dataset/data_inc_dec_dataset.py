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
            
            #TODO: to be removed
            initial_train_split = int((self.len_dataset/self.initial_split))
            


            first_split = []
            second_split = []

            #TODO: controllare di aver fatto i due split iniziali correttamente
            #now divide the two first splits of the dataset
            for idx_class in range(self.total_classes):
                #takes the class name from the idx
                class_name = [ c for c, idx in self.dataset.class_to_idx.items() if idx == idx_class ]

                #iterate over the behaviors for class_name
                for it_behaviors in range(self.dataset.classes_behaviors[class_name]):
                    
                    current_behavior_indices = np.where(np.array(self.dataset.behaviors) == it_behaviors)[0]

                    num_data_first_split = int(len(current_behavior_indices)/self.initial_split)

                    #create the two splits for each behavior
                    first_split_indices = current_behavior_indices[:num_data_first_split]
                    second_split_indices = current_behavior_indices[num_data_first_split:]
                    #add these indices for the first task
                    first_split.extend(list(first_split_indices))
                    second_split.extend(list(second_split_indices))

            #TODO: forse da rimuovere questi due 
            #number of video to substitute from the first split and to add from the second split
            n_first_split = int((len(first_split)/self.initial_split))
            n_second_split = int((len(second_split)/self.initial_split))


            for i in range(self.n_task):

                for idx_class in range(self.total_classes):
            

                    #takes indices of the class
                    current_class_indices = np.where(np.array(self.dataset.targets) == idx_class)[0]

                    #takes indices of the class from the first split
                    f_class_indices = [idx for idx in current_class_indices if idx in train_indices_list[0]]

                    sec_class_indices = [idx for idx in current_class_indices if idx in second_split_indices]

                    #number of data from the first split to be removed
                    f_data_task = int(len(f_class_indices)/self.n_task)

                    #number of data from the second split to be added
                    sec_data_task = int(len(sec_class_indices)/self.n_task)

                    #indices from the first split 
                    f_idx = current_class_indices[f_data_task*i:]

                    #indices from the second split 
                    s_idx = current_class_indices[:sec_data_task*i]

                    #add and remove data and make the list of indices for the task
                    train_indices_list[i].extend(list(f_idx + s_idx))

            #TODO: controllare funzioni questa logica per creare liste di indici
                
            
            
                
                        
            
            cl_train_dataset = [Subset(self.dataset, ids)  for ids in train_indices_list]
            cl_train_sizes = [len(ids) for ids in train_indices_list]

  
            return cl_train_dataset, cl_train_sizes, None, None
        
        else:
            #test
            
            #TODO: fare logica per test set?
            #dovrei prendere comunque tutto il test set... modificare in questa maniera
            cl_test_dataset = []
            cl_test_sizes =[]

            return cl_test_dataset, cl_test_sizes, None, None