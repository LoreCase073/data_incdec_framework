from torch.utils.data import Subset, WeightedRandomSampler
import numpy as np
import torch

""" TODO: modificare come spiegato su notion, in Refactoring e pulizia codice.
    TLDR: fare che elementi interni a dataset sono passati dall'esterno (per pulizia e chiarezza codice)
    e implementare collect anche senza behaviors per uso di Verizon.
 """
class DataIncDecBaselineDataset():
    def __init__(self, dataset,  
                    n_task, initial_split,
                    total_classes, behaviors_check='yes', train=True, validation=None, valid_size=None):
        
        self.dataset = dataset
        self.train = train 

        #initial_split will explain how to split the data
        #will usually be 50-50
        self.initial_split = initial_split
        #n_task will also say how much will need to split the second 50% of data
        self.n_task = n_task
        self.len_dataset = len(dataset)
        #number of classes
        self.total_classes = total_classes

        self.validation = validation
        #compute class sample count for the sample weights to be used in weighted random sample
        self.class_sample_count = np.array([len(np.where(self.dataset.targets == t)[0]) for t in np.unique(self.dataset.targets)])
        self.sample_weight = 1. / self.class_sample_count
        self.behaviors_check = behaviors_check


    

    def collect(self):
        #train
        if self.train:

            # List of list, each list contains indices in the entire dataset to accumulate for the task
            train_indices_list = [[] for _ in range(self.n_task)]  
            
            first_split, second_split = self.get_initial_splits()


            
            for idx_class in range(self.total_classes):
                #takes indices of the class
                current_class_indices = np.where(np.array(self.dataset.targets) == idx_class)[0]
                np.random.shuffle(current_class_indices)
                for i in range(self.n_task):     

                    #takes indices of the class from the first split
                    f_class_indices = [idx for idx in current_class_indices if idx in first_split]

                    sec_class_indices = [idx for idx in current_class_indices if idx in second_split]

                    #number of data from the first split to be removed
                    if self.n_task > 1:
                        f_data_task = int(len(f_class_indices)/(self.n_task-1))
                    else:
                        f_data_task = int(len(f_class_indices))
                    #number of data from the second split to be added
                    if self.n_task > 1:
                        sec_data_task = int(len(sec_class_indices)/(self.n_task-1))
                    else:
                        sec_data_task = int(len(sec_class_indices))

                    #indices from the first split 
                    f_idx = f_class_indices[f_data_task*i:]

                    #indices from the second split 
                    s_idx = sec_class_indices[:sec_data_task*i]

                    #add and remove data and make the list of indices for the task
                    train_indices_list[i].extend(list(f_idx + s_idx))
                      
            
            cl_train_dataset = [Subset(self.dataset, ids)  for ids in train_indices_list]
            cl_train_sizes = [len(ids) for ids in train_indices_list]

            #TODO: implementare validation set, sia per validation esterno sia da separarlo da train
            val_indices_list = [[] for _ in range(self.n_task)] 
            if self.validation != None:
                #Here validation passed from out of the train, same for all the tasks
                for i in range(self.n_task):
                    for idx_class in range(self.total_classes):
                        current_class_indices = np.where(np.array(self.validation.targets) == idx_class)[0]
                        val_indices_list[i].extend(list(current_class_indices))
            else:
                #TODO: implement if validation is not passed from out of the train
                pass

            cl_val_dataset = [Subset(self.validation, ids)  for ids in val_indices_list]
            cl_val_sizes =[len(ids) for ids in val_indices_list]

  
            return cl_train_dataset, cl_train_sizes, cl_val_dataset, cl_val_sizes
        
        else:
            #TEST
            
            # test indices should be equal for each task
            test_indices_list = [[] for _ in range(self.n_task)] 
            for i in range(self.n_task):
                    for idx_class in range(self.total_classes):
                        current_class_indices = np.where(np.array(self.dataset.targets) == idx_class)[0]
                        test_indices_list[i].extend(list(current_class_indices))
            
            cl_test_dataset = [Subset(self.dataset, ids)  for ids in test_indices_list]
            cl_test_sizes =[len(ids) for ids in test_indices_list]

            return cl_test_dataset, cl_test_sizes, None, None
        
    def get_weighted_random_sampler(self,indices):
        
        samples_weight = np.array(self.sample_weight[[self.dataset.targets[i] for i in indices]])
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

        return sampler
    
    def get_initial_splits(self):
        first_split = []
        second_split = []
        # do the splits including behavior information in the splits, if not just work on the classes
        if self.behaviors_check == 'yes':
            #now divide the two first splits of the dataset
            for idx_class in range(self.total_classes):
                #takes the class name from the idx
                #class_to_idx {class: idx}
                class_name = [ c for c, idx in self.dataset.class_to_idx.items() if idx == idx_class ][0]

                #iterate over the behaviors for class_name
                #classes_behaviors {class: [sub1,sub2...]}
                for it_behaviors in (self.dataset.classes_behaviors[class_name]):
                    
                    #Return indices of elements from the current it_behaviors
                    current_behavior_indices = np.where(np.array(self.dataset.behaviors) == it_behaviors)[0]
                    np.random.shuffle(current_behavior_indices)

                    num_data_first_split = int(len(current_behavior_indices)/self.initial_split)

                    #create the two splits for each behavior
                    first_split_indices = current_behavior_indices[:num_data_first_split]
                    second_split_indices = current_behavior_indices[num_data_first_split:]
                    #add these indices for the first task
                    first_split.extend(list(first_split_indices))
                    second_split.extend(list(second_split_indices))
        else:
            for idx_class in range(self.total_classes):
                #takes the class name from the idx
                #class_to_idx {class: idx}
                
                current_class_indices = np.where(np.array(self.dataset.targets) == idx_class)[0]
                np.random.shuffle(current_class_indices)

                num_data_first_split = int(len(current_class_indices)/self.initial_split)

                #create the two splits for each behavior
                first_split_indices = current_class_indices[:num_data_first_split]
                second_split_indices = current_class_indices[num_data_first_split:]
                #add these indices for the first task
                first_split.extend(list(first_split_indices))
                second_split.extend(list(second_split_indices))
        
        return first_split, second_split


""" 
Work in progress...
 """
class DataIncrementalDecrementalPipelineDataset():
    def __init__(self, dataset, behaviors_dictionary, 
                    n_task, initial_split,
                    total_classes, behaviors_check='yes', train=True, validation=None, valid_size=None):
        
        self.dataset = dataset
        self.train = train
        self.behaviors_dictionary = behaviors_dictionary

        #initial_split will explain how to split the data
        #will usually be 50-50
        self.initial_split = initial_split
        #n_task will also say how much will need to split the second 50% of data
        self.n_task = n_task
        self.len_dataset = len(dataset)
        #number of classes
        self.total_classes = total_classes

        self.validation = validation
        #compute class sample count for the sample weights to be used in weighted random sample
        """ TODO: necessario anche nel caso di DataDecremental?
        se necessario anche nel caso DataDecremental, devo prenderlo ad ogni iterazione diverso probabilmente, 
        da prendere nella funzione direttamente
        TODO:modificare
         """
        """ self.class_sample_count = np.array([len(np.where(self.dataset.targets == t)[0]) for t in np.unique(self.dataset.targets)])
        self.sample_weight = 1. / self.class_sample_count
         """
        self.behaviors_check = behaviors_check


    

    def collect(self):
        #train
        if self.train:

            # List of list, each list contains indices in the entire dataset to accumulate for the task
            train_indices_list = [[] for _ in range(self.n_task)]  
            
            first_split, second_split = self.get_initial_splits()



            #TODO: controllare di aver fatto l'eliminazione dei behaviors e quindi la suddivisione del dataset correttamente

            for idx_class in range(self.total_classes):
                # get the class name to use in behaviors_dictionary
                class_name = [ c for c, idx in self.dataset.class_to_idx.items() if idx == idx_class ][0]
                for it_behaviors in (self.dataset.classes_behaviors[class_name]):
                    #Return indices of elements from the current it_behaviors
                    current_behavior_indices = np.where(np.array(self.dataset.behaviors) == it_behaviors)[0]
                    np.random.shuffle(current_behavior_indices)
                    
                    # now iterate over the behaviors that should be different for each task
                    for idx_task in range(self.n_task):
                       
                        # get the current task behavior_dict
                        current_behaviors_dict = self.behaviors_dictionary[idx_task]
                        #check if the behavior selected is in the current task
                        if it_behaviors in current_behaviors_dict[class_name]:

                            #takes indices of the current behavior from the first split and second split
                            f_class_indices = [idx for idx in current_behavior_indices if idx in first_split]

                            sec_class_indices = [idx for idx in current_behavior_indices if idx in second_split]

                            #number of data from the second split to be added
                            if self.n_task > 1:
                                sec_data_task = int(len(sec_class_indices)/(self.n_task-1))
                            else:
                                sec_data_task = int(len(sec_class_indices))

                            #indices from the first split, i take all of them
                            f_idx = f_class_indices[:]

                            #indices from the second split, i take only a portion, incremental with the n_task
                            s_idx = sec_class_indices[:sec_data_task*idx_task]

                            #add and remove data and make the list of indices for the task
                            train_indices_list[idx_task].extend(list(f_idx + s_idx))


                      
            
            cl_train_dataset = [Subset(self.dataset, ids)  for ids in train_indices_list]
            cl_train_sizes = [len(ids) for ids in train_indices_list]

            #TODO: controllare che validation sia preso in maniera corretta
            val_indices_list = [[] for _ in range(self.n_task)] 
            if self.validation != None:
                
                #Here validation passed from out of the train, but it's not the same for all the tasks
                for idx_class in range(self.total_classes):
                    # get the class name to use in behaviors_dictionary
                    class_name = [ c for c, idx in self.validation.class_to_idx.items() if idx == idx_class ][0]
                    for idx_task in range(self.n_task):
                        # get the current task behavior_dict
                        current_behaviors_dict = self.behaviors_dictionary[idx_task]
                        # now iterate over the behaviors that should be different for each task
                        for it_behaviors in (current_behaviors_dict[class_name]):
                            #Return indices of elements from the current it_behaviors
                            current_behavior_indices = np.where(np.array(self.validation.behaviors) == it_behaviors)[0]

                            #add and remove data and make the list of indices for the task
                            val_indices_list[idx_task].extend(list(current_behavior_indices))
            else:
                #TODO: implement if validation is not passed from out of the train
                pass

            cl_val_dataset = [Subset(self.validation, ids)  for ids in val_indices_list]
            cl_val_sizes =[len(ids) for ids in val_indices_list]

  
            return cl_train_dataset, cl_train_sizes, cl_val_dataset, cl_val_sizes
        
        else:
            #TEST
            
            #TODO: controllare logica per test, dovrebbe essere sempre uguale per ogni task
            test_indices_list = [[] for _ in range(self.n_task)] 
            for i in range(self.n_task):
                    for idx_class in range(self.total_classes):
                        current_class_indices = np.where(np.array(self.dataset.targets) == idx_class)[0]
                        test_indices_list[i].extend(list(current_class_indices))
            
            cl_test_dataset = [Subset(self.dataset, ids)  for ids in test_indices_list]
            cl_test_sizes =[len(ids) for ids in test_indices_list]

            return cl_test_dataset, cl_test_sizes, None, None
        
    def get_weighted_random_sampler(self,indices):
        """ 
         TODO: controllare di aver fatto il sampler correttamente """
        tmp_targets = [self.dataset.targets[i] for i in indices]
        class_sample_count = np.array([len(np.where(tmp_targets == t)[0]) for t in np.unique(tmp_targets)])
        sample_weight = 1. / class_sample_count

        samples_weight = np.array(sample_weight[[self.dataset.targets[i] for i in indices]])
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

        return sampler
    
    def get_initial_splits(self):
        first_split = []
        second_split = []
        # do the splits including behavior information in the splits, if not just work on the classes
        if self.behaviors_check == 'yes':
            #now divide the two first splits of the dataset
            for idx_class in range(self.total_classes):
                #takes the class name from the idx
                #class_to_idx {class: idx}
                class_name = [ c for c, idx in self.dataset.class_to_idx.items() if idx == idx_class ][0]

                #iterate over the behaviors for class_name
                #classes_behaviors {class: [sub1,sub2...]}
                for it_behaviors in (self.dataset.classes_behaviors[class_name]):
                    
                    #Return indices of elements from the current it_behaviors
                    current_behavior_indices = np.where(np.array(self.dataset.behaviors) == it_behaviors)[0]
                    np.random.shuffle(current_behavior_indices)

                    num_data_first_split = int(len(current_behavior_indices)/self.initial_split)

                    #create the two splits for each behavior
                    first_split_indices = current_behavior_indices[:num_data_first_split]
                    second_split_indices = current_behavior_indices[num_data_first_split:]
                    #add these indices for the first task
                    first_split.extend(list(first_split_indices))
                    second_split.extend(list(second_split_indices))
        else:
            for idx_class in range(self.total_classes):
                #takes the class name from the idx
                #class_to_idx {class: idx}
                
                current_class_indices = np.where(np.array(self.dataset.targets) == idx_class)[0]
                np.random.shuffle(current_class_indices)

                num_data_first_split = int(len(current_class_indices)/self.initial_split)

                #create the two splits for each behavior
                first_split_indices = current_class_indices[:num_data_first_split]
                second_split_indices = current_class_indices[num_data_first_split:]
                #add these indices for the first task
                first_split.extend(list(first_split_indices))
                second_split.extend(list(second_split_indices))
        
        return first_split, second_split