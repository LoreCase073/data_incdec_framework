import numpy as np
import os
from datetime import datetime
import sys 

class Logger():
    def __init__(self, out_path, n_task, task_dict, test_sizes, begin_time=None) -> None:
        self.acc_taw = np.zeros((n_task, n_task))
        self.forg_taw = np.zeros((n_task, n_task))
        
        self.acc_tag = np.zeros((n_task, n_task))
        self.forg_tag  = np.zeros((n_task, n_task))

        self.perstep_acc_taw = np.zeros((n_task, n_task))
        self.perstep_acc_tag = np.zeros((n_task, n_task))

        self.pred_taw = np.zeros((n_task, n_task))
        self.pred_tag = np.zeros((n_task, n_task))
 
        self.task_len  =  [len(item) for item in task_dict.values()]
        self.test_sizes = test_sizes

        self.out_path = os.path.join(out_path, "logger")
        
        if begin_time is None:
            self.begin_time = datetime.now()
        else:
            self.begin_time = begin_time
        
        self.begin_time_str = self.begin_time.strftime("%Y-%m-%d-%H-%M")
        sys.stdout = FileOutputDuplicator(sys.stdout,
                                          os.path.join(out_path, 'stdout-{}.txt'.format(self.begin_time_str)), 'w')
        sys.stderr = FileOutputDuplicator(sys.stderr,
                                          os.path.join(out_path, 'stderr-{}.txt'.format(self.begin_time_str)), 'w')
        


    def update_accuracy(self, current_training_task_id, test_id, acc_taw_value, acc_tag_value):
        self.acc_taw[current_training_task_id, test_id] = acc_taw_value * 100
        self.acc_tag[current_training_task_id, test_id] = acc_tag_value * 100

        self.pred_taw[current_training_task_id, test_id] =  acc_taw_value * self.test_sizes[test_id]
        self.pred_tag[current_training_task_id, test_id] = acc_tag_value * self.test_sizes[test_id]

        self.perstep_acc_taw[current_training_task_id, test_id] =  (acc_taw_value*100) *  self.task_len[test_id]
        self.perstep_acc_tag[current_training_task_id, test_id] = (acc_tag_value*100) *  self.task_len[test_id] 





    def update_forgetting(self, current_training_task_id, test_id):
        self.forg_taw[current_training_task_id, test_id] = self.acc_taw[:current_training_task_id, test_id].max(0) - self.acc_taw[current_training_task_id, test_id]
        self.forg_tag[current_training_task_id, test_id] = self.acc_tag[:current_training_task_id, test_id].max(0) - self.acc_tag[current_training_task_id, test_id]
  

    def print_latest(self, current_training_task_id, test_id):
        print('\n >>> Test on task {:2d} : TAw acc={:5.1f}%, forg={:5.1f}%'
              ' | TAg acc ={:5.1f}%, forg ={:5.1f}%  <<<'.format(test_id, 
                                                                self.acc_taw[current_training_task_id, test_id],  self.forg_taw[current_training_task_id, test_id],
                                                                self.acc_tag[current_training_task_id, test_id],  self.forg_tag[current_training_task_id, test_id]))


    def compute_average(self):

        self.avg_acc_taw = self.acc_taw.sum(1) / np.tril(np.ones(self.acc_taw.shape[0])).sum(1)
        self.avg_perstep_acc_taw = self.perstep_acc_taw.sum(1) / (np.tril(np.array(self.task_len))).sum(1)
    
        
        self.avg_acc_tag = self.acc_tag.sum(1) / np.tril(np.ones(self.acc_tag.shape[0])).sum(1)
        self.avg_perstep_acc_tag = self.perstep_acc_tag.sum(1) / (np.tril(np.array(self.task_len))).sum(1)
     
        if  np.array_equal(self.forg_taw, np.zeros((self.forg_taw.shape[0],self.forg_taw.shape[0]))):
            self.avg_forg_taw = np.zeros(self.forg_taw.shape[0])
            self.avg_forg_tag  = np.zeros(self.forg_taw.shape[0])

        else:
            np.seterr(invalid='ignore')
            self.avg_forg_taw = self.forg_taw.sum(1) / np.tril(np.ones(self.forg_taw.shape[0]) - np.eye(self.forg_taw.shape[0])).sum(1)
            self.avg_forg_tag  = self.forg_tag.sum(1) / np.tril(np.ones(self.forg_tag.shape[0])- np.eye(self.forg_tag.shape[0])).sum(1)
           
          
    def print_file(self):
        np.savetxt(os.path.join(self.out_path, 'acc_taw.out'), self.acc_taw, delimiter=',', fmt='%.3f')
        np.savetxt(os.path.join(self.out_path, 'pred_taw.out'), self.pred_taw, delimiter=',', fmt='%.3f')
        #np.savetxt(os.path.join(self.out_path, "perstep_acc_taw.out"), self.perstep_acc_taw, delimiter=',', fmt='%.3f')

        
        np.savetxt(os.path.join(self.out_path, 'acc_tag.out'), self.acc_tag, delimiter=',', fmt='%.3f')
        np.savetxt(os.path.join(self.out_path, 'pred_tag.out'), self.pred_tag, delimiter=',', fmt='%.3f')
        #np.savetxt(os.path.join(self.out_path, 'perstep_acc_tag.out'), self.perstep_acc_tag, delimiter=',', fmt='%.3f')

        np.savetxt(os.path.join(self.out_path, 'avg_acc_taw.out'), self.avg_acc_taw, delimiter=',', fmt='%.3f')
        np.savetxt(os.path.join(self.out_path, 'avg_perstep_acc_taw.out'), self.avg_perstep_acc_taw, delimiter=',', fmt='%.3f')     

        
        np.savetxt(os.path.join(self.out_path, 'avg_acc_tag.out'), self.avg_acc_tag, delimiter=',', fmt='%.3f')
        np.savetxt(os.path.join(self.out_path, 'avg_perstep_acc_tag.out'),  self.avg_perstep_acc_tag, delimiter=',', fmt='%.3f')
 
        
        np.savetxt(os.path.join(self.out_path, 'forg_taw.out'), self.forg_taw, delimiter=',', fmt='%.3f')       
        np.savetxt(os.path.join(self.out_path, 'forg_tag.out'), self.forg_tag, delimiter=',', fmt='%.3f')

        np.savetxt(os.path.join(self.out_path, 'avg_forg_taw.out'), self.avg_forg_taw, delimiter=',', fmt='%.3f')
        np.savetxt(os.path.join(self.out_path, 'avg_forg_tag.out'), self.avg_forg_tag, delimiter=',', fmt='%.3f')
 



    



class FileOutputDuplicator(object):
    def __init__(self, duplicate, fname, mode):
        self.file = open(fname, mode)
        self.duplicate = duplicate

    def __del__(self):
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.duplicate.write(data)

    def flush(self):
        self.file.flush()


""" TODO: da modificare in maniera tale da gestire solo un validation e un test set, non multipli, perchè per ora
    solo uno è presente.
"""
class IncDecLogger():
    def __init__(self, out_path, n_task, task_dict, test_sizes, num_classes, begin_time=None, validation_mode=False) -> None:
        self.acc = np.zeros((n_task, n_task))
        self.mean_ap = np.zeros((n_task, n_task))
        self.map_weighted = np.zeros((n_task, n_task))
        self.ap = np.zeros((n_task, n_task, num_classes))
        #TODO: vedere se necessaria
        self.acc_per_class = np.zeros((n_task, n_task, num_classes))

        self.pred_acc = np.zeros((n_task, n_task))
        #TODO: vedere se con nuovo task_dict, da modificare cosa restituire
        self.task_len  =  [item for item in task_dict.values()]
        self.test_sizes = test_sizes
        self.best_epoch = np.full(n_task,-1)
        if validation_mode:
            self.out_path = os.path.join(out_path, "validation_logger")
        else:
            self.out_path = os.path.join(out_path, "logger")
        
        if begin_time is None:
            self.begin_time = datetime.now()
        else:
            self.begin_time = begin_time
        
        self.begin_time_str = self.begin_time.strftime("%Y-%m-%d-%H-%M")
        sys.stdout = FileOutputDuplicator(sys.stdout,
                                          os.path.join(out_path, 'stdout-{}.txt'.format(self.begin_time_str)), 'w')
        sys.stderr = FileOutputDuplicator(sys.stderr,
                                          os.path.join(out_path, 'stderr-{}.txt'.format(self.begin_time_str)), 'w')
        


    def update_accuracy(self, current_training_task_id, test_id, acc_value, ap_value, acc_per_class, mean_ap, map_weighted):
        self.acc[current_training_task_id, test_id] = acc_value * 100
        self.mean_ap[current_training_task_id, test_id] = mean_ap * 100
        self.map_weighted[current_training_task_id, test_id] = map_weighted * 100
        self.ap[current_training_task_id, test_id] = ap_value
        self.acc_per_class[current_training_task_id, test_id] = acc_per_class
         
  

    def print_latest(self, current_training_task_id, test_id):
        print('\n >>> Test on task {:2d} : acc={:5.1f}%, '
              ' |  <<<'.format(test_id, 
                                        self.acc[current_training_task_id, test_id]))


    def compute_average(self):
        self.avg_acc = self.acc.sum(1) / np.tril(np.ones(self.acc.shape[0])).sum(1)
            
           
          
    def print_file(self):
        np.savetxt(os.path.join(self.out_path, 'acc.out'), self.acc, delimiter=',', fmt='%.3f')
        np.savetxt(os.path.join(self.out_path, 'mean_ap.out'), self.mean_ap, delimiter=',', fmt='%.3f')
        np.savetxt(os.path.join(self.out_path, 'map_weighted.out'), self.map_weighted, delimiter=',', fmt='%.3f')
        np.savetxt(os.path.join(self.out_path, 'ap.out'), np.reshape(self.ap,(self.ap.shape[0]*self.ap.shape[1],-1)), delimiter=',', fmt='%.3f')
        np.savetxt(os.path.join(self.out_path, 'acc_per_class.out'), np.reshape(self.acc_per_class,(self.acc_per_class.shape[0]*self.acc_per_class.shape[1],-1)), delimiter=',', fmt='%.3f')
        np.savetxt(os.path.join(self.out_path, 'avg_acc.out'), self.avg_acc, delimiter=',', fmt='%.3f')


    def print_best_epoch(self, best_epoch, task_id):
        self.best_epoch[task_id] = best_epoch
        np.savetxt(os.path.join(self.out_path, 'best_epoch_tasks.out'), self.best_epoch, delimiter=',', fmt='%d')
 
