import torch
from tqdm import tqdm
from copy import deepcopy

from continual_learning.IncrementalApproach import IncrementalApproach
from continual_learning.models.BaseModel import BaseModel
from continual_learning.metrics.metric_evaluator import MetricEvaluator
 


class FineTuningMethod(IncrementalApproach):
    def __init__(self, args, device, out_path, class_per_task, task_dict):
        super().__init__(args, device, out_path, class_per_task, task_dict)
        self.model = BaseModel(backbone=self.backbone, dataset=args.dataset)
        self.print_running_approach()


    def print_running_approach(self):
        super(FineTuningMethod, self).print_running_approach()
        

    def pre_train(self,  task_id, trn_loader, test_loader):
        self.model.add_classification_head(len(self.task_dict[task_id]))
        self.model.to(self.device)
        # necessary only for tsne 
        self.old_model = deepcopy(self.model)
        self.old_model.eval()
        self.old_model.to(self.device)
        
        super(FineTuningMethod, self).pre_train(task_id)


    def train(self, task_id, train_loader, epoch, epochs):
        self.model.to(self.device)
        self.model.train()
        
       
            
        for images, targets in  tqdm(train_loader):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
 

            outputs, _ = self.model(images)
            
            loss = self.criterion(outputs, targets, task_id)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


    def criterion(self, outputs, targets, t):
        targets = self.rescale_targets(targets, t)
        return torch.nn.functional.cross_entropy(outputs[t], targets)
        
        
        
    def post_train(self, task_id, trn_loader=None):
        pass 


    def eval(self, current_training_task, test_id, loader, epoch,   verbose):
        metric_evaluator = MetricEvaluator(self.out_path, self.task_dict)
        
        cls_loss, n_samples = 0, 0 
   
        with torch.no_grad():
            self.model.eval()
            for images, targets  in  loader:
                images = images.to(self.device)
                targets = targets.type(dtype=torch.int64).to(self.device)
                current_batch_size = images.shape[0]
                n_samples += current_batch_size
 
                outputs, features = self.model(images)
                _, old_features = self.old_model(images)
                
                cls_loss += self.criterion(outputs, targets, test_id) * current_batch_size
                 

                metric_evaluator.update(targets, self.rescale_targets(targets, test_id), 
                                        self.tag_probabilities(outputs), 
                                        self.taw_probabilities(outputs, test_id),
                                        )

         
            taw_acc,  tag_acc  = metric_evaluator.get(verbose=verbose)
 
              
            self.log(current_training_task, test_id, epoch, cls_loss/n_samples, tag_acc , taw_acc)          
            
            if verbose:
                print(" - classification loss: {}".format(cls_loss/n_samples))

            return taw_acc, tag_acc, cls_loss/n_samples
        
    
    def log(self, current_training_task, test_id, epoch, cls_loss,   tag_acc , taw_acc):
        name_tb = "training_task_" + str(current_training_task) + "/dataset_" + str(test_id) + "_classification_loss"
        self.logger.add_scalar(name_tb, cls_loss, epoch)


        name_tb = "training_task_" + str(current_training_task) + "/dataset_" + str(test_id) + "_TAG_accuracy "
        self.logger.add_scalar(name_tb, tag_acc, epoch)

        name_tb = "training_task_" + str(current_training_task) + "/dataset_" + str(test_id) + "_TAW_accuracy"
        self.logger.add_scalar(name_tb, taw_acc, epoch)