import torch
from tqdm import tqdm
from copy import deepcopy

from continual_learning.IncrementalApproach import IncrementalApproach
from continual_learning.models.BaseModel import BaseModel
from continual_learning.metrics.metric_evaluator import MetricEvaluator

class LWF(IncrementalApproach):
    def __init__(self, args, device, out_path, class_per_task, task_dict ):
        
        super(LWF, self).__init__(args, device, out_path, class_per_task, task_dict )
        self.model = BaseModel(backbone=self.backbone, dataset=args.dataset)

        # Lwf args 
        self.T = args.lwf_T
        self.lwf_lamb = args.lwf_lamb
        
        self.print_running_approach()

    def print_running_approach(self):
        super(LWF, self).print_running_approach()
        print("- temperature: {}".format(self.T))
        print("- lambda lwf: {}".format(self.lwf_lamb))
 

    def pre_train(self,  task_id, trn_loader, test_loader):
        self.model.add_classification_head(len(self.task_dict[task_id]))
        self.model.to(self.device)
        # Restore best and save model for future tasks
        self.old_model = deepcopy(self.model)
        self.old_model.eval()
        self.old_model.freeze_all()
        self.old_model.to(self.device)
        
        super(LWF, self).pre_train(task_id)

    
    def post_train(self, task_id, trn_loader):
        pass 


    def train(self, task_id, trn_loader, epoch, epochs):
        self.model.train()

     
 
        for images, targets in tqdm(trn_loader):
            images = images.to(self.device)
            targets = targets.to(self.device)
  
            
            outputs_old = None
            
            # Forward old model
            if task_id > 0:
                outputs_old, _= self.old_model(images)

            # Forward current model
            outputs, _ = self.model(images)

            cls_loss, lwf_loss = self.criterion(outputs, targets, task_id, outputs_old, test_id=task_id)
            loss = cls_loss + lwf_loss

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
 

    def cross_entropy(self, outputs, targets, exp=1.0, size_average=True, eps=1e-5):
        """Calculates cross-entropy with temperature scaling"""
        out = torch.nn.functional.softmax(outputs, dim=1)
        tar = torch.nn.functional.softmax(targets, dim=1)
        if exp != 1:
            out = out.pow(exp)
            out = out / out.sum(1).view(-1, 1).expand_as(out)
            tar = tar.pow(exp)
            tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
        out = out + eps / out.size(1)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        ce = -(tar * out.log()).sum(1)
        if size_average:
            ce = ce.mean()
        return ce

    
    def criterion(self, outputs, targets, t, outputs_old, test_id=None):
        """Returns the loss value"""
        cls_loss, lwf_loss = 0, 0
        if t > 0:
            # Knowledge distillation loss for all previous tasks
            lwf_loss += self.lwf_lamb * self.cross_entropy(torch.cat(list(outputs.values())[:t], dim=1),
                                                   torch.cat(list(outputs_old.values())[:t], dim=1), exp=1.0 / self.T)
       
        # Current cross-entropy loss -- with exemplars use all heads
     
        cls_loss += torch.nn.functional.cross_entropy(outputs[test_id], self.rescale_targets(targets, test_id))  
        
        return cls_loss, lwf_loss
    
        
    def eval(self, current_training_task, test_id, loader, epoch,  verbose):
        metric_evaluator = MetricEvaluator(self.out_path, self.task_dict)

        cls_loss, lwf_loss  = 0, 0 
        n_samples = 0

        with torch.no_grad():
            self.old_model.eval()
            self.model.eval()
            
            for images, targets in loader:
                images = images.to(self.device)
                targets = targets.type(dtype=torch.int64).to(self.device)
                
                current_batch_size = images.shape[0]
                n_samples += current_batch_size 
                
                original_labels = deepcopy(targets)
                
           
                outputs, _ = self.model(images)
                outputs_old, _ = self.old_model(images)
                
   
                

                cls_loss_batch, lwf_loss_batch = self.criterion(outputs, targets, current_training_task, outputs_old, test_id=test_id)
                cls_loss += cls_loss_batch * current_batch_size
                lwf_loss += lwf_loss_batch * current_batch_size
                
                metric_evaluator.update(original_labels, self.rescale_targets(targets, test_id), 
                                        self.tag_probabilities(outputs), 
                                        self.taw_probabilities(outputs, test_id),
                                        )
            
   
            taw_acc,  tag_acc = metric_evaluator.get(verbose=verbose)
            
            if current_training_task > 0:
                overall_loss = cls_loss/n_samples + lwf_loss/n_samples  
            else:
                overall_loss = cls_loss/n_samples
                
         
            if current_training_task > 0:
            
                self.log(current_training_task, test_id, epoch, cls_loss/n_samples,
                            lwf_loss/n_samples , tag_acc, taw_acc) 
            else:
                
                self.log(current_training_task, test_id, epoch, cls_loss/n_samples, 0.0,  tag_acc, taw_acc) 
                    
            if verbose:
                print(" - classification loss: {}".format(cls_loss/n_samples))
                if current_training_task > 0:
                    print(" - lwf loss: {}".format(lwf_loss/n_samples))
 
        
                
            return taw_acc,  tag_acc, overall_loss 
        
    
    def log(self, current_training_task, test_id, epoch,  cls_loss, lwf_loss,  tag_acc, taw_acc):
        name_tb = "training_task_" + str(current_training_task) + "/dataset_" + str(test_id) + "_classification_loss"
        self.logger.add_scalar(name_tb, cls_loss, epoch)

        name_tb = "training_task_" + str(current_training_task) + "/dataset_" + str(test_id) + "_TAG_accuracy"
        self.logger.add_scalar(name_tb, tag_acc, epoch)

        name_tb = "training_task_" + str(current_training_task) + "/dataset_" + str(test_id) + "_TAW_accuracy"
        self.logger.add_scalar(name_tb, taw_acc, epoch)
        
        if current_training_task > 0:
            name_tb = "training_task_" + str(current_training_task) + "/dataset_" + str(test_id) + "_lwf_loss"
            self.logger.add_scalar(name_tb, lwf_loss, epoch)