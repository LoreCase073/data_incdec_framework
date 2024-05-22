import torch
from tqdm import tqdm
from copy import deepcopy

from continual_learning.IncrementalApproach import IncrementalApproach
from continual_learning.models.BaseModel import BaseModel
from continual_learning.metrics.metric_evaluator import MetricEvaluator
from continual_learning.utils.empirical_fisher import EmpiricalFIM
from torch import nn 

class ElasticWeightConsolidation(IncrementalApproach):
    def __init__(self, args, device, out_path, class_per_task, task_dict):
        
        super(ElasticWeightConsolidation, self).__init__(args, device, out_path, class_per_task, task_dict)
        self.model = BaseModel(backbone=self.backbone,dataset=args.dataset)
        self.lamb = 5000
        self.alpha = 0.5
        self.fisher = None
        self.older_params = None
        self.print_running_approach()

    def print_running_approach(self):
        super(ElasticWeightConsolidation, self).print_running_approach()
        print("- lambda ewc: {}".format(self.lamb))
        print("- alpha ewc: {}".format(self.alpha))
 

    def pre_train(self,  task_id, trn_loader, test_loader):
        self.model.add_classification_head(len(self.task_dict[task_id]))
        self.model.to(self.device)
        if  task_id == 0:
            self.auxiliary_classifier = nn.Linear(512, len(self.task_dict[task_id]) * 3 )
            self.auxiliary_classifier.to(self.device)
        else:
            self.auxiliary_classifier = None 
        
        super(ElasticWeightConsolidation, self).pre_train(task_id)

    
    def post_train(self, task_id, trn_loader):
        fisher_matrix = EmpiricalFIM(self.device, self.out_path)
        fisher_matrix.compute(self.model, trn_loader, task_id)
        if task_id == 0:
            self.fisher = fisher_matrix.get()
        else:
            current_fisher = fisher_matrix.get()
            for n in self.fisher.keys():
                self.fisher[n] = (self.alpha * self.fisher[n] + (1 - self.alpha) * current_fisher[n])
                
        self.older_params = {n: p.clone().detach() for n, p in self.model.backbone.named_parameters() if p.requires_grad}

        


    def train(self, task_id, trn_loader, epoch, epochs):
        self.model.train()
 
        for images, targets in tqdm(trn_loader):
            images = images.to(self.device)
            targets = targets.to(self.device)

            if task_id == 0:
                images_rot = torch.stack([torch.rot90(images, k, (2, 3)) for k in range(1, 4)], 1)
                images_rot = images_rot.view(-1, 3, self.image_size, self.image_size)
                targets_rot = torch.stack([(targets + len(self.task_dict[task_id])) + 2 * (targets - min(self.task_dict[task_id])) + k for k in range(0, 3)], 1).view(-1)
                targets = torch.cat([targets, targets_rot], dim=0)
                images = torch.cat([images, images_rot], dim=0)

            # Forward current model
            outputs, features  = self.model(images)

            if task_id == 0:
                out_rot = self.auxiliary_classifier(features)
                outputs[task_id] = torch.cat([outputs[task_id], out_rot],axis=1)


            cls_loss, ewc_loss = self.criterion(outputs, targets, task_id, test_id=task_id)
    
            loss = cls_loss + ewc_loss 
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    
    def criterion(self, outputs, targets, t,   test_id=None):
        """Returns the loss value"""
        loss = 0
        ewc_loss = 0
        if t > 0:
      
            # Eq. 3: elastic weight consolidation quadratic penalty
            for n, p in self.model.backbone.named_parameters():
                if n in self.fisher.keys():
                    ewc_loss += torch.sum(self.fisher[n] * (p - self.older_params[n]).pow(2)) / 2
            loss += self.lamb * ewc_loss
            
        return loss + torch.nn.functional.cross_entropy(outputs[test_id], self.rescale_targets(targets, test_id)), self.lamb*ewc_loss
    
        
    def eval(self, current_training_task, test_id, loader, epoch,  verbose):
        metric_evaluator = MetricEvaluator(self.out_path, self.task_dict)

        cls_loss, ewc_loss  = 0, 0 
        n_samples = 0
        with torch.no_grad():
            
            self.model.eval()
            
            for images, targets in loader:
                images = images.to(self.device)
                targets = targets.type(dtype=torch.int64).to(self.device)
                
                current_batch_size = images.shape[0]
                n_samples += current_batch_size 
                
                original_labels = deepcopy(targets)
                
           
                outputs, features = self.model(images)
            

                cls_loss_batch, ewc_loss = self.criterion(outputs, targets, current_training_task, test_id=test_id)
                
                cls_loss += cls_loss_batch * current_batch_size
                
                metric_evaluator.update(original_labels, self.rescale_targets(targets, test_id), 
                                        self.tag_probabilities(outputs), 
                                        self.taw_probabilities(outputs, test_id),
                                        )
            
 
            taw_acc,  tag_acc = metric_evaluator.get(verbose=verbose)
            
            if current_training_task > 0:
                overall_loss = cls_loss/n_samples + ewc_loss
            else:
                overall_loss = cls_loss/n_samples
                
 
            if verbose:
                print(" - classification loss: {}".format(cls_loss/n_samples))
                if current_training_task > 0:
                    print(" - ewc loss: {}".format(ewc_loss))
 
    
            return taw_acc,  tag_acc, overall_loss      
        