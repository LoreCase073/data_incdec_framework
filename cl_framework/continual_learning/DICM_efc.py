import torch
from tqdm import tqdm
from copy import deepcopy

from sklearn.utils import compute_class_weight

from continual_learning.IncrementalApproach import IncrementalApproach
from continual_learning.models.BaseModel import BaseModel
from continual_learning.metrics.metric_evaluator_incdec import MetricEvaluatorIncDec
from continual_learning.metrics.metric_evaluator_incdec_multilabel import MetricEvaluatorIncDec_multilabel
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import PrecisionRecallDisplay
import os
import pandas as pd
from torch import nn
from torch.utils.data import WeightedRandomSampler, SequentialSampler, DataLoader

def jacobian_in_batch(y, x):
        '''
        Compute the Jacobian matrix in batch form.
        Return (B, D_y, D_x)
        '''

        batch = y.shape[0]
        single_y_size = np.prod(y.shape[1:])
        y = y.view(batch, -1)
        vector = torch.ones(batch).to(y)

        # Compute Jacobian row by row.
        # dy_i / dx -> dy / dx
        # (B, D) -> (B, 1, D) -> (B, D, D)
        jac = [torch.autograd.grad(y[:, i], x, 
                                grad_outputs=vector, 
                                retain_graph=True,
                                create_graph=True)[0].view(batch, -1)
                    for i in range(single_y_size)]
        jac = torch.stack(jac, dim=1)
        
        return jac


def save_efm(cov, task_id, out_path):
    print("Saving Empirical Feature Matrix")
    torch.save(cov, os.path.join(out_path, "efm_task_{}.pt".format(task_id)))


def isPSD(A, tol=1e-7):
    A = A.cpu().numpy()
    E = np.linalg.eigvalsh(A)
    print("Maximum eigenvalue {}".format(np.max(E)))
    return np.all(E > -tol)


class DICM_efc(IncrementalApproach):
    
    def __init__(self, args, device, out_path, task_dict, total_classes, class_to_idx, subcategories_dict, all_subcategories_dict, multilabel,no_class_check):
        self.total_classes = total_classes
        
        self.n_accumulation = args.n_accumulation
        super().__init__(args, device, out_path, total_classes, task_dict)
        self.class_to_idx = class_to_idx
        self.class_names = list(class_to_idx.keys())
        self.model = BaseModel(backbone=self.backbone, dataset=args.dataset)
        self.old_model = None
        self.model.add_classification_head(self.total_classes)
        
        self.criterion_type = args.criterion_type
        self.criterion = self.select_criterion(args.criterion_type)
        self.all_subcategories_dict = all_subcategories_dict
        
        self.freeze_backbone = args.freeze_backbone
        self.efc_lambda = args.efc_lambda
        self.damping = args.damping
        self.previous_efm = None
        self.print_running_approach()
        # to check if working with multilabel with samples with no classses
        self.no_class_check = no_class_check
        self.multilabel = multilabel

    def print_running_approach(self):
        super(DICM_efc, self).print_running_approach()
        print("\n efc_hyperparams")
        print("- efc_lamb: {}".format(self.efc_lambda))
        print("- damping: {}".format(self.damping)) 
        

    def pre_train(self,  task_id, trn_loader, test_loader):
        self.model.to(self.device)

        self.old_model = deepcopy(self.model)
        self.old_model.freeze_all()
        self.old_model.to(self.device)
        self.old_model.eval()

        super(DICM_efc, self).pre_train(task_id)


    def reset_model(self):
        print('Reset model')
        self.model.reset_backbone()
        self.model.heads = nn.ModuleList()
        self.model.add_classification_head(self.total_classes)


    def substitute_head(self, num_classes):
        self.model.heads = nn.ModuleList()
        self.model.add_classification_head(num_classes)


    def train(self, task_id, train_loader, epoch, epochs):
        print(torch.cuda.current_device())
        self.model.to(self.device)
        self.model.train()
        
       
        train_loss, train_efc_loss, n_samples = 0, 0, 0
        self.optimizer.zero_grad()
        # if to work with loss accumulation, when batch size is too small
        if self.n_accumulation > 0:
            count_accumulation = 0
            for batch_idx, (images, targets, binarized_targets, _, _) in enumerate(tqdm(train_loader)):
                images = images.to(self.device)
                labels = self.select_proper_targets(targets, binarized_targets).to(self.device)
                current_batch_size = images.shape[0]
                n_samples += current_batch_size

                _, old_features = self.old_model(images)

                outputs, features = self.model(images)
                cls_loss, efc_loss = self.train_computation(outputs, labels, task_id, features, old_features, current_batch_size)
                loss = cls_loss + efc_loss

                loss.backward()
                train_loss += cls_loss.detach() * current_batch_size
                count_accumulation += 1
                if ((batch_idx + 1) % self.n_accumulation == 0) or (batch_idx + 1 == len(train_loader)):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
        else:
            for images, targets, binarized_targets, _, _ in  tqdm(train_loader):
                images = images.to(self.device)
                labels = self.select_proper_targets(targets, binarized_targets).to(self.device)
                current_batch_size = images.shape[0]
                n_samples += current_batch_size

                _, old_features = self.old_model(images)

                outputs, features = self.model(images)
                cls_loss, efc_loss = self.train_computation(outputs, labels, task_id, features, old_features, current_batch_size)
                loss = cls_loss + efc_loss
         
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                train_loss += cls_loss.detach() * current_batch_size
        
        self.train_log(task_id, epoch, train_loss/n_samples)  

    
    def train_computation(self, outputs, labels, task_id, features, old_features, current_batch_size):

        cls_loss, efc_loss = 0, 0

        if task_id > 0:
            efc_loss += self.efm_loss(features[:current_batch_size], old_features)
            cls_loss += self.criterion(outputs[0], labels)
        else:
            # first task only a cross entropy loss
            cls_loss += self.criterion(outputs[0], labels)

        return cls_loss, efc_loss
    

    def efm_loss(self, features, old_features):
        features = features.view(features.shape[0], -1)
        features = features.unsqueeze(1)
        old_features = old_features.view(old_features.shape[0], -1)
        old_features = old_features.unsqueeze(1)
        matrix_reg = self.efc_lambda *  self.previous_efm + self.damping * torch.eye(self.previous_efm.shape[0], device=self.device) 
        efc_loss = torch.mean(torch.bmm(torch.bmm((features - old_features), matrix_reg.expand(features.shape[0], -1, -1)), (features - old_features).permute(0,2,1)))
        return  efc_loss

    # cross entropy correct for our task of video classification
    def select_criterion(self, criterion_type):
        # here is 0 cause we only have one head
        if criterion_type == "multiclass":
            return torch.nn.CrossEntropyLoss()
        elif criterion_type == "multilabel":
            return torch.nn.BCEWithLogitsLoss()
        

    
        
    def post_train(self, task_id, train_loader=None):
        #create new Dataloader to do sequential sampling
        tmp_loader = DataLoader(train_loader.dataset, batch_size=train_loader.batch_size, shuffle=False, num_workers=train_loader.num_workers)

        n_samples_batches = len(tmp_loader.dataset) // tmp_loader.batch_size

        self.model.eval() 
        # ensure that gradients are zero
        self.model.zero_grad()   
        empirical_feat_mat = torch.zeros((self.model.get_feat_size(), self.model.get_feat_size()), requires_grad=False).to(self.device)
  
        for batch_idx, (images, _, _, _, _) in enumerate(tqdm(tmp_loader)):

            with torch.no_grad():
                _, gap_out = self.model(images.to(self.device))
            gap_out.requires_grad = True
            out = self.model.heads[0](gap_out)

            
            
            out_size = out.shape[1]
            
            log_p = nn.LogSigmoid()(out) # (n_batch,class)
            jac = jacobian_in_batch(log_p, gap_out).detach() 
            gap_out.detach()
       
            with torch.no_grad():
                p = torch.exp(log_p) # (n_batch,class)
                efm_per_batch = torch.zeros((images.shape[0], self.model.get_feat_size(), self.model.get_feat_size()), device=self.device)

                for c in range(out_size):
                        efm_per_batch +=  p[:,c].view(images.shape[0], 1, 1) * torch.bmm(jac[:,c, :].unsqueeze(1).permute(0,2,1), jac[:,c, :].unsqueeze(1)) 
                        
                empirical_feat_mat +=  torch.sum(efm_per_batch,dim=0)   


        n_samples = n_samples_batches * train_loader.batch_size
     
        # divide by the total number of samples 
        empirical_feat_mat = empirical_feat_mat/n_samples

        self.previous_efm = empirical_feat_mat
        if isPSD(self.previous_efm):
            print("EFM is semidefinite positive")

        # print rank of matrix
        matrix_rank = torch.linalg.matrix_rank(self.previous_efm)
        print("Matrix Rank: {}".format(matrix_rank))
        # save matrix after each task for analysis
        save_efm(self.previous_efm, task_id, self.out_path)


    def val_computation(self, outputs, labels, task_id, features, old_features, current_batch_size):

        cls_loss, efc_loss = 0, 0

        if task_id > 0:
            efc_loss = self.efm_loss(features[:current_batch_size], old_features)
            cls_loss = self.criterion(outputs[0], labels)
        else:
            # first task only a cross entropy loss
            cls_loss = self.criterion(outputs[0], labels)

        return cls_loss, efc_loss
    
    def eval(self, current_training_task, test_id, loader, epoch, verbose, testing=None):
        if self.multilabel:
            metric_evaluator = MetricEvaluatorIncDec_multilabel(self.out_path, self.total_classes, self.criterion_type, self.all_subcategories_dict, self.class_to_idx)
        else:
            metric_evaluator = MetricEvaluatorIncDec(self.out_path, self.total_classes, self.criterion_type, self.all_subcategories_dict, self.class_to_idx)

        
        val_cls_loss, val_efc_loss, n_samples = 0, 0, 0
        with torch.no_grad():
            self.model.eval()
            self.old_model.eval()
            for images, targets, binarized_targets, subcategory, data_path in tqdm(loader):
                images = images.to(self.device)

                labels = self.select_proper_targets(targets, binarized_targets).to(self.device)
                current_batch_size = images.shape[0]
                n_samples += current_batch_size

                _, old_features = self.old_model(images)

                outputs, features = self.model(images)
                # self.criterion(outputs[0], labels)   
                cls_loss, efc_loss = self.val_computation(outputs, labels, test_id, features, old_features, current_batch_size)

                loss = cls_loss + efc_loss
                 

                metric_evaluator.update(targets, binarized_targets.float().squeeze(dim=1),
                                        self.compute_probabilities(outputs, 0), subcategory, data_path)
                
                val_cls_loss += cls_loss * current_batch_size
                val_efc_loss += efc_loss * current_batch_size
                

            acc, ap, acc_per_class, mean_ap, map_weighted, precision_per_class, recall_per_class, exact_match, ap_per_subcategory, recall_per_subcategory, accuracy_per_subcategory, precision_per_subcategory = metric_evaluator.get(verbose=verbose)

            confusion_matrix, precision, recall = metric_evaluator.get_precision_recall_cm()
            
            
            
              
            

            if testing != None:
                self.save_error_analysis(current_training_task,
                                        self.class_names,
                                        metric_evaluator.get_data_paths(),
                                        metric_evaluator.get_predictions(),
                                        metric_evaluator.get_targets(),
                                        metric_evaluator.get_probabilities(),
                                        metric_evaluator.get_subcategories(),
                                        testing
                                        )
                if self.criterion_type == "multiclass":
                    cm_figure = self.plot_confusion_matrix(confusion_matrix, self.class_names)
                elif self.criterion_type == "multilabel":
                    cm_figure = None

                pr_figure = self.plot_pr_curve(precision, recall)
            else:
                cm_figure = None
                pr_figure = None


            self.log(current_training_task, test_id, epoch, cls_loss/n_samples, acc, mean_ap, map_weighted, confusion_matrix, cm_figure, pr_figure, testing)


            if verbose:
                print(" - classification loss: {}".format(val_cls_loss/n_samples))
                print(" - efc loss: {}".format(val_efc_loss/n_samples))

            return acc, ap, cls_loss/n_samples, acc_per_class, mean_ap, map_weighted, precision_per_class, recall_per_class, exact_match, ap_per_subcategory, recall_per_subcategory, accuracy_per_subcategory, precision_per_subcategory
        

    def log(self, current_training_task, test_id, epoch, cls_loss , acc, mean_ap, map_weighted, confusion_matrix, cm_figure, pr_figure, testing):

        if testing == None:
            name_tb = "training_task_" + str(current_training_task) + "/dataset_" + str(test_id) + "_classification_loss"
            self.logger.add_scalar(name_tb, cls_loss, epoch)

            name_tb = "training_task_" + str(current_training_task) + "/dataset_" + str(test_id) + "_accuracy"
            self.logger.add_scalar(name_tb, acc, epoch)

            name_tb = "training_task_" + str(current_training_task) + "/dataset_" + str(test_id) + "_mAP"
            self.logger.add_scalar(name_tb, mean_ap, epoch)

            name_tb = "training_task_" + str(current_training_task) + "/dataset_" + str(test_id) + "_weighted_mAP"
            self.logger.add_scalar(name_tb, map_weighted, epoch)

        elif testing == 'val':
            name_tb = "validation_dataset_"+ str(current_training_task) + "/task_" + str(test_id) + "_accuracy"
            self.logger.add_scalar(name_tb, acc, epoch)

            name_tb = "validation_dataset_"+ str(current_training_task) + "/task_" + str(test_id) + "_mAP"
            self.logger.add_scalar(name_tb, mean_ap, epoch)

            name_tb = "validation_dataset_"+ str(current_training_task) + "/task_" + str(test_id) + "_weighted_mAP"
            self.logger.add_scalar(name_tb, map_weighted, epoch)

            if cm_figure is not None:
                name_tb = "validation_dataset_"+ str(current_training_task) + "/task_" + str(test_id) + "_confusion_matrix"
                self.logger.add_figure(name_tb,cm_figure,epoch)

            name_tb = "validation_dataset_"+ str(current_training_task) + "/task_" + str(test_id) + "_pr_curve"
            self.logger.add_figure(name_tb,pr_figure,epoch)

            #saving confusion matrices
            if self.criterion_type == "multiclass":
                cm_path = os.path.join(self.out_path,'confusion_matrices')
                if not os.path.exists(cm_path):
                    os.mkdir(cm_path)
                cm_name_path = os.path.join(cm_path,"task_{}_validation_cm.npy".format(test_id))
                cm_name_path_txt = os.path.join(cm_path,"task_{}_validation_cm.out".format(test_id))
                np.save(cm_name_path,confusion_matrix)
                np.savetxt(cm_name_path_txt,confusion_matrix, delimiter=',', fmt='%d')
            elif self.criterion_type == "multilabel":
                # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.multilabel_confusion_matrix.html
                # matrix will have the form:
                """ 
                TN FP
                FN TP
                """
                cm_path = os.path.join(self.out_path,'confusion_matrices')
                if not os.path.exists(cm_path):
                    os.mkdir(cm_path)
                for i in range(self.total_classes):
                    cm_name_path = os.path.join(cm_path,"task_{}_class_{}_validation_cm.npy".format(test_id,i))
                    cm_name_path_txt = os.path.join(cm_path,"task_{}_class_{}_validation_cm.out".format(test_id,i))
                    np.save(cm_name_path,confusion_matrix[i])
                    np.savetxt(cm_name_path_txt,confusion_matrix[i], delimiter=',', fmt='%d')

            
        elif testing == 'test':
            name_tb = "test_dataset_"+ str(current_training_task)+ "/task_" + str(test_id) + "_accuracy"
            self.logger.add_scalar(name_tb, acc, epoch)

            name_tb = "test_dataset_"+ str(current_training_task) + "/task_" + str(test_id) + "_mAP"
            self.logger.add_scalar(name_tb, mean_ap, epoch)

            name_tb = "test_dataset_"+ str(current_training_task) + "/task_" + str(test_id) + "_weighted_mAP"
            self.logger.add_scalar(name_tb, map_weighted, epoch)

            if cm_figure is not None:
                name_tb = "test_dataset_"+ str(current_training_task) + "/task_" + str(test_id) + "_confusion_matrix"
                self.logger.add_figure(name_tb,cm_figure,epoch)

            name_tb = "test_dataset_"+ str(current_training_task) + "/task_" + str(test_id) + "_pr_curve"
            self.logger.add_figure(name_tb,pr_figure,epoch)

            #saving confusion matrices
            if self.criterion_type == "multiclass":
                cm_path = os.path.join(self.out_path,'confusion_matrices')
                if not os.path.exists(cm_path):
                    os.mkdir(cm_path)
                cm_name_path = os.path.join(cm_path,"task_{}_test_cm.npy".format(test_id))
                cm_name_path_txt = os.path.join(cm_path,"task_{}_test_cm.out".format(test_id))
                np.save(cm_name_path,confusion_matrix)
                np.savetxt(cm_name_path_txt,confusion_matrix, delimiter=',', fmt='%d')
            elif self.criterion_type == "multilabel":
                # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.multilabel_confusion_matrix.html
                # matrix will have the form:
                """ 
                TN FP
                FN TP
                """
                cm_path = os.path.join(self.out_path,'confusion_matrices')
                if not os.path.exists(cm_path):
                    os.mkdir(cm_path)
                for i in range(self.total_classes):
                    cm_name_path = os.path.join(cm_path,"task_{}_class_{}_test_cm.npy".format(test_id,i))
                    cm_name_path_txt = os.path.join(cm_path,"task_{}_class_{}_test_cm.out".format(test_id,i))
                    np.save(cm_name_path,confusion_matrix[i])
                    np.savetxt(cm_name_path_txt,confusion_matrix[i], delimiter=',', fmt='%d')
        

    def train_log(self, current_training_task, epoch, cls_loss):
        name_tb = "training_task_" + str(current_training_task) + "/training_classification_loss"
        self.logger.add_scalar(name_tb, cls_loss, epoch)



    def compute_probabilities(self, outputs, head_id):
        if self.criterion_type == "multiclass":
            probabilities = torch.nn.Softmax(dim=1)(outputs[head_id])
        else:
            probabilities = torch.nn.Sigmoid()(outputs[head_id])
        return probabilities
    
    def select_proper_targets(self, targets, binarized_targets):
        if self.criterion_type == "multiclass":
            return targets
        elif self.criterion_type == "multilabel":
            return binarized_targets.float().squeeze(dim=1)

    def plot_confusion_matrix(self, cm, class_names):
        """
        Returns a matplotlib figure containing the plotted confusion matrix.

        Args:
            cm (array, shape = [n, n]): a confusion matrix of integer classes
            class_names (array, shape = [n]): String names of the integer classes
        """
        figure = plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        # Compute the labels from the normalized confusion matrix.
        labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        return figure
    

    def plot_pr_curve(self, precision, recall):
        """
        Returns a matplotlib figure containing the plotted confusion matrix.
        """
        figure, ax = plt.subplots(figsize=(8, 8))
        for i in range(self.total_classes):
            display = PrecisionRecallDisplay(
                recall=recall[i],
                precision=precision[i]
            )
            display.plot(ax=ax, name=f"Precision-recall for class {self.class_names[i]}")
        return figure
    

    def save_error_analysis(self, task_id, class_names, data_paths, predictions, targets, probabilities, subcategory, testing):
        ea_path = os.path.join(self.out_path,'error_analysis')
        if not os.path.exists(ea_path):
            os.mkdir(ea_path)
        
        if testing == 'val':
            name_file = os.path.join(ea_path,"task_{}_validation_error_analysis.csv".format(task_id))
        elif testing == 'test':
            name_file = os.path.join(ea_path,"task_{}_test_error_analysis.csv".format(task_id))

        probs = {}
        for i in range(self.total_classes):
            probs[class_names[i]] = probabilities[:,i]
        
        if self.criterion_type == "multilabel":
            binary_targets = {}
            for i in range(self.total_classes):
                binary_targets["target_" + class_names[i]] = targets[:,i]
            
            d = {'video_path':data_paths, 'prediction':predictions, 'subcategory': subcategory}
            unified_dict = d | probs | binary_targets
        elif self.criterion_type == "multiclass":
            d = {'video_path':data_paths, 'prediction':predictions, 'target':targets, 'subcategory': subcategory}
            unified_dict = d | probs
        df = pd.DataFrame(unified_dict)
        df.to_csv(name_file, index=False)