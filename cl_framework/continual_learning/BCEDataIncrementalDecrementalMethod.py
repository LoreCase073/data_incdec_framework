import torch
from tqdm import tqdm
from copy import deepcopy

from sklearn.utils import compute_class_weight

from continual_learning.IncrementalApproach import IncrementalApproach
from continual_learning.models.BaseModel import BaseModel
from continual_learning.metrics.metric_evaluator_incdec import BCEMetricEvaluatorIncDec
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import PrecisionRecallDisplay
import os
import pandas as pd
 

#TODO: vedere se ereditare da IncrementalApproach ha senso e se modificare qualcosa
class BCEDataIncrementalDecrementalMethod(IncrementalApproach):
    #TODO: modificare init, non necessito di class_per_task probabilmente, non so di task_dict
    
    def __init__(self, args, device, out_path, task_dict, total_classes, behaviors_per_task, behavior_dicts):
        #TODO: class_per_task, come passarlo
        self.total_classes = total_classes
        self.imbalanced = args.imbalanced
        self.loss_accumulation = args.accumulation
        self.n_accumulation = args.n_accumulation
        super().__init__(args, device, out_path, total_classes, task_dict)
        #TODO: vedere se da BaseModel necessito di modificare qualcosa in caso
        self.class_names = list(behavior_dicts[0].keys())

        self.model = BaseModel(backbone=self.backbone, dataset=args.dataset)
        #TODO: forse modificare come aggiungere head, tanto la si crea una sola volta
        self.model.add_classification_head(self.total_classes)
        self.print_running_approach()


    def print_running_approach(self):
        super(BCEDataIncrementalDecrementalMethod, self).print_running_approach()
        
    #TODO: modifica info necessarie per impostare parametri per il pretraining
    def pre_train(self,  task_id, trn_loader, test_loader):
        self.model.to(self.device)
        # necessary only for tsne 
        self.old_model = deepcopy(self.model)
        self.old_model.eval()
        self.old_model.to(self.device)
        
        #TODO: modificare, forse non prendere quello precedente e cambiare direttamente la logica
        #perchè 2 tipi di approcci diversi
        #TODO: forse da non modificare nulla, ma non so per la logica delle teste multiple...
        super(BCEDataIncrementalDecrementalMethod, self).pre_train(task_id)

    #TODO: forse non necessario da cambiare, controllare...
    def train(self, task_id, train_loader, epoch, epochs):
        print(torch.cuda.current_device())
        self.model.to(self.device)
        self.model.train()
        
       
        train_loss, n_samples = 0, 0
        self.optimizer.zero_grad()
        #if to work with loss accumulation, when batch size is too small
        if self.loss_accumulation:
            count_accumulation = 0
            for batch_idx, (images, targets, _, _) in enumerate(tqdm(train_loader)):
                images = images.to(self.device)
                targets = targets.to(self.device)
                current_batch_size = images.shape[0]
                n_samples += current_batch_size
                outputs, _ = self.model(images)
                loss = self.criterion(outputs, targets)             
                loss.backward()
                train_loss += loss * current_batch_size
                count_accumulation += 1
                if ((batch_idx + 1) % self.n_accumulation == 0) or (batch_idx + 1 == len(train_loader)):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
        else:
            for images, targets, _, _ in  tqdm(train_loader):
                images = images.to(self.device)
                targets = targets.to(self.device)
                current_batch_size = images.shape[0]
                n_samples += current_batch_size
                outputs, _ = self.model(images)
                loss = self.criterion(outputs, targets)             
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                train_loss += loss * current_batch_size
        
        self.train_log(task_id, epoch, train_loss/n_samples)  

    # for multilabel i'll have to use BCE loss
    def criterion(self, outputs, targets):
        #TODO: rescale targets non dovrebbe servire...
        #targets = self.rescale_targets(targets, t)
        #here is 0 cause we only have a head
        return torch.nn.functional.binary_cross_entropy_with_logits(outputs[0], targets)
        
        
        
    def post_train(self, task_id, trn_loader=None):
        pass 

    
    #magari ci sono già implementati metodi interessanti
    def eval(self, current_training_task, test_id, loader, epoch, verbose, testing=None):
        metric_evaluator = BCEMetricEvaluatorIncDec(self.out_path, self.task_dict, self.total_classes)

        
        cls_loss, n_samples = 0, 0 
        with torch.no_grad():
            self.model.eval()
            for images, targets, behavior, data_path in tqdm(loader):
                images = images.to(self.device)
                targets = targets.type(dtype=torch.int64).to(self.device)
                current_batch_size = images.shape[0]
                n_samples += current_batch_size
 
                outputs, features = self.model(images)
                #_, old_features = self.old_model(images)
                
                cls_loss += self.criterion(outputs, targets) * current_batch_size
                 

                metric_evaluator.update(targets, 
                                        self.compute_probabilities(outputs, 0), behavior, data_path)
                

            acc, ap, acc_per_class, mean_ap, map_weighted = metric_evaluator.get(verbose=verbose)

            confusion_matrix, precision, recall = metric_evaluator.get_precision_recall_cm()
            
            

            cm_figure = self.plot_confusion_matrix(confusion_matrix, self.class_names)

            pr_figure = self.plot_pr_curve(precision, recall)
              
            self.log(current_training_task, test_id, epoch, cls_loss/n_samples, acc, mean_ap, map_weighted, confusion_matrix, cm_figure, pr_figure, testing)

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
                """ self.save_ap_classes(current_training_task,
                                      self.class_names,
                                      ap,
                                      testing) """
            if verbose:
                print(" - classification loss: {}".format(cls_loss/n_samples))

            return acc, ap, cls_loss/n_samples, acc_per_class, mean_ap, map_weighted
        

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

            name_tb = "validation_dataset_"+ str(current_training_task) + "/task_" + str(test_id) + "_confusion_matrix"
            self.logger.add_figure(name_tb,cm_figure,epoch)

            name_tb = "validation_dataset_"+ str(current_training_task) + "/task_" + str(test_id) + "_pr_curve"
            self.logger.add_figure(name_tb,pr_figure,epoch)


            #saving confusion matrices
            cm_path = os.path.join(self.out_path,'confusion_matrices')
            if not os.path.exists(cm_path):
                os.mkdir(cm_path)
            cm_name_path = os.path.join(cm_path,"task_{}_validation_cm.npy".format(test_id))
            cm_name_path_txt = os.path.join(cm_path,"task_{}_validation_cm.out".format(test_id))
            np.save(cm_name_path,confusion_matrix)
            np.savetxt(cm_name_path_txt,confusion_matrix)
            
        elif testing == 'test':
            name_tb = "test_dataset_"+ str(current_training_task)+ "/task_" + str(test_id) + "_accuracy"
            self.logger.add_scalar(name_tb, acc, epoch)

            name_tb = "test_dataset_"+ str(current_training_task) + "/task_" + str(test_id) + "_mAP"
            self.logger.add_scalar(name_tb, mean_ap, epoch)

            name_tb = "test_dataset_"+ str(current_training_task) + "/task_" + str(test_id) + "_weighted_mAP"
            self.logger.add_scalar(name_tb, map_weighted, epoch)

            name_tb = "test_dataset_"+ str(current_training_task) + "/task_" + str(test_id) + "_confusion_matrix"
            self.logger.add_figure(name_tb,cm_figure,epoch)

            name_tb = "test_dataset_"+ str(current_training_task) + "/task_" + str(test_id) + "_pr_curve"
            self.logger.add_figure(name_tb,pr_figure,epoch)

            #saving confusion matrices
            cm_path = os.path.join(self.out_path,'confusion_matrices')
            if not os.path.exists(cm_path):
                os.mkdir(cm_path)
            cm_name_path = os.path.join(cm_path,"task_{}_test_cm.npy".format(test_id))
            cm_name_path_txt = os.path.join(cm_path,"task_{}_test_cm.out".format(test_id))
            np.save(cm_name_path,confusion_matrix)
            np.savetxt(cm_name_path_txt,confusion_matrix)
        
        

    def train_log(self, current_training_task, epoch, cls_loss):
        name_tb = "training_task_" + str(current_training_task) + "/training_classification_loss"
        self.logger.add_scalar(name_tb, cls_loss, epoch)

    # for multilabel, we use Sigmoid instead of Softmax
    def compute_probabilities(self, outputs, head_id):
      probabilities = torch.nn.Sigmoid(dim=1)(outputs[head_id])
      return probabilities
    

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

        Args:
            cm (array, shape = [n, n]): a confusion matrix of integer classes
            class_names (array, shape = [n]): String names of the integer classes
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
        for i in range(len(class_names)):
            probs[class_names[i]] = probabilities[:,i]

        d = {'video_path':data_paths, 'prediction':predictions, 'target':targets, 'subcategory': subcategory}
        unified_dict = d | probs
        df = pd.DataFrame(unified_dict)
        df.to_csv(name_file, index=False)


    """ def save_ap_classes(self, task_id, class_names, ap, testing):
        ea_path = os.path.join(self.out_path,'mAP')
        if not os.path.exists(ea_path):
            os.mkdir(ea_path)
        
        if testing == 'val':
            name_file = os.path.join(ea_path,"task_{}_meanAP_validation_per_class.csv".format(task_id))
        elif testing == 'test':
            name_file = os.path.join(ea_path,"task_{}_meanAP__test_per_class.csv".format(task_id))

        

        df = pd.DataFrame(ap.numpy(), columns=class_names)
        df.to_csv(name_file, index=False) """