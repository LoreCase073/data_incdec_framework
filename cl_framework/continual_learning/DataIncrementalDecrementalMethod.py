import torch
from tqdm import tqdm
from copy import deepcopy

from sklearn.utils import compute_class_weight

from continual_learning.IncrementalApproach import IncrementalApproach
from continual_learning.models.BaseModel import BaseModel
from continual_learning.metrics.metric_evaluator_incdec import MetricEvaluatorIncDec
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import PrecisionRecallDisplay
 

#TODO: vedere se ereditare da IncrementalApproach ha senso e se modificare qualcosa
class DataIncrementalDecrementalMethod(IncrementalApproach):
    #TODO: modificare init, non necessito di class_per_task probabilmente, non so di task_dict
    
    def __init__(self, args, device, out_path, task_dict, total_classes, behaviors_per_task, behavior_dicts, imbalanced=True):
        #TODO: class_per_task, come passarlo
        self.total_classes = total_classes
        self.imbalanced = imbalanced
        super().__init__(args, device, out_path, total_classes, task_dict)
        #TODO: vedere se da BaseModel necessito di modificare qualcosa in caso
        self.class_names = list(behavior_dicts[0].keys())

        self.model = BaseModel(backbone=self.backbone, dataset=args.dataset)
        #TODO: forse modificare come aggiungere head, tanto la si crea una sola volta
        self.model.add_classification_head(self.total_classes)
        self.print_running_approach()


    def print_running_approach(self):
        super(DataIncrementalDecrementalMethod, self).print_running_approach()
        
    #TODO: modifica info necessarie per impostare parametri per il pretraining
    def pre_train(self,  task_id, trn_loader, test_loader):
        #TODO: aggiunta della classification head non penso debba dipendere dal task_id, siccome
        #il numero di classi non cambia
        #self.model.add_classification_head(len(self.task_dict[task_id]))
        self.model.to(self.device)
        # necessary only for tsne 
        self.old_model = deepcopy(self.model)
        self.old_model.eval()
        self.old_model.to(self.device)
        
        #TODO: modificare, forse non prendere quello precedente e cambiare direttamente la logica
        #perchè 2 tipi di approcci diversi
        #TODO: forse da non modificare nulla, ma non so per la logica delle teste multiple...
        super(DataIncrementalDecrementalMethod, self).pre_train(task_id)

    #TODO: forse non necessario da cambiare, controllare...
    def train(self, task_id, train_loader, epoch, epochs):
        print(torch.cuda.current_device())
        self.model.to(self.device)
        self.model.train()

        if self.imbalanced:
            class_weights = torch.Tensor(compute_class_weight('balanced', classes=np.unique(train_loader.dataset.dataset.targets), 
                                                              y=train_loader.dataset.dataset.targets)).to(self.device)
        else:
            class_weights=None
        
       
        train_loss, n_samples = 0, 0
        for images, targets, _ in  tqdm(train_loader):
            images = images.to(self.device)
            targets = targets.to(self.device)
            current_batch_size = images.shape[0]
            n_samples += current_batch_size
 

            outputs, _ = self.model(images)
            loss = self.criterion(outputs, targets, class_weights=class_weights)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss * current_batch_size
        
        self.train_log(task_id, epoch, train_loss/n_samples)  

    #TODO: t è il task_id, immagino di doverlo usare poi per salvare data del task
    #cross entropy correct for our task of video classification
    def criterion(self, outputs, targets, class_weights=None):
        #TODO: rescale targets non dovrebbe servire...
        #targets = self.rescale_targets(targets, t)
        #here is 0 cause we only have a head
        return torch.nn.functional.cross_entropy(outputs[0], targets, class_weights)
        
        
        
    def post_train(self, task_id, trn_loader=None):
        pass 

    #TODO: definire evaluation step
    #TODO: definire quindi accuracy e AP
    #TODO: accuracy e AP vanno probabilmente generate per ogni classe e per ogni sub-behavior, 
    # per identificare comportamenti anomali al variare dei dati e dei training
    #TODO: controllare https://torchmetrics.readthedocs.io/en/stable/classification/average_precision.html
    #magari ci sono già implementati metodi interessanti
    def eval(self, current_training_task, test_id, loader, epoch, verbose, testing=False):
        #TODO: modificare anche metric evaluator per gestire anche AP e differenziazione tra classi...
        metric_evaluator = MetricEvaluatorIncDec(self.out_path, self.task_dict, self.total_classes)

        if self.imbalanced:
            class_weights = torch.Tensor(compute_class_weight('balanced', classes=np.unique(loader.dataset.dataset.targets), 
                                                              y=loader.dataset.dataset.targets)).to(self.device)
        else:
            class_weights=None
        
        cls_loss, n_samples = 0, 0 
        with torch.no_grad():
            self.model.eval()
            for images, targets, _  in tqdm(loader):
                images = images.to(self.device)
                targets = targets.type(dtype=torch.int64).to(self.device)
                current_batch_size = images.shape[0]
                n_samples += current_batch_size
 
                outputs, features = self.model(images)
                _, old_features = self.old_model(images)
                
                cls_loss += self.criterion(outputs, targets, class_weights=class_weights) * current_batch_size
                 

                metric_evaluator.update(targets, 
                                        self.compute_probabilities(outputs, 0))

            #task aware accuracy e task agnostic accuracy
            acc, ap, acc_per_class, mean_ap, map_weighted, confusion_matrix, precision, recall = metric_evaluator.get(verbose=verbose)
            

            cm_figure = self.plot_confusion_matrix(confusion_matrix, self.class_names)

            pr_figure = self.plot_pr_curve(precision, recall)
              
            self.log(current_training_task, test_id, epoch, cls_loss/n_samples, acc, mean_ap, map_weighted, cm_figure, pr_figure, testing)          
            
            if verbose:
                print(" - classification loss: {}".format(cls_loss/n_samples))

            return acc, ap, cls_loss/n_samples, acc_per_class, mean_ap, map_weighted
        
    #TODO: definire log da fare...
    def log(self, current_training_task, test_id, epoch, cls_loss , acc, mean_ap, map_weighted, cm_figure, pr_figure, testing):

        if not testing:
            name_tb = "training_task_" + str(current_training_task) + "/dataset_" + str(test_id) + "_classification_loss"
            self.logger.add_scalar(name_tb, cls_loss, epoch)

            name_tb = "training_task_" + str(current_training_task) + "/dataset_" + str(test_id) + "_accuracy"
            self.logger.add_scalar(name_tb, acc, epoch)

            name_tb = "training_task_" + str(current_training_task) + "/dataset_" + str(test_id) + "_mAP"
            self.logger.add_scalar(name_tb, mean_ap, epoch)

            name_tb = "training_task_" + str(current_training_task) + "/dataset_" + str(test_id) + "_weighted_mAP"
            self.logger.add_scalar(name_tb, map_weighted, epoch)

            name_tb = "training_task_" + str(current_training_task) + "/dataset_" + str(test_id) + "_confusion_matrix"
            self.logger.add_figure(name_tb,cm_figure,epoch)

            name_tb = "training_task_" + str(current_training_task) + "/dataset_" + str(test_id) + "_pr_curve"
            self.logger.add_figure(name_tb,pr_figure,epoch)
        else:
            name_tb = "test_task" + "/dataset_" + str(test_id) + "_accuracy"
            self.logger.add_scalar(name_tb, acc, epoch)

            name_tb = "test_task" + "/dataset_" + str(test_id) + "_mAP"
            self.logger.add_scalar(name_tb, mean_ap, epoch)

            name_tb = "test_task" + "/dataset_" + str(test_id) + "_weighted_mAP"
            self.logger.add_scalar(name_tb, map_weighted, epoch)

            name_tb = "test_task" + "/dataset_" + str(test_id) + "_confusion_matrix"
            self.logger.add_figure(name_tb,cm_figure,epoch)

            name_tb = "test_task" + "/dataset_" + str(test_id) + "_pr_curve"
            self.logger.add_figure(name_tb,pr_figure,epoch)
        
        

    def train_log(self, current_training_task, epoch, cls_loss):
        name_tb = "training_task_" + str(current_training_task) + "/training_classification_loss"
        self.logger.add_scalar(name_tb, cls_loss, epoch)


    def compute_probabilities(self, outputs, head_id):
      probabilities = torch.nn.Softmax(dim=1)(outputs[head_id])
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