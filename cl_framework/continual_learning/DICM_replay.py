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
 

class DICM_replay(IncrementalApproach):
    
    def __init__(self, args, device, out_path, task_dict, total_classes, class_to_idx, subcategories_dict, all_subcategories_dict, multilabel,no_class_check):
        self.total_classes = total_classes
        
        self.n_accumulation = args.n_accumulation
        super().__init__(args, device, out_path, total_classes, task_dict)
        self.class_to_idx = class_to_idx
        self.class_names = list(class_to_idx.keys())
        self.model = BaseModel(backbone=self.backbone, dataset=args.dataset)
        self.model.add_classification_head(self.total_classes)
        self.print_running_approach()
        self.criterion_type = args.criterion_type
        self.criterion = self.select_criterion(args.criterion_type)
        self.all_subcategories_dict = all_subcategories_dict
        self.freeze_backbone = args.freeze_backbone
        # save paths for the dataset in order to fix the files names in order to be saved
        self.data_path_prefix = None
        self.data_path_suffix = None
        # to check if working with multilabel with samples with no classses
        self.no_class_check = no_class_check
        self.multilabel = multilabel


    def print_running_approach(self):
        super(DICM_replay, self).print_running_approach()
        

    def pre_train(self,  task_id, trn_loader, test_loader):
        self.model.to(self.device)
        self.data_path_prefix, self.data_path_suffix = trn_loader.dataset.get_prefix_suffix_data_path()

        # TODO: aggiungere logica di replay buffering
        # qui sarà da caricare un Dataloader e caricare gli elementi
        # deve essere anche controllato di caricare soltanto gli elementi spariti, non tutti gli esempi
        # mi salvo una lista in post_train da li posso recuperare il resto


        #TODO: fare che poi la backbone è freezata fino ad un punto preciso, ovvero dove estraggo le features

        if task_id > 0 and self.freeze_backbone == 'yes':
            self.model.freeze_backbone()
            print('Backbone will be frozen for this task.')
        super(DICM_replay, self).pre_train(task_id)


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

        #TODO: aggiungere iter() per caricare dai due dataloader separatamente
        feature_iterator = iter(feature_dataloader)
       
        train_loss, n_samples = 0, 0
        self.optimizer.zero_grad()
        # if to work with loss accumulation, when batch size is too small
        if self.n_accumulation > 0:
            count_accumulation = 0
            for batch_idx, (images, targets, binarized_targets, _, _) in enumerate(tqdm(train_loader)):
                images = images.to(self.device)
                labels = self.select_proper_targets(targets, binarized_targets).to(self.device)
                current_batch_size = images.shape[0]
                n_samples += current_batch_size

                #TODO: fare next per ottenere nuovo example dal nuovo dataloader
                try:
                    replay_features = next(feature_iterator)
                except StopIteration:
                    feature_iterator = iter(feature_dataloader)
                    replay_features = next(feature_iterator)

                #TODO: modificare in maniera che in inferenza, prenda sia images che features e le ricombini ad un determinato stadio della rete
                outputs, _ = self.model(images)
                loss = self.criterion(outputs[0], labels)

                loss.backward()
                train_loss += loss.detach() * current_batch_size
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

                #TODO: fare next per ottenere nuovo example dal nuovo dataloader
                try:
                    replay_features = next(feature_iterator)
                except StopIteration:
                    feature_iterator = iter(feature_dataloader)
                    replay_features = next(feature_iterator)

                #TODO: modificare in maniera che in inferenza, prenda sia images che features e le ricombini ad un determinato stadio della rete
                outputs, _ = self.model(images)
                loss = self.criterion(outputs[0], labels)             
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                train_loss += loss.detach() * current_batch_size
        
        self.train_log(task_id, epoch, train_loss/n_samples)  


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

        #n_samples_batches = len(tmp_loader.dataset) // tmp_loader.batch_size

        features_path = os.path.join(self.out_path,'extracted_features')
        if not os.path.exists(features_path):
            os.mkdir(features_path)
        task_id_features_path = os.path.join(features_path,'task_{}'.format(task_id))
        if not os.path.exists(task_id_features_path):
            os.mkdir(task_id_features_path)
        

        # file to save the examples that were saved
        name_file = os.path.join(task_id_features_path,"task_{}_feature_data.csv".format(task_id))
        data_paths_list = []
        labels_list = []
        subcategories_list = []

        self.model.eval()

        with torch.no_grad():
            
            for batch_idx, (images, targets, binarized_targets, subcategory, data_path) in enumerate(tqdm(tmp_loader)):
                lab = self.select_proper_targets(targets,binarized_targets)
                
                features = self.model.backbone.extract_blocks_features(images)
                for i in range(features.shape[0]):
                    # TODO: save features and then save a list of the elements used
                    # fix the data path to only include the name
                    features_names = data_path[i].replace(self.data_path_prefix, '').replace(self.data_path_suffix,'')
                    torch.save(features[i], features_names + '.pt')
                data_paths_list.append(data_path)
                labels_list.append(lab)
                subcategories_list.append(subcategory)
                    
        
        self.save_features_list(self.class_names, data_paths_list, labels_list, subcategories_list, name_file)





    
    def eval(self, current_training_task, test_id, loader, epoch, verbose, testing=None):
        if self.multilabel:
            metric_evaluator = MetricEvaluatorIncDec_multilabel(self.out_path, self.total_classes, self.criterion_type, self.all_subcategories_dict, self.class_to_idx)
        else:
            metric_evaluator = MetricEvaluatorIncDec(self.out_path, self.total_classes, self.criterion_type, self.all_subcategories_dict, self.class_to_idx)


        
        cls_loss, n_samples = 0, 0 
        with torch.no_grad():
            self.model.eval()
            for images, targets, binarized_targets, subcategory, data_path in tqdm(loader):
                images = images.to(self.device)
                labels = self.select_proper_targets(targets, binarized_targets).to(self.device)
                current_batch_size = images.shape[0]
                n_samples += current_batch_size
 
                outputs, features = self.model(images)
                
                cls_loss += self.criterion(outputs[0], labels) * current_batch_size
                 

                metric_evaluator.update(targets, binarized_targets.float().squeeze(dim=1),
                                        self.compute_probabilities(outputs, 0), subcategory, data_path)
                

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
                print(" - classification loss: {}".format(cls_loss/n_samples))

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


    def save_features_list(self, class_names, data_paths, targets, subcategory, save_path):
        
        
        if self.criterion_type == "multilabel":
            binary_targets = {}
            for i in range(len(class_names)):
                binary_targets["target_" + class_names[i]] = targets[:,i]
            
            d = {'video_path':data_paths, 'subcategory': subcategory}
            unified_dict = d | binary_targets
        elif self.criterion_type == "multiclass":
            d = {'video_path':data_paths, 'target':targets, 'subcategory': subcategory}
            unified_dict = d
        df = pd.DataFrame(unified_dict)
        df.to_csv(save_path, index=False)