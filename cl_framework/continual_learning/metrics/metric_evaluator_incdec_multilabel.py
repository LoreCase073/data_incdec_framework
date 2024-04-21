import torch
import os
from sklearn.metrics import accuracy_score, confusion_matrix, multilabel_confusion_matrix, precision_recall_curve, PrecisionRecallDisplay
import matplotlib.pyplot as plt
import numpy as np
from torchmetrics.classification import MultilabelAveragePrecision, Recall, Precision, MultilabelAccuracy
from sklearn.preprocessing import label_binarize


class MetricEvaluatorIncDec_multilabel():
    def __init__(self, out_path, num_classes, criterion_type, all_subcategories_dict, class_to_idx):
        self.out_path = out_path
        self.num_classes = num_classes
        self.class_to_idx = class_to_idx

        self.probabilities = []
        self.labels = []
        self.binarized_labels = []
        self.subcategory = []
        self.data_paths = []

        self.criterion_type = criterion_type
        # this is required to extract the ap for each subcategory
        self.all_subcategories_dict = all_subcategories_dict



    def update(self, labels, binarized_labels, probabilities, subcategory, data_path):
        self.probabilities.append(probabilities.cpu())
        self.labels.append(labels.cpu())
        self.binarized_labels.append(binarized_labels.cpu())
        subcategory = list(subcategory)
        data_path = list(data_path)
        for i in range(len(subcategory)):
            self.subcategory.append(subcategory[i])
            self.data_paths.append(data_path[i])

    

    def get(self, verbose):
        self.probabilities = torch.cat(self.probabilities)
        self.labels = torch.cat(self.labels).type(dtype=torch.int64)
        self.binarized_labels = torch.cat(self.binarized_labels).type(dtype=torch.int64)
        

        

        
        ap_metric = MultilabelAveragePrecision(num_labels=self.num_classes, average=None)
        map_metric = MultilabelAveragePrecision(num_labels=self.num_classes, average='macro')
        map_weighted_metric = MultilabelAveragePrecision(num_labels=self.num_classes, average='weighted')

        acc_per_class_metric = MultilabelAccuracy(num_labels=self.num_classes,average=None)

        acc_per_class = acc_per_class_metric(self.probabilities, self.binarized_labels)

        #acc = self.compute_multilabel_accuracy(self.binarized_labels.numpy(), pos_max_prediction)
        acc = accuracy_score(self.binarized_labels.numpy(), (self.probabilities >= 0.5).numpy())
        #TODO: verificare come calcolare queste precision e recall, attualmente le calcolo come se fosse un task multilabel
        # forse fare che prendo il massimo e calcolo come in multitask, almeno per il nostro caso, ignorando VZC?
        precision_per_class_metric = Precision(task='multilabel', average=None, num_labels=self.num_classes)
        recall_per_class_metric = Recall(task='multilabel', average=None, num_labels=self.num_classes)
        exact_match = accuracy_score(self.binarized_labels, (self.probabilities >= 0.5).numpy())

        
        
        
        ap = ap_metric(self.probabilities, self.binarized_labels)
        mean_ap = map_metric(self.probabilities, self.binarized_labels)
        map_weighted = map_weighted_metric(self.probabilities, self.binarized_labels)
        
        precision_per_class = precision_per_class_metric(self.probabilities, self.binarized_labels)
        recall_per_class = recall_per_class_metric(self.probabilities, self.binarized_labels)


        # now get ap for each subcategory and recall for each subcategory
        ap_per_subcategory = self.get_ap_per_subcategory(self.probabilities, self.binarized_labels, self.subcategory)
        recall_per_subcategory = self.get_recall_per_subcategory(self.probabilities, self.binarized_labels, self.subcategory)
        accuracy_per_subcategory = self.get_accuracy_per_subcategory(self.probabilities, self.binarized_labels, self.subcategory)
        precision_per_subcategory = self.get_precision_per_subcategory(self.probabilities, self.binarized_labels, self.subcategory)

        if verbose:
            print(" - task accuracy: {}".format(acc))
            print(" - task average precision: {}".format(ap))
            print(" - task acc per class: {}".format(acc_per_class))
            print(" - task precision per class: {}".format(precision_per_class))
            print(" - task recall per class: {}".format(recall_per_class))
            print(" - task mAP: {}".format(mean_ap))
            print(" - task weighted mAP: {}".format(map_weighted))

        return acc, ap, acc_per_class, mean_ap, map_weighted, precision_per_class, recall_per_class, exact_match, ap_per_subcategory, recall_per_subcategory, accuracy_per_subcategory, precision_per_subcategory

    def get_precision_recall_cm(self):
        if self.criterion_type == "multiclass":
            c_matrix = confusion_matrix(self.labels, torch.max(self.probabilities, axis = 1)[1].numpy())
        elif self.criterion_type == "multilabel":
            # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.multilabel_confusion_matrix.html
            # matrix will have the form:
            """ 
            TN FP
            FN TP
              """
            c_matrix = multilabel_confusion_matrix(self.binarized_labels, (self.probabilities >= 0.5).numpy())


        # precision recall curve
        if self.criterion_type == "multiclass":
            Y = label_binarize(self.labels, classes=[i for i in range(self.num_classes)])
        elif self.criterion_type == "multilabel":
            Y = self.binarized_labels
        precision = dict()
        recall = dict()
        for i in range(self.num_classes):
            precision[i], recall[i], _ = precision_recall_curve(Y[:, i],
                                                                self.probabilities[:, i])

        return c_matrix, precision, recall
        

    def get_data_paths(self):
        return self.data_paths
    
    def get_subcategories(self):
        return self.subcategory
    
    def get_predictions(self):
        return torch.max(self.probabilities, axis = 1)[1].numpy()
    
    def get_probabilities(self):
        return self.probabilities.numpy()
    
    def get_targets(self):
        if self.criterion_type == 'multiclass':
            return self.labels
        elif self.criterion_type == 'multilabel':
            return self.binarized_labels
    
    
    
    
    def get_ap_per_subcategory(self, predictions, labels, subcategories):
        ap_per_subcategory = {}
        ap_metric = MultilabelAveragePrecision(num_labels=self.num_classes, average=None)
        for class_name in self.all_subcategories_dict:
            # do not do it for the nothing class
            if class_name != 'nothing':
                idx_class = self.class_to_idx[class_name]
                class_subcategories = self.all_subcategories_dict[class_name]
                for idx_subcat in range(len(class_subcategories)):
                    
                    subcategory_name = class_subcategories[idx_subcat]
                    current_class_indices = np.where(np.array(labels[:,idx_class]) == 1)[0].tolist()
                    current_subcategory_indices = np.where((np.array(subcategories) == subcategory_name))[0].tolist()

                    current_subcategory_indices = list(set(current_subcategory_indices).intersection(current_class_indices))

                    other_classes_indices = np.where(np.array(labels[:,idx_class]) != 1)[0].tolist()

                    subset_indices = torch.IntTensor(list(current_subcategory_indices + other_classes_indices))

                    subset_predictions = torch.index_select(predictions, dim=0, index=subset_indices)
                    subset_labels = torch.index_select(labels, dim=0, index=subset_indices)
                    
                    ap_per_subcategory[subcategory_name] = ap_metric(subset_predictions, subset_labels)[idx_class]

        return ap_per_subcategory
    
    def get_recall_per_subcategory(self, predictions, labels, subcategories):
        recall_per_subcategory = {}
        recall_metric = Recall(task='multilabel',num_labels=self.num_classes, average=None)
        for class_name in self.all_subcategories_dict:
            if class_name != 'nothing':
                idx_class = self.class_to_idx[class_name]
                class_subcategories = self.all_subcategories_dict[class_name]
                for idx_subcat in range(len(class_subcategories)):
                    
                    subcategory_name = class_subcategories[idx_subcat]
                    current_class_indices = np.where(np.array(labels[:,idx_class]) == 1)[0].tolist()
                    current_subcategory_indices = np.where((np.array(subcategories) == subcategory_name))[0].tolist()

                    current_subcategory_indices = list(set(current_subcategory_indices).intersection(current_class_indices))

                    other_classes_indices = np.where(np.array(labels[:,idx_class]) != 1)[0].tolist()

                    subset_indices = torch.IntTensor(list(current_subcategory_indices + other_classes_indices))

                    subset_predictions = torch.index_select(predictions, dim=0, index=subset_indices)
                    subset_labels = torch.index_select(labels, dim=0, index=subset_indices)
                    
                    recall_per_subcategory[subcategory_name] = recall_metric(subset_predictions, subset_labels)[idx_class]

        return recall_per_subcategory
    

    def get_precision_per_subcategory(self, predictions, labels, subcategories):
        precision_per_subcategory = {}
        precision_metric = Precision(task='multilabel',num_labels=self.num_classes, average=None)
        for class_name in self.all_subcategories_dict:
            if class_name != 'nothing':
                idx_class = self.class_to_idx[class_name]
                class_subcategories = self.all_subcategories_dict[class_name]
                for idx_subcat in range(len(class_subcategories)):
                    
                    subcategory_name = class_subcategories[idx_subcat]
                    current_class_indices = np.where(np.array(labels[:,idx_class]) == 1)[0].tolist()
                    current_subcategory_indices = np.where((np.array(subcategories) == subcategory_name))[0].tolist()

                    current_subcategory_indices = list(set(current_subcategory_indices).intersection(current_class_indices))

                    other_classes_indices = np.where(np.array(labels[:,idx_class]) != 1)[0].tolist()

                    subset_indices = torch.IntTensor(list(current_subcategory_indices + other_classes_indices))

                    subset_predictions = torch.index_select(predictions, dim=0, index=subset_indices)
                    subset_labels = torch.index_select(labels, dim=0, index=subset_indices)
                    
                    precision_per_subcategory[subcategory_name] = precision_metric(subset_predictions, subset_labels)[idx_class]

        return precision_per_subcategory
    

    def get_accuracy_per_subcategory(self, predictions, labels, subcategories):
        accuracy_per_subcategory = {}
        accuracy_metric = MultilabelAccuracy(num_labels=self.num_classes, average=None)
        for class_name in self.all_subcategories_dict:
            if class_name != 'nothing':
                idx_class = self.class_to_idx[class_name]
                class_subcategories = self.all_subcategories_dict[class_name]
                for idx_subcat in range(len(class_subcategories)):
                    
                    subcategory_name = class_subcategories[idx_subcat]
                    current_class_indices = np.where(np.array(labels[:,idx_class]) == 1)[0].tolist()
                    current_subcategory_indices = np.where((np.array(subcategories) == subcategory_name))[0].tolist()

                    current_subcategory_indices = list(set(current_subcategory_indices).intersection(current_class_indices))

                    other_classes_indices = np.where(np.array(labels[:,idx_class]) != 1)[0].tolist()

                    subset_indices = torch.IntTensor(list(current_subcategory_indices + other_classes_indices))

                    subset_predictions = torch.index_select(predictions, dim=0, index=subset_indices)
                    subset_labels = torch.index_select(labels, dim=0, index=subset_indices)
                    
                    accuracy_per_subcategory[subcategory_name] = accuracy_metric(subset_predictions, subset_labels)[idx_class]

        return accuracy_per_subcategory