import torch
import os
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from torchmetrics.classification import MulticlassAccuracy, MulticlassAveragePrecision

class MetricEvaluatorIncDec():
    def __init__(self, out_path, task_dict, num_classes):
        self.out_path = out_path
        self.num_classes = num_classes

        self.probabilities = []
        self.labels = []

        self.task_dict = task_dict



    def update(self, labels, probabilities):
        self.probabilities.append(probabilities.cpu())
        self.labels.append(labels.cpu())

    

    def get(self, verbose):
        self.probabilities = torch.cat(self.probabilities)
        self.labels = torch.cat(self.labels).cpu().numpy()
        #TODO: check if it is axis = 1, but should be
        
        ap_metric = MulticlassAveragePrecision(num_classes=self.num_classes, average=None)
        map_metric = MulticlassAveragePrecision(num_classes=self.num_classes, average='macro')
        map_weighted_metric = MulticlassAveragePrecision(num_classes=self.num_classes, average='weighted')
        acc_per_class_metric = MulticlassAccuracy(num_classes=self.num_classes, average=None)

        acc = accuracy_score(self.labels, torch.max(self.probabilities, axis = 1)[1].cpu().numpy())
        acc_per_class = acc_per_class_metric(self.probabilities, torch.tensor(self.labels))

        ap = ap_metric(self.probabilities, torch.tensor(self.labels))
        mean_ap = map_metric(self.probabilities, torch.tensor(self.labels))
        map_weighted = map_weighted_metric(self.probabilities, torch.tensor(self.labels))

        confusion_matrix = confusion_matrix(self.labels, torch.max(self.probabilities, axis = 1)[1].cpu().numpy())

        if verbose:
            print(" - task accuracy: {}".format(acc))
            print(" - task average precision: {}".format(ap))
            print(" - task acc per class: {}".format(acc_per_class))
            print(" - task mAP: {}".format(mean_ap))
            print(" - task weighted mAP: {}".format(map_weighted))

        return acc, ap, acc_per_class, mean_ap, map_weighted, confusion_matrix
    
    def log_pr_curves(self, logger, classes_names,epoch, test_id, testing):

        for i in range(len(classes_names)):
            self.add_pr_curve_tensorboard(logger, i, self.probabilities, self.labels, classes_names[i], epoch,test_id, testing)
        

    def add_pr_curve_tensorboard(logger, class_index, test_probs, test_label, class_name, epoch,test_id, testing):
        '''
        Takes in a "class_index" from 0 to 9 and plots the corresponding
        precision-recall curve
        '''
        tensorboard_truth = test_label == class_index
        tensorboard_probs = test_probs[:, class_index]
        if not testing:
            name_tb = "training_task_" + str(epoch) + "/dataset_" + str(test_id) + "_class_" + str(class_name) + "pr_curve"
        else:
            name_tb = "test_task" + "/dataset_" + str(test_id) + "_class_" + str(class_name) + "pr_curve"
        logger.add_pr_curve(name_tb,
                            tensorboard_truth,
                            tensorboard_probs,
                            global_step=epoch)