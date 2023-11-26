import torch
import os
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, PrecisionRecallDisplay
import matplotlib.pyplot as plt
import numpy as np
from torchmetrics.classification import MulticlassAccuracy, MulticlassAveragePrecision
from sklearn.preprocessing import label_binarize


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

        c_matrix = confusion_matrix(self.labels, torch.max(self.probabilities, axis = 1)[1].cpu().numpy())

        # precision recall curve
        Y = label_binarize(self.labels, classes=[i for i in range(self.num_classes)])
        precision = dict()
        recall = dict()
        for i in range(self.num_classes):
            precision[i], recall[i], _ = precision_recall_curve(Y[:, i],
                                                                self.probabilities[:, i])
        
        

        if verbose:
            print(" - task accuracy: {}".format(acc))
            print(" - task average precision: {}".format(ap))
            print(" - task acc per class: {}".format(acc_per_class))
            print(" - task mAP: {}".format(mean_ap))
            print(" - task weighted mAP: {}".format(map_weighted))

        return acc, ap, acc_per_class, mean_ap, map_weighted, c_matrix, precision, recall
