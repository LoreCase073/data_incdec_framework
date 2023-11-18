import torch
import os
from sklearn.metrics import accuracy_score, average_precision_score
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
        acc = accuracy_score(self.labels, torch.max(self.probabilities, axis = 1)[1].cpu().numpy())
        ap_metric = MulticlassAveragePrecision(num_classes=self.num_classes, average=None)

        acc_per_class_metric = MulticlassAccuracy(num_classes=self.num_classes, average=None)

        acc_per_class = acc_per_class_metric(self.probabilities, torch.tensor(self.labels))

        ap = ap_metric(self.probabilities, torch.tensor(self.labels))

        if verbose:
            print(" - task accuracy: {}".format(acc))
            print(" - task average precision: {}".format(ap))
            print(" - task acc per class: {}".format(acc_per_class))

        return acc, ap, acc_per_class
    


