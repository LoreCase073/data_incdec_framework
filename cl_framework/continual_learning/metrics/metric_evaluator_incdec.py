import torch
import os
from sklearn.metrics import accuracy_score, average_precision_score
import matplotlib.pyplot as plt
import numpy as np

class MetricEvaluatorIncDec():
    def __init__(self, out_path, task_dict):
        self.out_path = out_path

        self.probabilities = []
        self.labels = []

        self.task_dict = task_dict



    def update(self, labels, probabilities):
        self.probabilities.append(probabilities)
        self.labels.append(labels)

    

    def get(self, verbose):
        self.probabilities = torch.cat(self.probabilities)
        self.labels = torch.cat(self.labels).cpu().numpy()
        #TODO: check if it is axis = 1, but should be
        acc = accuracy_score(self.labels, torch.max(self.probabilities, axis = 1)[1].cpu().numpy())
        ap = average_precision_score(self.labels, torch.max(self.probabilities, axis = 1)[1].cpu().numpy())
 

        if verbose:
            print(" - task accuracy: {}".format(acc))
            print(" - task average precision: {}".format(ap))

        return acc, ap
    


