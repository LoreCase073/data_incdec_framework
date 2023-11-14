import torch
import os
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

class MetricEvaluatorIncDec():
    def __init__(self, out_path, task_dict):
        self.out_path = out_path

        self.probabilities = []
        self.labels = []

        self.task_dict = task_dict



    def update(self, labels, taw_probabilities):
        self.probabilities.append(taw_probabilities)
        self.labels.append(labels)

    

    def get(self, verbose):
        self.probabilities = torch.cat(self.probabilities)
        self.labels = torch.cat(self.labels).cpu().numpy()

        acc = accuracy_score(self.taw_labels, torch.max(self.taw_probabilities, axis = 1)[1].cpu().numpy())
 

        if verbose:
            print(" - task accuracy: {}".format(acc))

        return acc 
    


