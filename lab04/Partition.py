"""
Partition class (holds feature information, feature values, and labels for a
dataset). Includes helper class Example.
Author: Sara Mathieson + Lamiaa Dakir
Date: 13/09/2019
"""
from math import log2

class Example:

    def __init__(self, features, label):
        """
        Helper class (like a struct) that stores info about each example.
        """
        # dictionary. key=feature name: value=feature value for this example
        self.features = features
        self.label = label #{0,1,2,…,K-1}

class Partition:

    def __init__(self, data, F):
        """
        Store information about a dataset
        """
        self.data = data # list of examples
        # dictionary. key=feature name: value=set of possible values
        self.F = F
        self.n = len(self.data)
        K = 0
        self.labels = []
        for x in self.data:
            if x.label not in self.labels:
                K +=1
                self.labels.append(x.label)
        self.K = K #Number of classes
