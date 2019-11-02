"""
Example and Partition data structures.
Authors: Sara Mathieson + Lamiaa Dakir
Date: 10/27/2019
"""
from math import log2

class Example:

    def __init__(self, features, label, weight):
        """
        Class to hold an individual example (data point) and its weight.
        features -- dictionary of features for this example (key: feature name,
            value: feature value for this example)
        label -- label of the example (-1 or 1)
        weight -- weight of the example (starts out as 1/n)
        """
        self.features = features
        self.label = label
        self.weight = weight

    def set_weight(self, new):
        """Change the weight on an example (used for AdaBoost)."""
        self.weight = new

class Partition:

    def __init__(self, data, F):
        """
        Class to hold a set of examples and their associated features, as well
        as compute information about this partition (i.e. entropy).
        data -- list of Examples
        F -- dictionary (key: feature name, value: list of feature values)
        """
        self.data = data
        self.F = F
        self.n = len(self.data)

    def prob(self,c, data):
        """
        calculates the probability of a certain label
        """
        num_label = 0
        total =0
        for x in data:
            if x.label == c :
                num_label += x.weight
            total += x.weight

        return num_label/total


    def entropy(self):
        """
        calculating the entropy
        """
        label_1_prob = self.prob(-1,self.data)
        label_2_prob = self.prob(1,self.data)
        if label_1_prob == 0 and label_2_prob == 0:
            H = 0
        elif label_1_prob == 0:
            H = -label_2_prob*log2(label_2_prob)
        elif label_2_prob == 0:
            H = -label_1_prob*log2(label_1_prob)
        else:
            H = -label_1_prob*log2(label_1_prob)-label_2_prob*log2(label_2_prob)
        return H

    def find_subdata(self,feature, feature_value):
        """
        finds subdata that have the specific feature
        """
        data = []
        for x in self.data:
            if x.features[feature] == feature_value:
                data.append(x)
        return data

    def conditional_entropy_feature(self,f, feature_val):
        """
        helper function to calculate the conditional entropy
        """
        data = self.find_subdata(f,feature_val)
        if len(data)== 0:
            label_1_prob = 0
            label_2_prob = 0
        else:
            label_1_prob = self.prob(-1,data)
            label_2_prob = self.prob(1,data)
        if label_1_prob == 0 and label_2_prob == 0:
            H_feature = 0
        elif label_1_prob == 0:
            H_feature = -label_2_prob*log2(label_2_prob)
        elif label_2_prob == 0:
            H_feature = -label_1_prob*log2(label_1_prob)
        else:
            H_feature = -label_1_prob*log2(label_1_prob)-label_2_prob*log2(label_2_prob)
        return H_feature



    def conditional_entropy(self,feature):
        """
        Calculates the conditional entropy
        """
        H  = 0
        for v in self.F[feature]:
            subdata = self.find_subdata(feature,v)
            prob = 0
            for x in subdata:
                prob += x.weight

            H += prob*self.conditional_entropy_feature(feature,v)
        return H

    def gain(self,feature):
        """
        function to calculate the information gain
        """
        return self.entropy()- self.conditional_entropy(feature)

    def best_feature(self):
        """
        finding the best feature using info gain
        """
        max_gain = 0
        best_feature = None
        for x in self.F:
            if self.gain(x) > max_gain:
                max_gain = self.gain(x)
                best_feature = x
        return best_feature

    def prob_pos(self):
        """
        probability of assigning a positive label at a leaf
        """
        if len(self.data) == 0:
            return 1/2
        else:
            return self.prob(1,self.data)
