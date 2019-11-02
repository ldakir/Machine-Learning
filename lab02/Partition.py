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
        self.label = label # in {-1, 1}

class Partition:

    def __init__(self, data, F):
        """
        Store information about a dataset
        """
        self.data = data # list of examples
        # dictionary. key=feature name: value=set of possible values
        self.F = F
        self.n = len(self.data)

    # TODO: implement entropy and information gain methods here!
    def prob(self,c, data):
        """
        calculates the probability of a certain label
        """
        num_label = 0
        total =0
        for x in data:
            if x.label == c :
                num_label += 1
            total +=1

        return num_label/total

    def counts(self,c, data):
        """
        counts how many labels are -1 and how many are 1
        """
        num_label = 0
        for x in data:
            if x.label == c :
                num_label += 1
        return num_label

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
        for x in self.F[feature]:
            prob = len(self.find_subdata(feature,x))/len(self.data)

            H += prob*self.conditional_entropy_feature(feature,x)
        return H

    def gain(self,feature):
        """
        function to calculate the information gain
        """
        return self.entropy()- self.conditional_entropy(feature)
