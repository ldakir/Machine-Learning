"""
Naive Bayes Class
Author: Lamiaa Dakir
Date: 09/27/2019
"""
from Partition import *
from math import log

class NaiveBayes:
    def __init__(self,partition):
        self.n = partition.n
        self.K = partition.K
        self.data = partition.data

        #Finding the number of possible feature values
        for x in partition.F:
            self.Fj =len(partition.F[x])
            break

        self.labels = partition.labels

    def N(self,k):
        result = 0
        for x in self.data:
            if x.label == k:
                result +=1
        return result

    def N_j(self,j,v,k):
        result = 0
        for x in self.data:
            for f in x.features:
                if f == j and x.features[j] == v:
                    if x.label == k:
                        result +=1
        return result

    def O(self,k):
        return (self.N(k)+1)/(self.n+self.K)

    def O_j(self,j,v,k):
        return (self.N_j(j,v,k)+1)/(self.N(k)+self.Fj)

    def P(self,k,features):
        """
        Naive Bayes model for the class label k
        """
        prob = 1
        for f in features:
            prob *= self.O_j(f,features[f],k)
        return self.O(k)*prob


    def classify(self,features):
        """
        function to classify text examples and predict label
        """
        probs = self.P(self.labels[0],features)
        y_hat = self.labels[0]

        for i in range (1,len(self.labels)):
            if self.P(self.labels[i],features)> probs:
                probs = self.P(self.labels[i],features)
                y_hat = self.labels[i]
        return y_hat
