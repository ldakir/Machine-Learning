"""
Decision tree data structure (recursive).
Author: Lamiaa Dakir
Date: 09/12/2019
"""
from Partition import *

class Node:
    def __init__(self):
        """
        Helper class that stores info about each node.
        """
        self.children = {}
        self.value = None
        self.depth = 0

class DecisionTree:
    def __init__(self):
        self.counts = []
        self.total_count = []
        self.leaf = []
        self.ans = ''

    def FindBestFeature(self,D,F):
        """
        Finding the feature with the highest gain
        """
        features = list(F.keys())
        best = features[0]
        for i in range(1,len(features)):
            if D.gain(best) < D.gain(features[i]):
                best = features[i]
        return best

    def stop(self,D,F,depth):
        """
        Specifiying stopping criteria
        """
        stop = False
        labels = []
        for x in D.data:
            labels.append(x.label)
        labels = set(labels)

        if len(labels) == 1 or len(F) == 0 or depth == 0 or len(D.data) == 0:
            stop = True
        return stop


    def MakeSubTree(self,D,F,max_depth,current_depth):
        """
        Recursive function to build training tree
        """
        node = Node()

        if self.stop(D,F,max_depth): #stopping criteria is met

            class_label = 0
            for x in D.data:
                class_label += x.label

            if class_label > 0:
                self.leaf.append('1')

            else:
                self.leaf.append('-1')

        else:
            if max_depth != None:
                max_depth -=1

            S = self.FindBestFeature(D,F)
            outcomes = F[S]
            self.total_count.append('[' + str(D.counts(-1,D.data))+', ' +str(D.counts(1,D.data)) +']')
            node.value = S
            outcomes.sort()

            for i in range(len(outcomes)):
                Dk = Partition(D.find_subdata(S,outcomes[i]),F)
                self.counts.append('[' + str(Dk.counts(-1,Dk.data))+', ' +str(Dk.counts(1,Dk.data)) +']')
                current_depth +=1
                node.children[outcomes[i]] = self.MakeSubTree(Dk, F,max_depth,current_depth)
                node.children[outcomes[i]].depth = current_depth

        return node

    def printDecisionTree(self, node):
        """
        Print root and tree nodes
        """
        leaf =self.leaf
        answer = self.print_help(node,leaf)
        return self.total_count[0] + '\n' +answer

    def print_help(self, node,leaf):
        """
        Helper function to recurse on each subtree
        """
        root = node.value
        depth = node.depth
        for x in node.children:
            if node.children[x].value != None:
                self.ans+=(depth *'\t'+root + '=' + x + ' '+self.counts[0]+'\n')
                self.counts = self.counts[1:]
                depth = node.children[x].depth
            else:
                self.ans+=(depth *'\t'+root + '=' + x + ' '+self.counts[0])
                self.counts = self.counts[1:]
                self.ans+=(': '+leaf[0]+'\n')
                leaf = leaf[1:]
            depth = node.depth
            self.print_help(node.children[x],leaf)
        return self.ans
