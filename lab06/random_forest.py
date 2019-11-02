"""
Implements Random Forests with decision stumps.
Authors: Lamiaa Dakir
Date: 10/27/2019
"""

import util
from random import randrange
from math import sqrt
from Partition import *
from DecisionStump import *
import numpy as np

def random_forest_train_data(train_partition,T):
    """
    Training data using the random forest algorithm
    """
    # training the data
    ensemble = []
    for i in range(int(T)):
        # create a bootstrap training dataset by randomly sampling from the original training data with replacement
        bootstrapped_data = []
        n = len(train_partition.data)
        for i in range(n):
            bootstrapped_data.append(train_partition.data[randrange(n)])


        # select a random subset of features without replacement
        num_features = int(round(sqrt(len(train_partition.F)),0))
        features = list(train_partition.F.keys())
        features_subset = {}
        for i in range(num_features):
            element = features[randrange(len(features))]
            features_subset[element] = train_partition.F[element]
            features.remove(element)

        # using the bootstrap sample and the subset of features, create a decision stump
        new_partition = Partition(bootstrapped_data,features_subset)
        decision_stump = DecisionStump(new_partition)
        ensemble.append(decision_stump)

    return ensemble

def random_forest_test_data(test_partition,ensemble,threshold):
    """
    Testing data using the random forest algorithm
    """
    # testing test example by running it through each classifier in the ensemble
    prediction = []
    for x in test_partition.data:
        result = []
        for d in ensemble:
            label = d.classify(x.features,threshold)
            result.append(label)
        y = sum(result)
        if y > 0:
            prediction.append(1)
        else:
            prediction.append(-1)



    #construst the confusion matrix
    confusion_matrix = np.zeros((2,2))
    accuracy =0
    for i in range(len(prediction)):
        if  prediction[i] == -1 and test_partition.data[i].label == -1:
            confusion_matrix[0][0] +=1
            accuracy +=1
        elif prediction[i] == 1 and test_partition.data[i].label  == 1:
            confusion_matrix[1][1] +=1
            accuracy +=1
        elif prediction[i] == -1 and test_partition.data[i].label  == 1:
            confusion_matrix[1][0] +=1

        elif prediction[i] == 1 and test_partition.data[i].label  == -1:
            confusion_matrix[0][1] +=1

    FPR = util.FPR(confusion_matrix)
    TPR = util.TPR(confusion_matrix)

    return confusion_matrix,FPR,TPR



def main():

    # read in data (y in {-1,1})
    opts = util.parse_args('Random forests')
    train_partition = util.read_arff(opts.train_filename)
    test_partition  = util.read_arff(opts.test_filename)
    T = opts.classifier_nums
    threshold = opts.thresh

    # training the data
    ensemble = random_forest_train_data(train_partition,T)

    # testing the data
    confusion_matrix,FPR,TPR = random_forest_test_data(test_partition,ensemble,threshold)

    print('T: '+ str(T) +' , thresh ' + str(threshold))
    print('\n')
    print(' prediction')
    print('   -1  1')
    print('   -----')
    print('-1| '+ str(int(confusion_matrix[0][0])) + '  ' + str(int(confusion_matrix[0][1])))
    print(' 1| '+ str(int(confusion_matrix[1][0])) + '  ' + str(int(confusion_matrix[1][1])))
    print('\n')

    # calculating the false positive rate and the true positive rate
    print('false positive: '+ str(FPR))
    print('true positive: '+ str(TPR))






if __name__ == "__main__":
    main()
