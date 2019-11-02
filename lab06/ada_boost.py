"""
Implements AdaBoost algorithm with decision stumps.
Authors: Lamiaa Dakir
Date: 10/27/2019
"""

import util
from math import exp
from numpy import log
from Partition import *
from DecisionStump import *

def weighted_error(data, classifier,threshold):
    """
    Computing the weighted error on a classifier.
    """
    error = 0
    for x in data:
        if(x.label != classifier.classify(x.features,threshold)):
            error += x.weight

    return error

def score(error):
    """
    Computing the score on a classifier.
    """
    return (1/2) * log((1-error)/error)

def constant(data, classifier,error, alpha_t,threshold):
    """
    Calculating the normalizing constant that ensures the weights sum to 1.
    """
    s = 0
    for x in data:
        s += x.weight*exp(-x.label*alpha_t*classifier.classify(x.features,threshold))

    return 1/s

def ada_boost_train_data(train_partition,T):
    """
    Training data using the AdaBoost algorithm
    """
    ensemble = []
    thresh_train = 0.5

    for i in range(int(T)):
        # training the classifier on the full weighted training dataset
        decision_stump = DecisionStump(train_partition)


        # computing the weighted error of the classifier
        w_error = weighted_error(train_partition.data, decision_stump,thresh_train)

        # computing the score of the classifier
        classifier_score = score(w_error)
        ensemble.append([decision_stump,classifier_score])


        # calculating the normalizing constant
        ct = constant(train_partition.data, decision_stump,w_error,classifier_score,thresh_train)

        # updating the weights
        for x in train_partition.data:
            new_weight = ct* x.weight*exp(-x.label*classifier_score*decision_stump.classify(x.features,thresh_train))
            x.set_weight(new_weight)
    return ensemble


def ada_boost_test_data(test_partition,ensemble,threshold):
    """
    Testing data using the AdaBoost algorithm
    """
    # testing test examples using classifiers
    prediction = []
    for x in test_partition.data:
        y = 0
        for d in ensemble:
            alpha_t = d[1]
            y += alpha_t*d[0].classify(x.features,threshold)
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

    # calculating the false positive rate and the true positive rate
    FPR = util.FPR(confusion_matrix)
    TPR = util.TPR(confusion_matrix)
    return confusion_matrix,FPR,TPR



def main():
    # read in data (y in {-1,1})
    opts = util.parse_args('AdaBoost')
    train_partition = util.read_arff(opts.train_filename)
    test_partition  = util.read_arff(opts.test_filename)
    T = opts.classifier_nums
    threshold = opts.thresh

    # training the data
    ensemble = ada_boost_train_data(train_partition,T)

    # testing the data
    confusion_matrix,FPR,TPR = ada_boost_test_data(test_partition,ensemble,threshold)

    # outputting confusion matrix
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
