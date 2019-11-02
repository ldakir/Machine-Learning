"""
Top level comment: be sure to include the purpose/contents of this file
as well as the author(s)
"""
import util
from Partition import *
from NaiveBayes import *
import numpy as np

def main():

    opts = util.parse_args()
    train_partition = util.read_arff(opts.train_filename)
    test_partition  = util.read_arff(opts.test_filename)


    #Creating Naive Bayes Model
    nb_model = NaiveBayes(train_partition)
    m = len(test_partition.labels)
    confusion_matrix = np.zeros((m,m)) #initializing the confusion matrix
    accuracy = 0
    for x in test_partition.data:
        y_hat = nb_model.classify(x.features)
        y = x.label
        confusion_matrix[y][y_hat] +=1
        if y == y_hat:
            accuracy+=1


    print('Accuracy: '+ str(round(accuracy/test_partition.n,6)) +' ('+str(accuracy) + ' out of ' + str(test_partition.n) +' correct)')
    print(confusion_matrix)


main()
