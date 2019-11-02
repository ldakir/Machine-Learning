"""
Author: Lamiaa Dakir
Date: 10/21/2019
Description: modeling the training data and predicting the labels for the test data
"""

import numpy as np
import util

# import our model class
from LogisticRegression import *

######################################################################
# main
######################################################################

def main() :
    #load data
    opts = util.parse_args()
    train_data = load_data(opts.train_filename)
    test_data = load_data(opts.test_filename)
    alpha = opts.alpha

    #append 1 to the feature vector
    n,p = train_data.X.shape
    ones = np.ones((n,1))
    train_data.X = np.concatenate((ones,train_data.X), axis =1)
    n,p = test_data.X.shape
    ones = np.ones((n,1))
    test_data.X = np.concatenate((ones,test_data.X), axis =1)

    #model the training data
    model_train = LogisticRegression()
    model_train.fit_SGD(train_data.X, train_data.y, alpha)

    #construst the confusion matrix
    confusion_matrix = np.zeros((2,2))
    accuracy =0
    for i in range(len(test_data.X)):
        if model_train.predict(test_data.X[i]) == 0 and test_data.y[i] == 0:
            confusion_matrix[0][0] +=1
            accuracy +=1
        elif model_train.predict(test_data.X[i]) == 1 and test_data.y[i] == 1:
            confusion_matrix[1][1] +=1
            accuracy +=1
        elif model_train.predict(test_data.X[i]) == 0 and test_data.y[i] == 1:
            confusion_matrix[1][0] +=1

        elif model_train.predict(test_data.X[i]) == 1 and test_data.y[i] == 0:
            confusion_matrix[0][1] +=1


    print('Accuracy: '+ str(round(accuracy/len(test_data.X),6)) +' ('+str(accuracy) + ' out of ' + str(len(test_data.X)) +' correct)')
    print('\n')
    print(' prediction')
    print('   0  1')
    print('  -----')
    print('0| '+ str(int(confusion_matrix[0][0])) + ' ' + str(int(confusion_matrix[0][1])))
    print('1| '+ str(int(confusion_matrix[1][0])) + ' ' + str(int(confusion_matrix[1][1])))

    



if __name__ == "__main__" :
    main()
