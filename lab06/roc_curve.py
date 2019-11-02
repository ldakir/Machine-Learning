"""
Run ensemble methods to create ROC curves.
Authors: Lamiaa Dakir
Date: 10/28/2019
"""
import util
from random_forest import *
from ada_boost import *
import optparse
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

# read in data (y in {-1,1})
train_partition = util.read_arff('data/mushroom_train.arff')
test_partition  = util.read_arff('data/mushroom_test.arff')

parser = optparse.OptionParser()
parser.add_option('-T', '--classifier_nums', type='int', help='Number of classifiers')
(opts, args) = parser.parse_args()
T = opts.classifier_nums

random_forest_FPRs = []
random_forest_TPRs = []

ada_boost_FPRs = []
ada_boost_TPRs = []

# Training and testing data using Random Forest
for t in np.arange(-0.1,1.1,0.1):
    threshold = t
    ensemble = random_forest_train_data(train_partition,T)
    confusion_matrix,FPR,TPR = random_forest_test_data(test_partition,ensemble,threshold)
    random_forest_FPRs.append(FPR)
    random_forest_TPRs.append(TPR)

# Training and testing data using AdaBoost
for t in np.arange(-0.1,1.1,0.1):
    threshold = t

    ensemble = ada_boost_train_data(train_partition,T)
    confusion_matrix,FPR,TPR = ada_boost_test_data(test_partition,ensemble,threshold)
    ada_boost_FPRs.append(FPR)
    ada_boost_TPRs.append(TPR)



plt.plot(ada_boost_FPRs,ada_boost_TPRs,'bo-',label ='AdaBoost')
plt.plot(random_forest_FPRs,random_forest_TPRs,'r*-',label ='Random Forest')


# title, axis labels, and legend
plt.title("ROC curve for Mushroom Dataset, T="+ str(T))
plt.legend()
plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.savefig('rf_vs_ada_T'+str(T)+'.png')
plt.show()
