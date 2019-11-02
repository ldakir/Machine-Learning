"""
Utils for ensemble methods.
Authors: Sara Mathieson + Lamiaa Dakir
Date: 11/1/2019
"""

from collections import OrderedDict
import optparse
# my files
from Partition import *

def parse_args(algorithm):
    """Parse command line arguments (train and test arff files)."""
    parser = optparse.OptionParser(description='run '+ algorithm +' method')

    parser.add_option('-r', '--train_filename', type='string', help='path to' +\
        ' train arff file')
    parser.add_option('-e', '--test_filename', type='string', help='path to' +\
        ' test arff file')
    parser.add_option('-T', '--classifier_nums', type='float', help='Number of classifiers')
    parser.add_option('-p', '--thresh', type='float', help='threshold')

    (opts, args) = parser.parse_args()

    mandatories = ['train_filename', 'test_filename', 'classifier_nums','thresh' ]
    for m in mandatories:
        if not opts.__dict__[m]:
            print('mandatory option ' + m + ' is missing\n')
            parser.print_help()
            sys.exit()

    return opts

def read_arff(filename):
    """Read arff file into Partition format."""
    arff_file = open(filename,'r')
    data = [] # list of Examples
    F = OrderedDict() # dictionary

    header = arff_file.readline()
    line = arff_file.readline().strip()

    # read the attributes
    while line != "@data":
        line = line.replace('{','').replace('}','').replace(',','')
        tokens = line.split()
        name = tokens[1][1:-1]
        features = tokens[2:]

        # label
        if name != "class":
            F[name] = features
        else:
            first = tokens[2]
        line = arff_file.readline().strip()

    # read the examples
    for line in arff_file:
        tokens = line.strip().split(",")
        X_dict = {}
        i = 0
        for key in F:
            val = tokens[i]
            X_dict[key] = val
            i += 1
        label = -1 if tokens[-1] == first else 1
        # set weight to None for now
        data.append(Example(X_dict,label,None))

    arff_file.close()

    # set weights on each example
    n = len(data)
    for i in range(n):
        data[i].set_weight(1/n)

    partition = Partition(data, F)
    return partition


def TPR(confusion_matrix):
    """Calculate the true positive rate."""
    TP = confusion_matrix[1][1] # true positive
    FN = confusion_matrix[1][0] # false negative
    return TP/(FN+TP)

def FPR(confusion_matrix):
    """Calculate the false positive rate."""
    FP = confusion_matrix[0][1] #false positive
    TN = confusion_matrix[0][0] # true negative
    return FP/(FP+TN)
