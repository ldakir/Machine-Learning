"""
Utils for Naive Bayes.
Author: Sara Mathieson
Date: 9/10/19
"""

# python imports
from collections import OrderedDict
import math
import numpy as np
import optparse
import sys

# my file imports
from Partition import *

def parse_args():
    """Parse command line arguments (train and test arff files)."""
    parser = optparse.OptionParser(description='Use Naive Bayes to predict')

    parser.add_option('-r', '--train_filename', type='string', help='path to' +\
        ' train arff file')
    parser.add_option('-e', '--test_filename', type='string', help='path to' +\
        ' test arff file')

    (opts, args) = parser.parse_args()

    mandatories = ['train_filename', 'test_filename']
    for m in mandatories:
        if not opts.__dict__[m]:
            print('mandatory option ' + m + ' is missing\n')
            parser.print_help()
            sys.exit()

    return opts

def read_arff(filename):
    """
    Read arff file into Partition format. Params:
    * filename (str), the path to the arff file
    """
    arff_file = open(filename,'r')
    data = [] # list of Examples
    F = OrderedDict() # key: feature name, value: list of feature values

    header = arff_file.readline()
    line = arff_file.readline().strip()

    # read the attributes
    while line != "@data":

        clean = line.replace('{','').replace('}','').replace(',','')
        tokens = clean.split()
        name = tokens[1][1:-1]

        # discrete vs. continuous feature
        if '{' in line:
            feature_values = tokens[2:]
        else:
            feature_values = "cont"

        # record features or label
        if name != "class":
            F[name] = feature_values
        else:
            # first will be label -1, second will be +1
            first = tokens[2]
        line = arff_file.readline().strip()

    # read the examples
    for line in arff_file:
        tokens = line.strip().split(",")
        X_dict = {}
        i = 0
        for key in F:
            val = tokens[i]
            if F[key] == "cont":
                val = float(tokens[i])
            X_dict[key] = val
            i += 1

        label = int(tokens[-1])
        # add to list of Examples
        data.append(Example(X_dict,label))

    arff_file.close()

    F_disc = OrderedDict()
    for feature in F:
        F_disc[feature] = F[feature]

    partition = Partition(data, F_disc)
    return partition
