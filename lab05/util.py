"""
Modified utils from lab #2.
Author: Sara Mathieson
Date: 9/10/19
"""

# python imports
import math
import numpy as np
import optparse
import sys


def parse_args():
    """Parse command line arguments (train and test arff files)."""
    parser = optparse.OptionParser(description='run decision tree method')

    parser.add_option('-r', '--train_filename', type='string', help='path to' +\
        ' train arff file')
    parser.add_option('-e', '--test_filename', type='string', help='path to' +\
        ' test arff file')
    parser.add_option('-a', '--alpha', type='float', help='alpha')

    (opts, args) = parser.parse_args()

    mandatories = ['train_filename', 'test_filename','alpha']
    for m in mandatories:
        if not opts.__dict__[m]:
            print('mandatory option ' + m + ' is missing\n')
            parser.print_help()
            sys.exit()

    return opts
