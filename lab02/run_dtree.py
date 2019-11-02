"""
main class
Author: Lamiaa Dakir
Date: 18/09/2019
"""

import util
from DecisionTree import *

def main():

    opts = util.parse_args()
    train_partition = util.read_arff(opts.train_filename, True)
    test_partition  = util.read_arff(opts.test_filename, False)

    # Instance of the DecisionTree class from the train_partition
    train_tree = DecisionTree()
    x = train_tree.MakeSubTree(train_partition, train_partition.F, opts.depth,0)


    def remove_leaf(node, leaf):
        """
        Helper function that removes leafs we won't be using
        """
        for x in node.children:
            if node.children[x].value == None:
                leaf = leaf[1:]
            leaf= remove_leaf(node.children[x],leaf)
        return leaf

    def classify(tree,input_test,leaf):
        """
        function that predicts the label for the training data
        """
        test_features = input_test.features
        root = tree.value
        val = test_features[root]
        for x in tree.children:
            if x == val:
                if tree.children[x].value != None:
                    return classify(tree.children[x],input_test,leaf)
                else:
                    leaf = leaf[0]
                    return leaf
            else:
                if tree.children[x].value == None:
                    leaf = leaf[1:]


                if tree.children[x].value != None:
                    leaf = remove_leaf(tree.children[x],leaf)
        return leaf


    def classify_continuous(tree,input_test,leaf):
        """
        function that predicts the label for the testing data (i.e handles continuous features)
        """
        test_features = input_test.features
        root = tree.value
        if '<=' in root:
            root =root.split('<=',2)
            f = root[0]
            f_val = root[1]
            test_val = test_features[f]
            for x in tree.children:
                if test_val <= float(f_val):
                    if x == 'True':
                        if tree.children[x].value != None:
                            return classify_continuous(tree.children[x],input_test,leaf)
                        else:

                            leaf = leaf[0]
                            return leaf
                    else:
                        if tree.children[x].value == None:
                            leaf = leaf[1:]
                        else:
                            leaf = remove_leaf(tree.children[x],leaf)
                else:
                    if x == 'False':
                        if tree.children[x].value != None:
                            return classify_continuous(tree.children[x],input_test,leaf)
                        else:
                            leaf = leaf[0]
                            return leaf
                    else:
                        if tree.children[x].value == None:
                            leaf = leaf[1:]
                        else:
                            leaf = remove_leaf(tree.children[x],leaf)
        else:
            val = test_features[root]
            for x in tree.children:
                if x == val:
                    if tree.children[x].value != None:
                        return classify_continuous(tree.children[x],input_test,leaf)
                    else:
                        leaf = leaf[0]
                        return leaf
                else:
                    if tree.children[x].value == None:
                        leaf = leaf[1:]


                    if tree.children[x].value != None:
                        leaf = remove_leaf(tree.children[x],leaf)
        return leaf


    #Running the classify function on the test data

    correct = 0
    for i in range(len(test_partition.data)):
        leaf = train_tree.leaf
        example = test_partition.data[i]
        expected_label = example.label
        if classify_continuous(x,example,leaf) == str(expected_label):
            correct +=1

    print(str(correct)+ ' out of ' +str(len(test_partition.data))+' correct')
    print('accuracy = '+str(round(correct/len(test_partition.data),4)))


main()
