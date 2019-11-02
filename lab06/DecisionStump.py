"""
Decision stump data structure (i.e. tree with depth=1), non-recursive.
Authors: Sara Mathieson + Lamiaa Dakir
Date: 10/27/2019
"""
import util
from Partition import *
import numpy as np

class DecisionStump:

    def __init__(self, partition):
        """
        Create a DecisionStump from the given partition of data by chosing one
        best feature and splitting the data into leaves based on this feature.
        """
        # key: edge, value: probability of positive (i.e. 1)
        self.children = {}

        # use weighted conditional entropy to select the best feature
        feature = partition.best_feature()
        self.name = feature

        # divide data into separate partitions based on this feature
        values = partition.F[feature]
        groups = {}
        for v in values:
            groups[v] = []
        for i in range(partition.n):
            v = partition.data[i].features[feature]
            groups[v].append(partition.data[i])

        # add a child for each possible value of the feature
        for v in values:
            new_partition = Partition(groups[v], partition.F)

            # weighted probability of a positive result
            prob_pos = new_partition.prob_pos()
            self.add_child(v,prob_pos)

    def get_name(self):
        """Getter for the name of the best feature (root)."""
        return self.name

    def add_child(self, edge, prob):
        """
        Add a child with edge as the feature value, and prob as the probability
        of a positive result.
        """
        self.children[edge] = prob

    def get_child(self, edge):
        """Return the probability of a positive result, given feature value."""
        return self.children[edge]

    def __str__(self):
        """Returns a string representation of the decision stump."""
        s = self.name + " =\n"
        for v in self.children:
            s += "  " + v + ", " + str(self.children[v]) + "\n"
        return s

    def classify(self, test_features, thresh=0.5):
        """
        Classify the test example (using features only) as +1 (positive) or -1
        (negative), using the provided threshold.
        """
        for x in test_features:
            if x == self.name:

                if self.children[test_features[x]] > thresh:
                    return 1
                else:
                    return -1



def main():
    opts = util.parse_args('')
    train_partition = util.read_arff(opts.train_filename)
    test_partition  = util.read_arff(opts.test_filename)

    for i in range(train_partition.n):
        example = train_partition.data[i]
        if i == 0 or i == 8:
            example.set_weight(0.25)
        else:
            example.set_weight(0.5/(train_partition.n-2))

    for x in train_partition.F:
        print(train_partition.gain(x))

    d = DecisionStump(train_partition)
    print(d)


    for x in test_partition.data:
        print(x.label,d.classify(x.features))


if __name__ == "__main__":
    main()
