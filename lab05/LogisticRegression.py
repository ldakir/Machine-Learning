"""
Author: Lamiaa Dakir
Date: 10/21/2019
Description: Data and LogisticRegression classes
"""
# import libraries
import os
import numpy as np
from math import sqrt, exp, log

######################################################################
# classes
######################################################################

class Data :

    def __init__(self, X=None, y=None) :
        """
        Data class taken from lab #3
            X       -- numpy array of shape (n,p), features
            y       -- numpy array of shape (n,), targets
        """
        # n = number of examples, p = dimensionality
        self.X = X
        self.y = y


    def load(self, filename) :
        """
        Load csv file into X array of features and y array of labels.
        filename (string)
        """
        # determine filename
        dir = os.path.dirname(__file__)
        f = os.path.join(dir, '', filename)

        # load data
        with open(f, 'r') as fid :
            data = np.loadtxt(fid, delimiter=",")

        # separate features and labels
        self.X = data[:,:-1]
        self.y = data[:,-1]

def load_data(filename) :
    data = Data()
    data.load(filename)
    return data

class LogisticRegression:

    def __init__(self) :
        """
        Logistic regression for binary classification
        coef_ are the weights
        """
        self.coef_ = None

    def fit_SGD(self, X, y, alpha, eps=1e-2, tmax=100):
        """
        Finds the coefficients that fits the data using stochastic gradient descent.
        Parameters:
            X       -- numpy array of shape (n,p), features
            y       -- numpy array of shape (n,), targets
            alpha   -- float, step size
            eps     -- float, convergence criterion
            tmax    -- integer, maximum number of iterations
        """

        n,p = X.shape
        self.coef_ = np.zeros(p)                 # coefficients

        # SGD loop
        t = 0
        while True:
            t += 1
            previous_cost = self.cost(X,y) #calculating the cost
            y = y.reshape((n,1))
            X = np.concatenate((y,X),axis=1)
            np.random.shuffle(X)
            y = X[:,0]
            X = X[:,1:]
            # iterate through examples
            for i in range(n) :
                derivative = np.dot((self.hw_x(X[i])-y[i]),X[i])
                self.coef_ = self.coef_ - alpha*derivative

            current_cost = self.cost(X,y) #calculating the current cost
            if abs(current_cost-previous_cost) < eps or tmax ==t:
                break


    def hw_x(self, x) :
        """
        Hypothesis Function
        Parameters:
            x -- one example with p features
        Returns:
            the probability of a positive prediction for one example
        """
        hw = 1/ (1+exp(-np.dot(self.coef_,x)))
        return hw

    def predict(self,x):
        """
        Predicting the label
        Parameters:
            x -- one example with p features
        Returns:
            0 or 1  -- prediction of the label
        """
        y_pred = 1/ (1+exp(-np.dot(self.coef_,x)))
        if y_pred > 0.5:
            return 1
        else:
            return 0

    def cost(self, X, y) :
        """
        Calculate the cost.
        Parameters:
            X       -- numpy array of shape (n,p), features
            y       -- numpy array of shape (n,), targets
        Returns:
            cost    --  J(w)
        """

        cost = 0
        n,p = X.shape
        for i in range(n):
            h_w = self.hw_x(X[i])
            if h_w == 1:
                cost += 0
            else:
                cost += np.dot(y[i],log(h_w))+ (1-y[i])*log(1-h_w)

        cost = -cost
        return cost
