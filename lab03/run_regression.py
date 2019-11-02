"""
Starter code authors: Yi-Chieh Wu, modified by Sara Mathieson
Authors: Lamiaa Dakir
Date: 09/25/2019
Description: modeling and fitting the training data and calculating the RMSE on the test data
"""

# This code was adapted from course material by Jenna Wiens (UMichigan).

# import libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import time

# import our model class
from PolynomialRegression import *

######################################################################
# main
######################################################################

def main() :
    # toy data
    X = np.array([2]).reshape((1,1))     # shape (n,p) = (1,1)
    y = np.array([3]).reshape((1,))      # shape (n,) = (1,)
    coef = np.array([4,5]).reshape((2,)) # shape (p+1,) = (2,), 1 extra for bias

    # load data
    train_data = load_data('regression_train.csv')
    test_data = load_data('regression_test.csv')


    ### ========== TODO : START ========== ###
    # part a: main code for visualizations
    print('Visualizing data...')
    #plot_data(train_data.X,train_data.y)
    #plot_data(test_data.X,test_data.y)





    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # parts b-e: main code for linear regression
    print('Investigating linear regression...')

    # model
    model = PolynomialRegression()

    # test part b -- soln: [[1 2]]
    print(model.generate_polynomial_features(X))


    # test part c -- soln: [14]
    model.coef_ = coef
    print(model.predict(X))

    # test part d, bullet 1 -- soln: 60.5
    print(model.cost(X, y))

    # test part d, bullets 2-3
    # for alpha = 0.01, soln: w = [2.441; -2.819], iterations = 616
    start_time = time.time()
    print('HERE')
    print(train_data.X)
    model.fit_SGD(train_data.X, train_data.y, 0.01)
    
    """print('sgd solution: %s' % str(model.coef_))
    SDG_time = time.time()

    # test part e -- soln: w = [2.446; -2.816]
    model.fit(train_data.X, train_data.y)
    print('closed_form solution: %s' % str(model.coef_))
    closed_form_time = time.time()

    print('Run time for SDG: ')
    print(SDG_time-start_time)

    print('Run time for closed_form solution: ')
    print(closed_form_time-SDG_time)



    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # parts f-h: main code for polynomial regression
    print('Investigating polynomial regression...')

    # toy data
    m = 2                                     # polynomial degree
    coefm = np.array([4,5,6]).reshape((3,))   # shape (3,), 1 bias + 3 coeffs

    # test part f -- soln: [[1 2 4]]
    model = PolynomialRegression(m)
    print(model.generate_polynomial_features(X))

    # test part g -- soln: 35.0
    model.coef_ = coefm
    print(model.rms_error(X, y))

    # non-test code (YOUR CODE HERE)

    train_model = PolynomialRegression(m=0)
    train_model.fit(train_data.X, train_data.y)
    d = [0,1,2,3,4,5,6,7,8,9,10]
    rmse_train = []
    rmse_test = []
    for i in d:
        train_model = PolynomialRegression(m=i)
        train_model.fit(train_data.X, train_data.y)
        rmse_train.append(train_model.rms_error(train_data.X, train_data.y))
        rmse_test.append(train_model.rms_error(test_data.X, test_data.y))



    plt.plot(d,rmse_test,'b')
    plt.title('RMSE for testing data')
    plt.xlabel('model complexity')
    plt.ylabel('RMSE')
    #plt.show()



    # Check: RMSE for d=0 should be 0.747268364185172
    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # parts i-j: main code for regularized regression
    print('Investigating regularized regression...')

    # test part i -- soln: [3 5.24e-10 8.29e-10]
    # note: your solution may be slightly different
    #       due to limitations in floating point representation
    #       you should get something close to [3 0 0]
    model = PolynomialRegression(m=2, reg_param=1e-5)
    model.fit(X, y)
    print(model.coef_)

    # non-test code (YOUR CODE HERE)

    ### ========== TODO : END ========== ###

    print(train_data.X)
    print(train_data.y)
    print("Done!")
"""

if __name__ == "__main__" :
    main()
