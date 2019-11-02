"""
purpose
Author: Lamiaa Dakir
Date: 9/6/19
"""

import numpy as np
from math import sqrt
train_data = np.loadtxt("zip.train")
test_data = np.loadtxt("zip.test")

def distance(input1, input2):
    """
    Calculating the distance between the test input and the example input
    """
    sum = 0
    for i in range(len(input1)):
        sum += (input1[i] -input2[i])**2

    dist = sqrt(sum)
    return dist

def nearest_neighbors(test_input, train_input,k):
    """
    Finds the nearest k neighbors to one test input
    """
    test_label = test_input[0]
    test_values = test_input[1:]
    distances = np.zeros([len(train_input),2]) # has shape [(distance,label)]
    for i in range(len(train_input)):
        train_label = train_input[i][0]
        distances[i][0]= distance(train_input[i][1:],test_values)
        distances[i][1]= train_label

    #Find the k nearest neighbors
    knn =[]
    steps = 0
    while steps < k:
        #Loop through k times
        min_distance = distances[0][0]
        min_label = distances[0][1]
        index = 0
        #Find the minimum distance
        for i in range(len(distances)):
            if min_distance > distances[i][0]:
                min_distance = distances[i][0]
                min_label = distances[i][1]

                index= i
        distances = np.delete(distances,index,axis=0)
        knn.append(min_label)
        steps +=1
    return knn

def prediction(k, knn_labels):
    """
    Predict the label for an input test
    """
    dict ={}
    for x in knn_labels:
        if x in dict:
            dict[x] += 1
        else:
            dict[x] = 0

    highest = max(dict.values())

    for x in dict:
        if dict[x] == highest:
            result = x
    return result

#Filtering Data to only consider label 2 and 3
filtered_test_data =[]

for i in range(len(test_data)):
    if test_data[i][0] ==2 or test_data[i][0] ==3  :
        filtered_test_data.append(test_data[i])

filtered_train_data =[]
for i in range(len(train_data)):
    if train_data[i][0] ==2 or train_data[i][0] ==3  :
        filtered_train_data.append(train_data[i])

#Let's run the input tests for each k
print('Nearest Neighbors:')
for k in range(1,11):
    correct = 0
    total = 0

    for i in range(len(filtered_test_data)):
        knn = nearest_neighbors(filtered_test_data[i],filtered_train_data,k)
        answer = prediction(k,knn)
        if answer == filtered_test_data[i][0]:
            correct += 1
        total +=1
    accuracy = (correct/total)*100
    print('K = '+ str(k) +', '+ str(round(accuracy,3))+'%')
