#!/usr/bin/env python

import numpy as np
from numpy import linalg as la
import os
from random import randint
from math import sqrt
from math import ceil
from math import floor
import matplotlib.pyplot as plt
import operator
import copy
import time
import random

class KNearestNeighbors:
    
    ## This is the base constructor that is called. It can accept 
    ## a file path to the data source and also checks if path is 
    ## valid. 
    def __init__(self, csv=None, k=5):
        self.csv = csv
        self.k = k

        ## Initializing error types
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        ## Check if path given is valid
        try:
            self.data = np.matrix(np.genfromtxt(csv, delimiter=','))
        except TypeError as te:
            pass

    ## This checks if the data within the object is ready for 
    ## analysis to begin
    def checkValid(self):
        try:
            os.path.isfile(self.csv)
            # print "Data source successfully loaded from {}".format(self.csv)
            return True
        except TypeError as te:
            return False
    
    ## This allows you to load in a file into the object with 
    ## a valid path to csv data source 
    def load(self, csv):
        self.csv = csv
        try:
            self.data = np.matrix(np.genfromtxt(csv, delimiter=','))
            # print "Data source successfully loaded from {}".format(self.csv)
        except TypeError as te:
            print "No valid data source provided: ", te
    
    def output(self):
        print self.data
    
    ## Provides the number of rows and columns within the dataset
    def dimension(self):
        return self.data.shape
    
    # Shuffles the rows of the dataset
    def shuffle(self):
        np.random.seed(0) ## Init seed to 0
        np.random.shuffle(self.data)
    
    ## This method is used to standardize the dataset that is provided. 
    def standardize(self, a, ave=None, std=None):
        D = a.shape

        ## An added feature now is to be able to pass the average and std
        ## from another dataset, in this case, the training data.
        ## If there is no std or mean passed in the argument, it will then 
        ## compute the mean and std of the data set provided. 
        
        if (ave is None) and (std is None):
            ## for axis vars, 0 means over columns and 1 means over rows
            ave = a.mean(0) ## This is used to find the mean over columns
            std = a.std(0, ddof=1) ## This is used to find the var over columns

        # print "Shape:\t\t\t{} \nMean:\t\t\t{} \nStandard Deviation:\t{}".format(D, ave, std)
        ## Standardizing data
        a = np.subtract(a, ave)
        a = np.divide(a, std)

        a = np.around(a, decimals=4)
        
        # print "\nAfter Standardization: \n{}\n".format(self.data)
        return a, ave, std
    
    ## Pulls out training data
    def training(self, ratio=(2./3.)):
        D = self.dimension()

        rows = ceil(D[0] * ratio)
        self.training_y = self.data[:rows, -1]
        self.training, self.ave_training, self.std_training = self.standardize(self.data[:rows, :-1])
        return self.training, self.training_y
    
    ## Pulls out testing data
    def testing(self, ratio=(2./3.)):
        D = self.dimension()
        rows = ceil(D[0] * ratio)
        
        self.testing_y = self.data[rows:, -1]
        self.testing, _, _ = self.standardize(self.data[rows:, :-1], self.ave_training, self.std_training)
        
        return self.testing, self.testing_y

    ## Calculate Manhattan Distance 
    def dist_l1(self, a, b):
        total = 0
        D = len(a)

        for i in range(0, D):
            total += (abs(a[i] - b[i]))
        return total
    
    ## This method takes in the current point on the testing data
    ## and then compares it with the training data to find the nearest 
    ## k points. After which it will predict the value of the testing
    ## data point and return the predicted classification.
    def nearestNeighbors(self, current):
        rows = self.training.shape[0]
        dist = []
        one = 0 ## 1 counter
        zero = 0 ## 0 counter
        
        ## Populate list of distances to training nodes
        for i in range(0, rows):
            training = self.training[i,:].tolist()
            dist.append(self.dist_l1(current, training))

        for i in range(0, self.k):
            index, _ = max(enumerate(dist), key=operator.itemgetter(1))
            value = self.training_y[index,0] ## pull out training data classifier

            if(value == 1):
                one++
            elif (values == 0):
                zero++

            ## This checks whether a particular classifier is in majority
            if (one > (self.k/2)):
                return 1
            elif (zero > (self.k/2)):
                return 0
            else:
                pass
        
        ## Default case, if by any chance the majority checks fails
        return -1

    def knn_prediction(self):
        rows = self.testing.shape[0]
        predicted_vals = []

        for i in range(0, rows):
            ## Need to convert to list so that dist calc can handle it. Expects list.
            current = self.testing[i,:].tolist()[0]
            predicted_vals.append(nearestNeighbors(current))

        self.predicted_y = predicted_vals

    ## This method is used to calculate the basic error types
    ## based on the predicted values on the training data vs
    ## the actal testing data
    def compute_errorTypes(self):
        rows = self.testing.shape[0]

        for i in range(0, rows):
            testing = self.testing_y[i, 0]
            predicted = self.predicted_y[i]

            if (testing == 1) and (predicted == 1):
                self.tp++
            elif (testing == 0) and (predicted == 1):
                self.fp++
            elif (testing == 1) and (predicted == 0):
                self.fn++
            elif (testing == 0) and (predicted == 0):
                self.tn++
            else:
                pass

    def precision(self):
        return (self.tp)/(self.tp + self.fp)

    def recall(self):
        return (self.tp)/(self.tp + self.fn)

    def fMeasure(self):
        precision = (self.tp)/(self.tp + self.fp)
        recall = (self.tp)/(self.tp + self.fn)

        fm = (2 * precision * recall) / (precision + recall)

        return fm

    def accuracy(self):
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)


