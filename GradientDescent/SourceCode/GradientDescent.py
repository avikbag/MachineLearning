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

class GradientDescent:
    
    ## This is the base constructor that is called. It can accept 
    ## a file path to the data source and also checks if path is 
    ## valid. 
    def __init__(self, csv=None, learning=0.01):
        self.csv = csv
        self.lr = learning
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
        np.random.shuffle(self.data[1:, :])
    
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

        rows = ceil((D[0]-1) * ratio) + 1 ## To account for header row
        self.training_y = self.data[1:rows, -1]
        temp, self.ave_training, self.std_training = self.standardize(self.data[1:rows, 1:-1])
        self.training = np.matrix(np.ones((rows - 1, D[1] - 1)))
        self.training[:, 1:] = temp
        return self.training, self.training_y
    
    ## Pulls out testing data
    def testing(self, ratio=(2./3.)):
        D = self.dimension()
        rows = ceil((D[0]-1) * ratio) + 1 ## To account for header row
        
        self.testing_y = self.data[rows:, -1]
        self.testing, _, _ = self.standardize(self.data[rows:, 1:-1], self.ave_training, self.std_training)
        
        return self.testing, self.testing_y

    def init_theta(self):
        random.seed(0)
        self.theta = [random.uniform(-1, 1) for i in range(0, self.testing.shape[1]+1)]
        self.theta = np.transpose(np.matrix(self.theta))
        return self.theta
    
    ## Computes the predicted value based on the regression model
    def prediction(self, data):
        rows = data.shape[0]
        columns = data.shape[1]
        
        temp = np.matrix(np.zeros((rows, 1)))

        for i in range(0, rows):
            total = self.theta[0, 0]
            for j in range(0, columns):
                total = total + (self.theta[j+1, 0] * data[i, j])
            temp[i, 0] = copy.deepcopy(total)
            total = 0
        
        return temp
	
    ## Computes the Root Mean Square Error
    def RMSE(self, y, y_hat):
        rows = y.shape[0]

        diff = y - y_hat

        sq = np.square(diff)

        total = np.sum(sq) / rows

        RMSE = sqrt(total)

	# print "RMSE : {}".format(RMSE)
        return RMSE
    
    def batch_gradient_descent(self):
        temp = []
        rows = self.training.shape[0]
        for j, item in enumerate(self.theta):
            cumulative_error = 0
            for i in range(0, rows):
                cumulative_error += (np.dot(self.training[i, :], self.theta) - self.training_y[i, 0]) * self.training[i, j]
            val = item[0, 0] - ( (self.lr / rows) * cumulative_error[0, 0])
            temp.append(val)
        
        self.theta = np.transpose(np.matrix(temp))

    def iterate(self, limit=1000000):
        training_RMSE_list = []
        testing_RMSE_list = []
        training_RMSE_past = 0
        for i in range(0, limit):
            ## RMSE training
            training_y_hat = self.prediction(self.training[:, 1:]) ## Need to exclude column of 1's
            training_RMSE = self.RMSE(self.training_y, training_y_hat)

            ## RMSE testing
            testing_y_hat = self.prediction(self.testing)
            testing_RMSE = self.RMSE(self.testing_y, testing_y_hat)
            
            ## Update theta using batch gradient descent
            self.batch_gradient_descent()

            training_RMSE_list.append(training_RMSE)
            testing_RMSE_list.append(testing_RMSE)
            
            if (np.spacing(1) > abs(training_RMSE_past - training_RMSE)):
                break;
            else:
                training_RMSE_past = training_RMSE

        return training_RMSE_list, testing_RMSE_list, (training_RMSE, testing_RMSE)

