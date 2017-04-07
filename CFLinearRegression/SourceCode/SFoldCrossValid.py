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

class SFoldCrossValidation:
    
    ## This is the base constructor that is called. It can accept 
    ## a file path to the data source and also checks if path is 
    ## valid. Can accept the number of folds intended, defaulted to 5 
    def __init__(self, csv=None, folds = 5):
        self.csv = csv
        self.folds = folds

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

        a = np.around(a, decimals=2)
        
        # print "\nAfter Standardization: \n{}\n".format(self.data)
        return a, ave, std
    
    # Pull out training data set, adds a column of 1 to the beginning
    def trainingSet(self, start, stop):
        D = self.dimension()
        # print "Start: {}, Stop: {}\n".format(start, stop)
        self.training_y = self.data[start:stop, -1]
        temp, self.ave_training, self.std_training = self.standardize(self.data[start:stop, 1:-1])
        self.training = np.matrix(np.ones((stop-start, D[1] - 1)))
        self.training[:, 1:] = temp
        # print "ave: - {}, std: - {}\n".format(self.ave_training, self.std_training)
        return self.training, self.training_y
    
    # Pulls out the testing data set 
    def testingSet(self, start, stop):
        D = self.dimension()
        
        temp = copy.deepcopy(self.data)
        temp = np.matrix(np.delete(temp, np.s_[start:stop], axis=0))
        
        self.testing, _, _ = self.standardize(temp[1:,1:-1], self.ave_training, self.std_training)
        self.testing_y = temp[1:, -1]

        
        return self.testing, self.testing_y
    
    ## Computes leat squares (theta)
    def thetaReg(self):
        xt = np.transpose(self.training)
        # print "XT: {}".format(xt)
        x = self.training
        # print "X: {}".format(x)
        y = self.training_y
        # print "Y: {}".format(y)

        A = la.inv(np.dot(xt, x))
        B = np.dot(xt, y)
        res = np.dot(A, B)
        
        res = np.around(res, decimals=3)
        self.theta = res
        

    def prediction(self):
        rows = self.testing.shape[0]
        columns = self.testing.shape[1]
        
        temp = np.matrix(np.zeros((rows, 1)))
        
        for i in range(0, rows):
            total = self.theta[0, 0]
            for j in range(0, columns):
                total = total + (self.theta[j+1, 0] * self.testing[i, j])
            temp[i, 0] = total
            total = 0
        
        self.y_hat = temp
    
    def difference2(self):
        rows = self.testing.shape[0]
        
        ## Returns the square difference between the 
        ## predicted data from the regression model 
        ## and the actual data
        diff = self.testing_y - self.y_hat
        sq = np.square(diff)

        return sq

    def crossValidation(self):
        rows = self.dimension()[0]- 1
        step = (rows-1) / self.folds
        mod  = (rows-1) % self.folds
        
        # print "Rows: {} \nSteps: {} \n Mod: {}\n".format(rows, step, mod)

        mse = []
        
        ## This iterative splices the different folds 
        ## and runs the regression model tests
        for i in range(0, self.folds):
            start = (i * step) + 1
            if (i == (self.folds - 1)):
                stop = start + step + mod + 1
            else:
                stop = start + step

            self.trainingSet(start, stop)
            self.testingSet(start, stop)
            self.thetaReg()
            self.prediction()
            temp = np.array(self.difference2()).flatten().tolist()
            mse.extend(temp)

        mean = np.mean(mse)
        RMSE = sqrt(mean)
        
        return RMSE


def main():
    src = 'x06Simple.csv'
    test = SFoldCrossValidation(src)
    test.shuffle()

    print test.crossValidation()


if __name__ == "__main__":
    main()
