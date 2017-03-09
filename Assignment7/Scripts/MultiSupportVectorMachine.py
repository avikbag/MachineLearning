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
from sklearn import svm
import itertools

class MultiSupportVectorMachine:
    
    ## This is the base constructor that is called. It can accept 
    ## a file path to the data source and also checks if path is 
    ## valid. 
    def __init__(self, csv=None):
        self.csv = csv

        ## Initializing error types
        self.success = 0
        self.fail = 0

        self.svm_obj = []
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
        # np.random.seed(0) ## Init seed to 0
        np.random.shuffle(self.data[2:, :])
    
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

        rows = ceil((D[0]-2) * ratio) + 2
        self.training_y = self.data[2:rows, -1]

        temp, self.ave_training, self.std_training = self.standardize(self.data[2:rows, :-2])
        f = temp.shape
        self.training = np.matrix(np.ones((f[0], f[1]+1)))
        self.training[:, :-1] = temp
        self.training[:, -1] = self.training_y
        
        return self.training, self.training_y
    
    ## Pulls out testing data
    def testing(self, ratio=(2./3.)):
        D = self.dimension()
        rows = ceil((D[0]-2) * ratio) + 2
        self.testing_y = self.data[rows:, -1]
        self.testing, _, _ = self.standardize(self.data[rows:, :-2], self.ave_training, self.std_training)
        
        return self.testing, self.testing_y

    ## Calculate Manhattan Distance 
    def dist_l1(self, a, b):
        total = 0
        D = len(a)

        for i in range(0, D):
            total += (abs(a[i] - b[i]))
        return total
    
    def find_majority(self, values):
        ctr = {}
        maximum = ( 0 , 0 )
        for i in values:
            if i in ctr: 
                ctr[i] += 1
            else: 
                ctr[i] = 1

            if ctr[i] > maximum[1]: 
                maximum = (i,ctr[i])

        return maximum[0]
    
    def classifiers(self):
        data = self.training_y.flatten().tolist()[0]
        sets = list(itertools.combinations(np.unique(data), 2))
        self.sets = sets

        # print self.sets

    ## Support vector generation is done by using the training data. The built
    ## in SVM class is available in python's scikit-learn. We will be using
    ## the linear model for classification. 
    def support_vector_generation(self):   
        for i in self.sets:
            
            data_feature = np.array(self.training)
            data_class = np.array(self.training_y)
            
            class0 = data_feature[data_feature[:, -1] == i[0]]
            class1 = data_feature[data_feature[:, -1] == i[1]]
            
            data_feature = np.concatenate((class0, class1), axis=0)

            class0 = data_class[data_class[:, -1] == i[0]]
            class1 = data_class[data_class[:, -1] == i[1]]
            
            data_class = np.concatenate((class0, class1), axis=0)
            data_class = data_class.flatten().tolist()
            
            temp = svm.SVC(kernel='linear')
            temp.fit(data_feature[:, :-1], data_class)
            
            ## Add svm object corresponding to each classifier
            self.svm_obj.append(temp)
    
    def predict_values(self):
        prediction = []
        rows = self.testing.shape[0]

        for i in self.svm_obj:
            temp = i.predict(self.testing)
            prediction.append(temp)
        
        output = np.transpose(np.matrix(prediction))
        self.predicted_y = []
        for i in range(0, rows):
            data = output[i,:].flatten().tolist()[0]
            self.predicted_y.append(self.find_majority(data))



	
    ## This method is used to calculate the basic error types
    ## based on the predicted values on the training data vs
    ## the actal testing data
    def compute_errorTypes(self):
        rows = self.testing.shape[0]
        
        for i in range(0, rows):
            testing = self.testing_y[i, 0]
            predicted = self.predicted_y[i]
            # print predicted

            if predicted == testing:
                self.success += 1
            else:
                self.fail += 1

        # print self.success, self.fail
        return self.success, self.fail
        # print self.tp, self.tn, self.fp, self.fn
