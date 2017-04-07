#!/usr/bin/env python

import numpy as np
from numpy import linalg as la
import os

class PCA:
    
    ## This is the base constructor that is called. It can accept 
    ## a file path to the data source and also checks if path is 
    ## valid
    def __init__(self, csv=None):
        self.csv = csv
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
    
    ## This method is used to standardize the dataset that is provided. 
    ## this also avoids the first column as the header column
    def standardize(self):
        a = self.data[:,1:]
        D = self.dimension()
        ## for axis vars, 0 means over columns and 1 means over rows
        ave = a.mean(0) ## This is used to find the mean over columns
        std = a.std(0, ddof=1) ## This is used to find the var over columns

        # print "Shape:\t\t\t{} \nMean:\t\t\t{} \nStandard Deviation:\t{}".format(D, ave, std)
        ## Standardizing data
        for i in range(0, D[1] - 1):
            a[:,i] = a[:,i] - ave[0,i]
            a[:,i] = a[:,i] / std[0,i]

        a = np.around(a, decimals=4)
        
        self.data[:,1:] = a
        # print "\nAfter Standardization: \n{}\n".format(self.data)
        return self.data
    
    ## This method allows for calculating the covariance matrix. Normally
    ## done after standarization. 
    def getCov(self):
        D = self.dimension()
        ## Pulls out all the data except the header column
        info = self.data[:,1:]
        ## Transposes the info variable
        infoT = info.transpose()
        self.covMat = np.dot(infoT, info) / (D[0] - 1)
        
        return self.covMat
    
    ## Evaluates the eigenvalues and eigenvectors for the 
    def eigen(self):
        self.eigVal, self.eigVec = la.eig(self.covMat)

        # print "Eigen Values: {}\n\n".format(self.eigVal)
        # print "Eigen Vector: {}".format(self.eigVec)
    
    ## This method evaluates the projection matrix(W). This takes
    ## in a the argument for the dimensionality for the end result
    def projectionMatrix(self, k):
        D = self.dimension()
        pMat = np.matrix(np.zeros((D[1] - 1, k)))
        
        val = self.eigVal
        for i in range(0, k):
            index = np.argmax(val)
            # print self.eigVec[index]
            val[index] = -999999
            pMat[:,i] = self.eigVec[:,index]
        # print pMat
        self.pMat = pMat
        
    ## Projects the data onto the projection matrix(W)
    def PCA_projection(self):
        D = self.dimension()
        k = self.pMat.shape[1]
        res = np.matrix(np.zeros((D[0], k + 1)))
        info = np.dot(self.data[:,1:], self.pMat)
        res[:,0] = self.data[:,0]
        res[:,1:] = info

        return res

