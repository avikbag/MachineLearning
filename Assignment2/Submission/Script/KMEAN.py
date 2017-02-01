#!/usr/bin/env python

import numpy as np
from numpy import linalg as la
import os
from random import randint
from math import sqrt
import matplotlib.pyplot as plt
import operator
import copy
import time

class kmean:
    
    ## This is the base constructor that is called. It can accept 
    ## a file path to the data source and also checks if path is 
    ## valid. Number of clusters is defaulted to 2, but can be set 
    ## to any value when calling constructor to object
    def __init__(self, csv=None, k=2):
        self.csv = csv
        self.clusters = []
        self.k = k
        self.ref = []

        ## Check if path given is valid
        try:
            self.data = np.matrix(np.genfromtxt(csv, delimiter=','))
        except TypeError as te:
            pass
        D = self.dimension()
        for i in range(0, k):
            self.clusters.append([])
    
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
    
    ## Extract desired dimensionality and column range. 
    ## Accepts a vector of indices to extract. Return 
    ## matrix includes the header column.
    def extract(self, vector):
        D = self.dimension()
        columns = len(vector)
        temp = np.matrix(np.zeros((D[0], columns + 1)))
        temp[:,0] = self.data[:,0]
        for i, val in enumerate(vector):
            ## Move to next column to account for header column
            temp[:,i+1] = self.data[:,val]
        self.data = temp

        return self.data

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
    
    ## Pulls out a random point from the data set
    ## as the random initial reference point
    def rng(self):
        rows = self.dimension()[0]
        index = randint(0, rows - 1)
        res = self.data[index, 1:].tolist()[0]
        return res ## Returns a list of (x,y) or any dimension of coordinates
    
    ## Expects a vector to calculate Euclidean Distance
    def dist(self, a, b):
        total = 0
        D = len(a)

        for i in range(0, D):
            total = total + ((a[i] - b[i])**2)
        return sqrt(total)
    
    ## Expects a vector to calculate Manhattan Distance
    def dist_l1(self, a, b):
        total = 0
        D = len(a)

        for i in range(0, D):
            total = total + (abs(a[i] - b[i]))
        return total

    ## Initializes the initial seed and then returns a
    ## vector of the points for the reference points
    def init_seeds(self):
        for i in range(0, self.k):
            self.ref.append(self.rng())
        return self.ref
    
    ## Picks out the closest point the the set of
    ## reference points
    def min_dist(self, a, reference):
        dist = []
        for i in reference:
            euclid_dist = self.dist(a, i)
            dist.append(euclid_dist)
            
        index, value = min(enumerate(dist), key=operator.itemgetter(1))
        return index, value
    
    ## This updates the reference points to the means of the values 
    ## of the given clusters
    def update_reference(self):
        for i in range(0, self.k):
            temp = np.array(self.clusters[i])
            mean = temp.mean(axis=0)
            self.ref[i] = mean
        
        return self.ref
    
    ## Expectation Maximization
    def em(self, old_ref, new_ref):
        val = 0
        for i in range(0, self.k):
            val = val + (self.dist_l1(old_ref[i], new_ref[i]))

        if val < np.spacing(1):
            return True

        return False

    ## Iterative process that allows you to run the clustering
    ## algorithm how many ever times you wish to run it. The default 
    ## iteration count is at 1, but can easily be passed as an argument
    ## to run how many ever iterations as required. Returns the resulting 
    ## clusters and the reference points used for those clusters.
    def iterate(self, lim=1):
        D = self.dimension()
        # print self.clusters
        ## Number of iterations for clustering
        for i in range(0, lim):
            ## Looping through the entire dataset to compare point to
            ## reference points 
            
            ## Reset clusters to account for new reference points
            self.clusters = []
            for reset in range(0, self.k):
                self.clusters.append([])

            for j in range(0, D[0]):
                point = self.data[j,1:].tolist()[0]
                index, value = self.min_dist(point, self.ref)
                self.clusters[index].append(self.data[j,1:].tolist()[0])
            
            ## Keep a track of the reference points used for current iteration
            temp = copy.deepcopy(self.ref)
            
            ## Update reference points for each cluster
            self.update_reference()

        return self.clusters, temp

    ## This method implements the expectation maximization as the
    ## break case. Once the criteria for the EM is met, it exits the
    ## iteration and returns the clusters, reference points and the number
    ## iterations it took. 
    def iterate_em(self):
        D = self.dimension()
        # print self.clusters
        ## Number of iterations for clustering
        EM = False
        count = 0
        while not EM:
            ## Looping through the entire dataset to compare point to
            ## reference points 
            
            ## Reset clusters to account for new reference points
            self.clusters = []
            for reset in range(0, self.k):
                self.clusters.append([])

            for j in range(0, D[0]):
                point = (self.data[j,1], self.data[j,2])
                index, value = self.min_dist(point, self.ref)
                self.clusters[index].append(self.data[j,1:].tolist()[0])
            
            ## Keep a track of the reference points used for current iteration
            temp = copy.deepcopy(self.ref)
            
            ## Update reference points for each cluster
            self.update_reference()
            
            ## Check Expectation Maximization
            ## At this point, self.ref contains the most up to date reference
            ## points, while temp contains the refrence points from the previous
            ## iteration
            EM = self.em(temp, self.ref)
            
            ## Keep track of number of iterations
            count = count + 1

        return self.clusters, temp, count

