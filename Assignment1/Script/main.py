#!/usr/bin/env python

import numpy as np
from numpy import linalg as la
import pygal 
import os

class PCA:
    
    def __init__(self, csv=None):
        self.csv = csv
        ## Check if path given is valid
        try:
            self.data = np.matrix(np.genfromtxt(csv, delimiter=','))
        except TypeError as te:
            pass
    
    def checkValid(self):
        try:
            os.path.isfile(self.csv)
            # print "Data source successfully loaded from {}".format(self.csv)
            return True
        except TypeError as te:
            return False
    
    def load(self, csv):
        self.csv = csv
        try:
            self.data = np.matrix(np.genfromtxt(csv, delimiter=','))
            # print "Data source successfully loaded from {}".format(self.csv)
        except TypeError as te:
            print "No valid data source provided: ", te
    
    def output(self):
        print self.data

    def dimension(self):
        return self.data.shape

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
    
    def getCov(self):
        D = self.dimension()
        info = self.data[:,1:]
        infoT = info.transpose()
        # print infoT.shape
        self.covMat = np.dot(infoT, info) / (D[0] - 1)
        
        # print self.covMat

        return self.covMat

    def eigen(self):
        self.eigVal, self.eigVec = la.eig(self.covMat)

        # print "Eigen Values: {}\n\n".format(self.eigVal)
        # print "Eigen Vector: {}".format(self.eigVec)
    
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
        
    def PCA_reduction(self):
        D = self.dimension()
        k = self.pMat.shape[1]
        res = np.matrix(np.zeros((D[0], k + 1)))
        info = np.dot(self.data[:,1:], self.pMat)
        res[:,0] = self.data[:,0]
        res[:,1:] = info
        # print type(self.pMat)
        # print "\n"
        # print self.data[:,1:] * self.pMat
        return res

def plot(a, name, color, chart=None):
    if chart == None:
        chart = pygal.XY(stroke=False)
    src = []
    D = a.shape

    for i in range(0, D[0]):
        src.append({'value': (a[i,0]*-1,a[i,1]*-1), 'color': color})
    chart.add(name, src)
    chart.render_to_file('./output.svg')
    return chart
        

def main():
    src = './diabetes.csv'
    test = PCA()
    # print test.checkValid()
    test.load(src)
    # test.output()
    # print test.dimension()

    test.standardize()
    test.getCov()
    test.eigen()
    test.projectionMatrix(2)
    output = test.PCA_reduction()
    output = np.array(output)

    class0 = np.around(output[output[:,0] < 0], decimals = 3)
    class1= np.around(output[output[:,0] > 0], decimals = 3)
    

    fig = plot(class0[:,1:], "chart 0", 'blue')
    fig = plot(class1[:,1:], "chart 1", 'red', fig)


if __name__ == "__main__":
    main()
