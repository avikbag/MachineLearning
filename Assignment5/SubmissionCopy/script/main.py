#!/usr/bin/env python

import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from KNearestNeighbors import KNearestNeighbors as knn

def main():
    src = './spambase.data'
    q1 = knn(src)
    
    ## Setting up training and testing data sets
    q1.shuffle()
    xtraining, ytraining = q1.training()
    xtesting, ytesting = q1.testing()

    q1.knn_prediction()
    q1.compute_errorTypes()
    print "Precision: {}".format(q1.precision())
    print "Recall   : {}".format(q1.recall())
    print "F-Measure: {}".format(q1.fMeasure())
    print "Accuracy : {}".format(q1.accuracy())

    

if __name__ == "__main__":
    main()
