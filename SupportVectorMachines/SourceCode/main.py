#!/usr/bin/env python

import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from SupportVectorMachine import SupportVectorMachine as svm
from MultiSupportVectorMachine import MultiSupportVectorMachine as msvm

def main():
    src = './spambase.data'
    q1 = svm(src)
    
    ## Setting up training and testing data sets
    q1.shuffle()
    xtraining, ytraining = q1.training()
    xtesting, ytesting = q1.testing()

    q1.support_vector_generation()
    q1.predict_values()
    q1.compute_errorTypes()
    print "Statistics for using in-built SVM library"
    print "(scikit-learn) for binary classification"
    print "__________________________________________"
    print "Precision: {0:.4f}".format(q1.precision())
    print "Recall   : {0:.4f}".format(q1.recall())
    print "F-Measure: {0:.4f}".format(q1.fMeasure())
    print "Accuracy : {0:.4f}\n".format(q1.accuracy())

    src = './CTG.csv'
    q2 = msvm(src)
    
    ## Setting up training and testing data sets
    q2.shuffle()
    xtraining, ytraining = q2.training()
    xtesting, ytesting = q2.testing()

    q2.classifiers()
    q2.support_vector_generation()
    q2.predict_values()
    success, fail = q2.compute_errorTypes()
    
    print "Multi Class support for in-built Binary SVM"
    print "Accuracy: {0:.4f}".format(float(success)/float(success+fail))


    

if __name__ == "__main__":
    main()
