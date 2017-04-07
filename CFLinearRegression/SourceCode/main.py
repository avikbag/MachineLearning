#!/usr/bin/env python

import numpy as np
from numpy import linalg as la
import os
from random import randint
from math import sqrt
import matplotlib.pyplot as plt
import operator
import copy
from CFLinearReg import cfLinearRegression as linReg
from SFoldCrossValid import SFoldCrossValidation as foldValid

def main():
    src = './x06Simple.csv'
    
    ## Question 1
    ## Creating Linear Regression object
    question2 = linReg(src)
    question2.shuffle()

    question2.training()
    question2.testing()

    theta = question2.theta()

    question2.prediction()
    
    ## Output Closed Form Linear Regression resulting RMSE
    RMSE = question2.RMSE()

    print "Theta from Cross Fold Linear Regression : \n{}\n".format(theta)
    print "RMSE from Closed Form Linear Regression : {}".format(RMSE)

    ## Question 2
    ## Creating S-Fold Cross Validation object
    
    question3 = foldValid(src)
    question3.shuffle()

    sfold = question3.crossValidation()

    print "RMSE from S-Fold Cross Validation : {}".format(sfold)


if __name__ == "__main__":
    main()
