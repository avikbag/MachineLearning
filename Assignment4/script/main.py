#!/usr/bin/env python

import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from GradientDescent import GradientDescent as gd

def main():
    src = './x06Simple.csv'
    learning_rate = 0.01
    q1 = gd(src, learning_rate)
    
    ## Setting up training and testing data sets
    q1.shuffle()
    q1.training()
    q1.testing()
    q1.init_theta()
    
    ## Running iterative processs limited to 1 million 
    ## iterations or change in training RMSE < eps
    training, testing, RMSE = q1.iterate()
    x = [x+1 for x in range(0, len(training))]
    
    ## Plotting 
    plt.plot(x, training, 'r-', label='Training Data')
    plt.plot(x, testing, 'b-', label='Testing Data')
    plt.title('Gradient Descent')
    plt.ylabel('RMSE of Training Data')
    plt.xlabel('Iterations')
    plt.legend(loc='upper right')
    plt.savefig('output.png')
    
    ## Results
    print "Number of iteratons: {}".format(len(training))
    print "Training RMSE: {} \nTesting RMSE: {}".format(RMSE[0], RMSE[1])

if __name__ == "__main__":
    main()
