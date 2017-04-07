#!/usr/bin/env python

import numpy as np
from numpy import linalg as la
import os
from random import randint
from math import sqrt
import matplotlib.pyplot as plt
import operator
import copy
from KMEAN import kmean as Kmean

def main():
    src = './diabetes.csv'
    ## Sets up Kmean object. Sets cluster number to 2 with data source 'src'
    hw2 = Kmean(src, 2)

    ## Extracts the desired columns for this assignment. 
    hw2.extract([6, 7]) ## Add one to the index to account for header column
    
    ## Standardize data
    data = hw2.standardize()
    
    ## Sets up initial seeds for reference points.
    init = hw2.init_seeds();

    a = np.array(data[:,1]) ## Standardized data for y-axis
    b = np.array(data[:,2]) ## standardized data for x-axis
    
    ## Plotting using matplotlib for Initial figure. Output is init.png
    plt.plot(b, a, 'rx')
    plt.plot(init[0][1], init[0][0], 'o', markeredgecolor='blue', markerfacecolor="None", markeredgewidth=1)
    plt.plot(init[1][1], init[1][0], 'o', markeredgecolor='blue', markerfacecolor="None", markeredgewidth=1)
    plt.title('Initial Seeds')
    plt.savefig('init.png')

    plt.close()
    
    ## Plotting for plot after 1st Iteration. Output is iteration1.png

    fin, ref = hw2.iterate(1)

    cluster1 = np.array(fin[0])
    cluster2 = np.array(fin[1])

    plt.plot(cluster1[:,1], cluster1[:,0], 'rx')
    plt.plot(cluster2[:,1], cluster2[:,0], 'bx')
    plt.plot(ref[0][1], ref[0][0], 'o', markeredgecolor='red', markerfacecolor="None", markeredgewidth=1)
    plt.plot(ref[1][1], ref[1][0], 'o', markeredgecolor='blue', markerfacecolor="None", markeredgewidth=1)
    plt.title('After 1 iteration')
    plt.savefig('iteration1.png')

    plt.close()

    ## Plotting for plot after n iterations using Expectation Maximization
    ## as the break case for the iterative process. Output is final.png
    fin, ref, count = hw2.iterate_em()

    cluster1 = np.array(fin[0])
    cluster2 = np.array(fin[1])

    plt.plot(cluster1[:,1], cluster1[:,0], 'rx')
    plt.plot(cluster2[:,1], cluster2[:,0], 'bx')
    plt.plot(ref[0][1], ref[0][0], 'o', markeredgecolor='red', markerfacecolor="None", markeredgewidth=1)
    plt.plot(ref[1][1], ref[1][0], 'o', markeredgecolor='blue', markerfacecolor="None", markeredgewidth=1)
    plt.title('Final Result. Iterations Required: {}'.format(count+1))
    plt.savefig('final.png')
    
    ## Output the number of iterations required to reach the EM. 
    print "Number of iterations to pass Expectation Maximization: ", count + 1

if __name__ == "__main__":
    main()
