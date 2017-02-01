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
from KMEAN import kmean as Kmean

def main():
    src = './diabetes.csv'
    tester = Kmean(src, 2)
    data = tester.extract([6, 7]) ## Add one to the index to account for header column
    tester.standardize()
    # tester.output()
    tester.rng()
    init = tester.init_seeds();

    a = np.array(data[:,1])
    b = np.array(data[:,2])
    
    plt.plot(b, a, 'rx')
    plt.plot(init[0][1], init[0][0], 'o', markeredgecolor='blue', markerfacecolor="None", markeredgewidth=1)
    plt.plot(init[1][1], init[1][0], 'o', markeredgecolor='blue', markerfacecolor="None", markeredgewidth=1)
    plt.savefig('init.png')

    plt.close()
    
    fin, init = tester.iterate()

    cluster1 = np.array(fin[0])
    cluster2 = np.array(fin[1])
    # cluster3 = np.array(fin[2])
    # cluster4 = np.array(fin[3])


    plt.plot(cluster1[:,1], cluster1[:,0], 'rx')
    plt.plot(cluster2[:,1], cluster2[:,0], 'bx')
    # plt.plot(cluster3[:,1], cluster3[:,0], 'gx')
    # plt.plot(cluster4[:,1], cluster4[:,0], 'cx')
    plt.plot(init[0][1], init[0][0], 'o', markeredgecolor='blue', markerfacecolor="None", markeredgewidth=1)
    plt.plot(init[1][1], init[1][0], 'o', markeredgecolor='blue', markerfacecolor="None", markeredgewidth=1)
    plt.savefig('iteration1.png')

    plt.close()

    fin, init, cout = tester.iterate_em()

    cluster1 = np.array(fin[0])
    cluster2 = np.array(fin[1])
    # cluster3 = np.array(fin[2])
    # cluster4 = np.array(fin[3])

    print cout

    plt.plot(cluster1[:,1], cluster1[:,0], 'rx')
    plt.plot(cluster2[:,1], cluster2[:,0], 'bx')
    # plt.plot(cluster3[:,1], cluster3[:,0], 'gx')
    # plt.plot(cluster4[:,1], cluster4[:,0], 'cx')
    plt.plot(init[0][1], init[0][0], 'o', markeredgecolor='blue', markerfacecolor="None", markeredgewidth=1)
    plt.plot(init[1][1], init[1][0], 'o', markeredgecolor='blue', markerfacecolor="None", markeredgewidth=1)
    plt.savefig('final.png')


if __name__ == "__main__":
    main()
