#!/usr/bin/env python

import numpy as np
from numpy import linalg as la
import pygal as py

def standardize(q1):
    D = q1.shape
    ## for axis vars, 0 means over columns and 1 means over rows
    ave = q1.mean(0) ## This is used to find the mean over columns
    std = q1.std(0, ddof=1) ## This is used to find the var over columns

    # print "Shape:\t\t\t{} \nMean:\t\t\t{} \nStandard Deviation:\t{}".format(D, ave, std)

    ## Standardizing data
    for i in range(0, D[1]):
        q1[:,i] = q1[:,i] - ave[0,i]
        q1[:,i] = q1[:,i] / std[0,i]
    
    q1 = np.around(q1, decimals=4)
    # print "\nAfter Standardization: \n{}\n".format(q1)
    
    return q1

def test():
    # q1 = np.matrix('''1. 2.; 
                    # 2. 3.;
                    # 3. 3.;
                    # 4. 5.;
                    # 5. 5.;
                    # 1. 0.;
                    # 2. 1.;
                    # 3. 1.;
                    # 3. 2.;
                    # 5. 3.;
                    # 6. 5.''')
    
    q1 = np.matrix('''4. 1.; 
                    2. 4.;
                    2. 3.;
                    3. 6.;
                    4. 4.;
                    9. 10.;
                    6. 8.;
                    9. 5.;
                    8. 7.;
                    10. 8.''')

    q1 = standardize(q1)

    print "The standardized matrix: \n{}\n".format(q1)
    
    covMat = np.dot(q1.transpose(), q1) / (10 - 1)
    covMat = np.around(covMat, decimals=3)
    print "\nCovariance Matrix: \n{}\n".format(covMat)

def question1():
    q1 = np.matrix('''-2. 1.; 
                    -5. -4.;
                    -3. 1.;
                    0. 3.;
                    -8. 11.;
                    -2. 5.;
                    1. 0.;
                    5. -1.;
                    -1. -3.;
                    6. 1.''')
    
    q1 = standardize(q1)
    D = q1.shape

    covMat = np.dot(q1.transpose(), q1) / (D[0] - 1)
    print "\nCovariance Matrix: \n{}\n".format(covMat)
    
    ## Eigen values and eigen vectors
    eigVal, eigVec = la.eig(covMat)
    print "Eigenvalues: \n{}\n".format(eigVal)
    print "Eigenvector: \n{}\n".format(eigVec)
    
    ## Using the max value of eigen value, use the corresponding 
    ## eigenvector for data projection
    res = np.dot(q1, eigVec[0])

    print "Result after PCA: \n{}\n".format(res)
    
    res2 = [(x, 0) for x in res]
    # print res2
    ## Plotting to XY chart
    chart = py.XY()
    chart.add("After PCA", res2)
    chart.render_to_file('./output.svg')

def question2():
    s0 = np.matrix('''-2. 1.; 
                    -5. -4.;
                    -3. 1.;
                    0. 3.;
                    -8. 11.''')
                   
    s1 = np.matrix('''-2. 5.;
                    1. 0.;
                    5. -1.;
                    -1. -3.;
                    6. 1.''')
    test = np.concatenate([s0, s1])
    print test
    # s1 = standardize(s1)
    # s2 = standardize(s2)
    
    print "Class 0: \n{}\n".format(s0)
    print "Class 1: \n{}\n".format(s1)
    
    # find the k possible values for features
    # this is done manually for now
    k0 = sorted([-2,1,-5,-4,-3,0,3,-8,11], key=int)
    k1 = sorted([-2,5,1,0,-1,-3,6], key=int) 

    print "Unique values in class 0: \n {}\n".format(k0)
    print "Unique values in class 1: \n {}\n".format(k1)
    
    ## For class 0
    feature_space0 = np.zeros((9, 2))
    ctr_p = ctr_n = 0
    index = 0

    # print np.asarray(s0[:,0].flatten()).tolist()[0].count(-8)
    for i in k0:
        # print i
        if np.asarray(s0[:,0].flatten()).tolist()[0].count(i) > 0:
            ctr_p += np.asarray(s0[:,0].flatten()).tolist()[0].count(i)
        if np.asarray(s0[:,1].flatten()).tolist()[0].count(i):
            ctr_n += np.asarray(s0[:,1].flatten()).tolist()[0].count(i)
        # print ctr_p, ctr_n
        feature_space0[index,:] = [ctr_p, ctr_n]
        ctr_p = ctr_n = 0
        index += 1

    print "Feature space 0: \n{}\n".format(feature_space0)
    

    ## For class 1
    feature_space1 = np.zeros((7, 2))
    ctr_p = ctr_n = 0
    index = 0

    # print np.asarray(s0[:,0].flatten()).tolist()[0].count(-8)
    for i in k1:
        # print i
        if np.asarray(s1[:,0].flatten()).tolist()[0].count(i) > 0:
            ctr_p += np.asarray(s1[:,0].flatten()).tolist()[0].count(i)
        if np.asarray(s1[:,1].flatten()).tolist()[0].count(i):
            ctr_n += np.asarray(s1[:,1].flatten()).tolist()[0].count(i)
        # print ctr_p, ctr_n
        feature_space1[index,:] = [ctr_p, ctr_n]
        ctr_p = ctr_n = 0
        index += 1

    print "Feature space 1: \n{}\n".format(feature_space1)

def lda():

    q1 = np.matrix('''-2. 1.; 
                    -5. -4.;
                    -3. 1.;
                    0. 3.;
                    -8. 11.;
                    -2. 5.;
                    1. 0.;
                    5. -1.;
                    -1. -3.;
                    6. 1.''')
    mean = q1.mean(0)
    std  = q1.std(0, ddof=1)

    q1 = standardize(q1)

    c0 = q1[:5,:]
    c1 = q1[5:,:]
    
    mean0 = c0.mean(0)
    std0 = c0.std(0, ddof=1)

    mean1 = c1.mean(0)
    std1 = c1.std(0, ddof=1)
    
    print "Mean: {}\t Std: {}\n".format(mean, std)
    print "Class 0: \n{} \n\nClass 1: \n{}\n".format(c0, c1)
    print "Class 0: \n\tMean: {}\n\tstd:{}".format(mean0, std0)
    print "Class 1: \n\tMean: {}\n\tstd:{}".format(mean1, std1)
    
    

    # Scatter Matrices
    covMat0 = np.dot(c0.transpose(), c0)
    print "\nScatter Matrix 0: \n{}\n".format(covMat0)

    covMat1 = np.dot(c1.transpose(), c1)
    print "\nScatter Matrix 1: \n{}\n".format(covMat1)

    # Within class scatter matrix
    scMat = covMat0 + covMat1
    scMat = np.around(scMat, decimals=3)
    scMatInv = la.inv(scMat)
    print "\nWithin Class Scatter Matrix: \n{}\n".format(scMat)
    print "\nWithin Class Scatter Matrix Inverse: \n{}\n".format(scMatInv)
    
    # Between class scatter matrix
    sb = np.dot((mean0 - mean1).transpose(), (mean0 - mean1))
    print "\nBetween Class Scatter Matrix: \n{}\n".format(sb)

    ## Eigen values and eigen vectors
    eigVal, eigVec = la.eig(scMatInv*sb)
    print "Eigenvalues: \n{}\n".format(eigVal)
    print "Eigenvector: \n{}\n".format(eigVec[1])
    
    W = np.matrix(eigVec[1])
    W = W.transpose()
    print W
    

def main():
    # question1()
    # question2()
    lda()

if __name__ == "__main__":
    main()
