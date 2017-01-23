#!/usr/bin/env python

import numpy as np
from numpy import linalg as la

def main():
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
    
    D = q1.shape
    ## for axis vars, 0 means over columns and 1 means over rows
    ave = q1.mean(0) ## This is used to find the mean over columns
    var = q1.var(0) ## This is used to find the var over columns

    print "Shape:\t\t{} \nMean:\t\t{} \nVariance:\t{}".format(D, ave, var)

    ## Standardizing data
    for i in range(0, D[1]):
        q1[:,i] = q1[:,i] - ave[0,i]
        q1[:,i] = q1[:,i] / var[0,i]
    
    q1 = np.around(q1, decimals=4)
    print "\nAfter Standardization: \n{}".format(q1)
    
    covMat = np.dot(q1.transpose(), q1) / (D[0] - 1)
    print "\nCovariance Matrix: \n{}\n".format(covMat)
    
    ## Eigen values and eigen vectors
    eigVal, eigVec = la.eig(covMat)
    print "Eigenvalues: \n{}\n".format(eigVal)
    print "Eigenvector: \n{}\n".format(eigVec)

if __name__ == "__main__":
    main()
