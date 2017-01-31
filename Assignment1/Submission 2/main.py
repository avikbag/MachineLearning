#!/usr/bin/env python

import numpy as np
from numpy import linalg as la
import pygal 
import os
from pca import PCA

## Method for normalizing data to match 
## pygal's plotting format
def plot(a, name, color, chart=None):
    if chart == None:
        chart = pygal.XY(stroke=False, 
                         xrange=(-6, 6),    
                         range=(-3.5, 4.5),
                         show_legend=False, 
                         title="PCA")

    src = []
    D = a.shape

    for i in range(0, D[0]):
        src.append({'value': (a[i,0]*-1,a[i,1]*-1), 'color': color})
    chart.add(name, src)
    chart.render_to_file('./output.svg')
    return chart
        

def main():
    src = './diabetes.csv'
    # Creates PCA object
    dr = PCA(src)


    # print dr.checkValid()
    # dr.load(src)
    # dr.output()
    # print dr.dimension()

    dr.standardize()
    dr.getCov()
    dr.eigen()
    dr.projectionMatrix(2)
    output = dr.PCA_projection()
    
    ## Convert output numpy.matrix to numpy.array
    ## to allow extracting -1, +1 classes
    output = np.array(output)
    
    ## Extracting the relevant classes
    class0 = np.around(output[output[:,0] == -1], decimals = 3)
    class1= np.around(output[output[:,0] == 1], decimals = 3)
    
    ## Plotting the results 
    fig = plot(class1[:,1:], "Class +1", 'red')
    fig = plot(class0[:,1:], "Class -1", 'blue', fig)


if __name__ == "__main__":
    main()
