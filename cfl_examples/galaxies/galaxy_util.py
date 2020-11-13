'''
Iman Wahle
11/13/2020
A somewhat random collection of data handling helper functions that
have come up while running cfl on galaxy data. These should eventually
be incorporated into cfl util code.
'''

import numpy as np

def reshape_input(vec, im_no):
    if im_no==0:
        return np.reshape(vec[:2601],(51,51))
    elif im_no==1:
        return np.reshape(vec[2601:],(51,51))
    elif im_no==2:
        return np.hstack([np.reshape(vec[:2601],(51,51)), np.reshape(vec[2601:],(51,51))])
    else:
        return

def calculate_arp(image): # arp = average radial profile = average 
    # source: https://stackoverflow.com/questions/48842320/what-is-the-best-way-to-calculate-radial-average-of-the-image-with-pythonf
    # create array of radii
    x,y = np.meshgrid(np.arange(image.shape[1]),np.arange(image.shape[0]))
    x0 = image.shape[1]//2
    y0 = image.shape[0]//2
    R = np.sqrt((x-x0)**2+(y-y0)**2)
    
    # calculate the mean
    eps = 0.5
    f = lambda r : image[(R >= r-eps) & (R < r+eps)].mean()
    r  = np.linspace(0.1,100,num=300)
    mean = np.vectorize(f)(r)

    return r,mean