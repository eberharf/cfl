'''
Iman Wahle
11/13/2020
A collection of data handling helper functions that have come up while running 
cfl on galaxy data. These should eventually be generalized and incorporated into 
cfl util code if they seem useful for generic datasets.
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def reshape_input(vec, im_no):
    ''' This function reshapes one or both flattened galaxy images as 51x51 arrays. 
    The input 'vec' is the concatenation of both flattened input images, and 'im_no'
    specifies which image to return as 2D. 
    Arguments:
        vec : a (5202,) np.array where [:2601] is flattened image0 and [2601:] 
              is flattened image1
        im_no : an int indicating which image to return reshaped.
                im_no=0: return image0
                im_no=1: return image1
                im_no=2: return both images horizontally concatenated.
    Returns:
        reshaped image(s) as np.array of shape (51,51) or (51, 102) if im_no==2
    '''
    if im_no==0:
        return np.reshape(vec[:2601],(51,51))
    elif im_no==1:
        return np.reshape(vec[2601:],(51,51))
    elif im_no==2:
        return np.hstack([np.reshape(vec[:2601],(51,51)), np.reshape(vec[2601:],(51,51))])
    else:
        return

# reference: https://stackoverflow.com/questions/48842320/what-is-the-best-way-to-calculate-radial-average-of-the-image-with-pythonf
def calculate_arp(image): # arp = average radial profile
    ''' Compute the average pixel value within epsilon of radius r from the center
        of an 2D array. 
        Arguments: 
            image : 2D np.array (constants are currently optimal for array
                    size (51,51))
        Returns:
            r : np.array of distance from image center for each point in mean
            mean : average pixel value at corresponding radii specified in r
    '''

    # create array of radii
    x,y = np.meshgrid(np.arange(image.shape[1]),np.arange(image.shape[0]))
    x0 = image.shape[1]//2
    y0 = image.shape[0]//2
    R = np.sqrt((x-x0)**2+(y-y0)**2)
    
    # calculate the mean
    eps = 0.5
    f = lambda r : image[(R >= r-eps) & (R < r+eps)].mean()
    r  = np.linspace(0.1,40,num=200)
    mean = np.vectorize(f)(r)

    return r,mean


# source: https://stackoverflow.com/questions/38208700/matplotlib-plot-lines-with-colors-through-colormap
def multiline(xs, ys, c, ax=None, **kwargs):
    """Plot lines with different colorings

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """

    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale 
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.autoscale()
    return lc