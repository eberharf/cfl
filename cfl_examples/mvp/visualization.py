'''
A module for Visualization methods of CFL results
Iman Wahle and Jenna Kahn
Aug 28 2020
'''
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


# global variables
X_COORDS_LEN = 9 #TODO: get rid of these (el nino specific) constants - pass them in to the Data object from user input
Y_COORDS_LEN = 55

#TODO: option to save plot instead of just showing it 

#TODO: do this 
# X_raw = x_scaler.inverse_transform(np.vstack([X_tr, X_ts]))
# Y_raw = y_scaler.inverse_transform(np.vstack([Y_tr, Y_ts]))

FIG_KWARGS = {'figsize' : (15,10), 'facecolor' : 'white'} #NOTE: these values are el nino-specific 
X_KWARGS = { 'cmap' : 'BrBG_r' }
Y_KWARGS = {'cmap' : 'coolwarm' }

def visualize(X, Y, x_lbls, y_lbls, fig_kwargs=None, X_kwargs=None, Y_kwargs=None):
    ''' Iterate over each cluster in X and Y to plot the cluster 
        average's difference from the global mean. Only good for spatially organized data 
        Arguments: 
            X : X dataset of dimensions [# observations, # features] (np.array)
            Y : Y dataset of dimensions [# observations, # features] (np.array)
            x_lbls : class label for each observation in X, of dimensions [# observations,] (np.array)
            y_lbls : class label for each observation in Y, of dimensions [# observations,] (np.array) 
            fig_kwargs : dictionary of keyword arguments to pass to plt.figure (specify appearance of plot)
            X_kwargs : dictionary of keyword arguments to pass to visualize_helper for the X plots
            Y_kwargs : dictionary of keyword arguments to pass to visualize_helper for the Y plots
        Returns: None
    '''
    fig = plt.figure(**fig_kwargs)
    visualize_helper(0, X, x_lbls, X_kwargs) #  np.linspace(-4,4,30)) #'in subplot 0, draw the x-figure'
    visualize_helper(1, Y, y_lbls, Y_kwargs) # np.linspace(-3,5.5,30))# 'in subplot 1, draw the y-figure'
    fig.show()

def visualize_helper(col, data, lbls, kwargs):
    ''' For a given variable (X or Y), plot each cluster average's difference from
        the global mean. 
        Arguments:
            col : which column in the figure to assign to this variable (int)
            data : X or Y dataset of dimensions [# observations, # features] (np.array)
            lbls : cluster labels of dimensions [# observations,] (np.array)
            levels : what levels to plot on contour plot (TODO: infer this from data)
            kwargs : dictionary of keyword arguments to pass to ax.contourf (specify appearance of plot)
        Returns: None
    '''
    n_classes = len(np.unique(lbls))
    for cluster_id in range(n_classes): 
        ax = plt.subplot2grid((4,2), (cluster_id, col)) 
        # Plot the cluster's mean difference from all frames' mean.
        cluster_mean = (data[lbls==cluster_id].mean(axis=0)-data.mean(axis=0)).reshape((Y_COORDS_LEN, X_COORDS_LEN)).T
        ax.contourf(range(Y_COORDS_LEN), range(X_COORDS_LEN), cluster_mean , **kwargs)
        ax.set_xticks([]); ax.set_yticks([])
