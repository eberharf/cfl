#Visualization Module 

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors



N_CLASSES = 4 #TODO: get rid of this 
X_COORDS_LEN = 9
Y_COORDS_LEN = 55
COORDS = {'y': np.linspace(10,-10, X_COORDS_LEN), 'x': np.linspace(142.5, 277.5, Y_COORDS_LEN)} #TODO: get rid of this 



#TODO: undo all the hard-coded numbers! 
#TODO: option to save plot instead of just showing it 

#TODO: do this 
# X_raw = x_scaler.inverse_transform(np.vstack([X_tr, X_ts]))
# Y_raw = y_scaler.inverse_transform(np.vstack([Y_tr, Y_ts]))


def visualize(X, Y, x_lbls, y_lbls):
    fig = plt.figure(figsize=(15,10), facecolor='white') # TODO: make figsize dynamically set
    visualize_helper(0, N_CLASSES, X, x_lbls,  np.linspace(-4,4,30)) #'in subplot 0, draw the x-figure'
    visualize_helper(1, N_CLASSES, Y, y_lbls, np.linspace(-3,5.5,30))# 'in subplot 1, draw the y-figure'
    fig.show()

def visualize_helper(col, n_classes, data, lbls, levels):
    for cluster_id in range(n_classes): 
        ax = plt.subplot2grid((4,2), (cluster_id, col)) 
        # Plot the cluster's mean difference from all frames' mean.
        cluster_mean = (data[lbls==cluster_id].mean(axis=0)-data.mean(axis=0)).reshape((Y_COORDS_LEN, X_COORDS_LEN)).T
        # TODO: pass in cmap
        ax.contourf(COORDS['x'].ravel(), COORDS['y'].ravel(), cluster_mean, levels=levels, cmap='BrBG_r')
        ax.set_xticks([]); ax.set_yticks([])
