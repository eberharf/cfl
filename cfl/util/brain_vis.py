# Iman Wahle
# July-August 2020
# Functions to visualize lesion dataset and CFL results

# imports
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ipywidgets import interact, fixed, IntSlider

import cfl.util.brain_util as BU
# %matplotlib inline

# global variables
HOME_PATH = os.getcwd()

# note: this requires the full scan, not just the masked template area
def plot_panels(brains, dims, mask_vec, save_path):
    ''' plots 3 panels along the three axes of the 3D brain volumes.
    If given a list of brains, this function binarizes the sum
    along the given axis of each brain and plots the frequency of this
    binary map across all brains.
    If given a single brain, this function plots the raw summed intensities across each axis.
    arguments:
        brains: np array with shape of either [?, np.product(dims)] or
                [np.product(dims),]
        dims: 3D dimensions of arrays stored in brains
        mask_vec: flattened array of size np.product(dims) with raster of brain location
        interactive: whether to plot in interactive mode (boolean)
        save_path: path (from HOME_PATH) to save file
    returns:
        panels: array of the three panels (each a 2D array) in the non-interactive case,
                otherwise None
    '''



    panel0 = np.zeros((dims[1], dims[2]))
    panel1 = np.zeros((dims[0], dims[2]))
    panel2 = np.zeros((dims[0], dims[1]))
    panels = [panel0, panel1, panel2]

    fig,ax = plt.subplots(1,3, figsize=(20,5))

    for pdim in range(len(panels)):
        # are we doing population frequency, or one brain?
        if brains.ndim==1: # one brain case
            panels[pdim] = np.sum(BU.unflatten(brains, dims), axis=pdim).astype(np.float32)
        elif brains.ndim==2: # multiple brains case
            for bi in range(brains.shape[0]):
                panels[pdim] += np.sum(BU.unflatten(brains[bi], dims), axis=pdim) != 0
        else:
            print('Check input dimensions for brains!')
            return

        # set voxels outside of template region to not plot
        mask_regions = np.where(np.sum(BU.unflatten(mask_vec, dims),axis=pdim)==0)
        panels[pdim][mask_regions] = np.nan
        panels[pdim] = np.ma.masked_invalid(panels[pdim])
        cmap = plt.get_cmap()
        cmap.set_bad(color = 'w', alpha = 1.)

        im = ax[pdim].imshow(np.rot90(panels[pdim]))
        ax[pdim].axis('off')
        divider = make_axes_locatable(ax[pdim])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im,cax)
    plt.show()
    return panels


# note: this requires the full scan, not just the masked template area
def plot_interactive_panels(brains, dims, mask_vec, dir_labels, figsize=(6,4), std_scale='std', colormap='coolwarm', column_titles=[], step=1):
    ''' plots 3 panels along the three axes of the 3D brain volumes.
    If given a list of brains, it will plot the frequency of cell-wise
    binarization at a given slice along each axis determined by the interactive sliders
    If given a single brain, this function plots the raw
    intensities at a given slice along each axis determined by the interactive sliders.
    TODO: update this description
    arguments:
        brains: np array with shape of either [?, np.product(dims)] or
                [np.product(dims),]
        dims: 3D dimensions of arrays stored in brains
        mask_vec: flattened array of size np.product(dims) with raster of brain location
        std_scale: 'free'=no color range set, 'std'=all slices on the same color range,
                   'corr'=all slices on -1 to 1 color range
    returns:
        panels: array of the three panels (each a 2D array) in the non-interactive case,
                otherwise None
    '''

    # create a copy to not alter the original array (otherwise the original gets modified)
    brains_vis = np.asarray(brains, np.floating) #the array needs to be float type or else assigning the NaNs later won't work 

    # handle 1 brain case
    if brains_vis.ndim < 2:
        brains_vis = np.expand_dims(brains_vis, 0)

    # set voxels outside of template region to not plot
    assert ~np.all(np.equal(mask_vec, 0)), "please give a mask vector with at least one non-zero value" #to avoid accidentally submitting a blank mask
    mask_regions = np.where(mask_vec==0)[0]
    for bi in range(brains_vis.shape[0]): 
        brains_vis[bi,mask_regions] = np.nan 
    brains_vis = np.ma.masked_invalid(brains_vis)

    # volumes is list of unflattened mri images
    volumes = []
    for bi in range(brains_vis.shape[0]):
        volumes.append(BU.unflatten(brains_vis[bi], dims).astype(np.float32))

    vmin = None
    vmax = None
    if std_scale=='std':
        vmin = np.nanmin(volumes)
        vmax = np.nanmax(volumes)
    elif std_scale=='corr':
        mag = np.max([np.abs(np.nanmin(volumes)), np.abs(np.nanmax(volumes))])
        vmin = -mag
        vmax = mag
    interact(update, volumes=fixed(volumes), dim=fixed(0), vmin=fixed(vmin), vmax=fixed(vmax), figsize=fixed(figsize), dir_labels=fixed(dir_labels), colormap=fixed(colormap), column_titles=fixed(column_titles), brain_slice=IntSlider(min=0,max=dims[0]-1,step=step, continuous_update=False))
    interact(update, volumes=fixed(volumes), dim=fixed(1), vmin=fixed(vmin), vmax=fixed(vmax), figsize=fixed(figsize), dir_labels=fixed(dir_labels), colormap=fixed(colormap), column_titles=fixed([]), brain_slice=IntSlider(min=0,max=dims[1]-1,step=step, continuous_update=False))
    interact(update, volumes=fixed(volumes), dim=fixed(2), vmin=fixed(vmin), vmax=fixed(vmax), figsize=fixed(figsize), dir_labels=fixed(dir_labels), colormap=fixed(colormap), column_titles=fixed([]), brain_slice=IntSlider(min=0,max=dims[2]-1,step=step, continuous_update=False))


def update(volumes, dim, vmin, vmax, figsize, dir_labels, colormap, brain_slice, column_titles):
    fig,ax = plt.subplots(1, len(volumes),figsize=figsize)
    fig.subplots_adjust(right=0.8)
    cmap = plt.get_cmap()
    cmap.set_bad(color = 'w', alpha = 1.)

    # build a rectangle in axes coords
    left, width = .1, .8
    bottom, height = .1, .8
    right = left + width
    top = bottom + height
    hcenter = 0.5*(left+right)
    vcenter = 0.5*(top+bottom)

    keys = ['saggital', 'coronal', 'horizontal']

    if not isinstance(ax, np.ndarray):
        ax = np.array([ax])

    # i think this block of code graphs each column of mri slices (for each image)
    # print(len(column_titles))
    for bi in range(len(volumes)):
        [[x0, y0], [x1, y1]] = ax[bi].get_position().get_points()
        ax_nested = fig.add_axes([x0 + (x1-x0)*0.15, y0 + (y1-y0)*0.15, (x1-x0)*0.7, (y1-y0)*0.7])
        ax_nested.axis('off')
        im = ax_nested.imshow(np.rot90(np.take(volumes[bi], brain_slice, dim)), vmin=vmin, vmax=vmax, cmap=colormap)
        if len(column_titles) > 0:
            # print(bi)
            # print(column_titles[bi])
            ax[bi].set_title(column_titles[bi])
            # this doesn't really work right now as it goes under the slider :|
        ax[bi].text(left, vcenter, dir_labels[keys[dim]][0],
                horizontalalignment='right',
                verticalalignment='center',
                transform=ax[bi].transAxes,
                fontweight='bold')
        ax[bi].text(right, vcenter, dir_labels[keys[dim]][1],
                horizontalalignment='left',
                verticalalignment='center',
                transform=ax[bi].transAxes,
                fontweight='bold')
        ax[bi].text(hcenter, top, dir_labels[keys[dim]][2],
                horizontalalignment='center',
                verticalalignment='bottom',
                transform=ax[bi].transAxes,
                fontweight='bold')
        ax[bi].text(hcenter, bottom, dir_labels[keys[dim]][3],
                horizontalalignment='center',
                verticalalignment='top',
                transform=ax[bi].transAxes,
                fontweight='bold')


        ax[bi].axis('off')
    # else:
    #     im = ax.imshow(np.rot90(np.take(volumes[0],brain_slice, dim)), vmin=vmin, vmax=vmax)
    #     ax.axis('off')

    # fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    fig.canvas.draw()


def save_interactive_panels(brain_corrs, dims, mask_vec, save_path):

    os.mkdir(save_path)
    np.save(os.path.join(save_path, 'brain_corrs'), brain_corrs)
    np.save(os.path.join(save_path, 'dims'), dims)
    np.save(os.path.join(save_path, 'mask_vec'), mask_vec)

def load_interactive_panels(save_path):

    brains = np.load(os.path.join(save_path, 'brain_corrs.npy'))
    dims = np.load(os.path.join(save_path, 'dims.npy'))
    mask_vec = np.load(os.path.join(save_path, 'mask_vec.npy'))
    return brains, dims, mask_vec