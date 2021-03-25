# Iman Wahle
# July-August 2020
# Helper functions for processing 3D brain data

# imports
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import mode

import nibabel as nib


# global variables
HOME_PATH = os.getcwd()

def flatten(x):
    ''' flattens a 3D brain to 1D array
    arguments:
        x: 3D brain
    returns:
        new_x: 1D flattened brain
    '''
    new_x = np.reshape(x, (int(np.product(x.shape)),))
    return new_x


def unflatten(x, dims):
    ''' reshapes flattened brain into 3D brain
    arguments:
        x: 1D flattened brain
        dims: (3,) array of 3D brain dimensions
    returns:
        new_x: 3D brain
    '''
    new_x = np.reshape(x, dims)
    return new_x

# TODO: add apply_mask/unapply_mask helper functions

def load_brain(fp, to_flatten=False, mask=None, ori='RAS', dtype=np.float32):
    ''' loads one nii.gz file
    arguments:
        fp: file path (from HOME_PATH) to nii.gz file (string)
        to_flatten: whether to return as 3D array or 1D flattened array (boolean)
        mask: 3D array specifying voxels to retain. If none, retains all voxels
        ori: what order voxels should be oriented in the 3D array (i.e. 'LAS', 'RAS', etc.)
            'LAS' = right-Left within posterior-Anterior within inferior-Superior
            'RAS' = left-Right within posterior-Anterior within inferior-Superior
            more info here: http://www.grahamwideman.com/gw/brain/orientation/orientterms.htm
    returns:
        img: if flatten is true, a 1D array. Otherwise, a 3D array
    '''

    #get the image information from the specified path
    img = nib.load(os.path.join(HOME_PATH, fp))

    # convert the affine transformation matrix (a matrix) to axis codes (a string eg 'RAS')
    cur_ori = nib.orientations.aff2axcodes(img.affine)
    target_ori = tuple(ori)

    # check that we have a valid orientation
    assert cur_ori[0] in ['R', 'L'] # right, left
    assert cur_ori[1] in ['A', 'P'] # anterior, posterior
    assert cur_ori[2] in ['S', 'I'] # superior, inferior
    # TODO make these assert statements more useful ^


    # if the orientation of the image is different than the specified orentation,
    # flip one axis of the affine transformation
    if cur_ori[0] != target_ori[0]:
        img = img.slicer[::-1, :, :]
    if cur_ori[1] != target_ori[1]:
        img = img.slicer[:, ::-1, :]
    if cur_ori[2] != target_ori[2]:
        img = img.slicer[:, :, ::-1]

    # and check that the orientations are equal now
    cur_ori = nib.orientations.aff2axcodes(img.affine)
    assert (cur_ori == target_ori), "Problem with orientation: {}, {}".format(cur_ori, target_ori)

    # get the actual image array
    img = img.get_fdata()

    # flatten the image, and (if applicable) the mask, to 1D
    if to_flatten:
        img = flatten(img)
        if np.all(mask is not None):
            mask = flatten(mask)

    #if a mask was given, apply the mask to the image
    if np.all(mask is not None):
        img = img[np.where(mask)[0]]
        assert len(img) == np.sum(mask, dtype=np.float32)

    return img.astype(dtype)


def load_data(fpX, fpY, brain_dims, mask_path=None, ori='RAS', dtype=np.float32):
    ''' load all nii.gz files in a folder and corresponding test data
    arguments:
        fpX: path to folder (from HOME_PATH) of nii.gz files
        fpY: path to test csv file (from HOME_PATH)
        brain_dims: array containing [dim0, dim1, dim2] of 3D brain (int array)
        mask_path: nii.gz file path containing raster of what voxels to keep unmasked (string)
    returns:
        X: 2D matrix of size [n_brains, np.product(brain_dims)]
        Y: 2D matrix of size [n_brains, n_tests]
    '''
    assert os.path.splitext(fpY)[1] == '.csv', "{} is not a path to a .csv file".format(fpY)

    full_fpX = os.path.join(HOME_PATH, fpX)
    assert os.path.isdir(full_fpX), "Cannot find {} or is not a directory".format(full_fpX)

    first_file = os.listdir(full_fpX)[0]
    X_extension = os.path.splitext(first_file)[1]
    assert X_extension in ['.nii', '.nii.gz'], "Files in fpX should be NIFTI images (.nii or .nii.gz) but are {}".format(X_extension)

    #assumption: the format of Y is two or more columns: first column is IDs, rest of columns are data

    # load behavioral test data (Y)
    Yraw = pd.read_csv(os.path.join(HOME_PATH, fpY))

    # get the IDs
    id_key = Yraw.keys()[0]
    brain_names = Yraw[id_key]

    # return just the Ydata (no IDs)
    Y = Yraw.values[:, 1:]

    # load brain template mask
    if mask_path:
        mask = load_brain(os.path.join(HOME_PATH, mask_path), to_flatten=False, mask=None, dtype=dtype)
    else:
        mask = None

    # load lesion data (X)
    n_brains = len(Yraw)
    if mask_path:
        X = np.zeros((n_brains, int(np.sum(mask, dtype=np.float32))), dtype=dtype)
    else:
        X = np.zeros((n_brains, int(np.product(brain_dims))), dtype=dtype)


    for i, brain_name in enumerate(brain_names):
        fp = os.path.join(full_fpX, brain_name + X_extension)
        X[i, :] = load_brain(fp, to_flatten=True, mask=mask, ori=ori)

    return X, Y

def coarsen(x, n_voxels, brain_dims, method='majority'):
    ''' coarsens brain by specified amount
    arguments:
        x: 3D array of brain or 1D array of flattened brain
        n_voxels: how many original voxels to include in new voxel along a dimension
        brain_dims: array containing [dim0, dim1, dim2] of 3D brain (int array)
        method: how the value of the new coarsened pixel is decided. Either the most
        common value is chosen ('majority') or the average of all the values is chosen ('avg')
    returns:
        new_x: 3D array of coarsened brain
    '''
    # make 3D if flattened
    if x.ndim < 3:
        x = unflatten(x, brain_dims)

    new_dims = np.ceil(np.divide(brain_dims, n_voxels)).astype(int)

    new_x = np.zeros(new_dims)
    for i in range(new_dims[0]):
        for j in range(new_dims[1]):
            for k in range(new_dims[2]):
                # select all voxels to contribute to coarsened voxel (i,j,k)
                istart = i*n_voxels
                jstart = j*n_voxels
                kstart = k*n_voxels
                iend = np.min((istart+n_voxels, brain_dims[0])) # handle edge case
                jend = np.min((jstart+n_voxels, brain_dims[1]))
                kend = np.min((kstart+n_voxels, brain_dims[2]))

                sample = x[istart:iend, jstart:jend, kstart:kend]

                if method == 'majority':
                    # the most common value in the previous cube of voxels is
                    # the value of the new voxel
                    new_x[i, j, k] = mode(sample, axis=None)[0][0]

                    # take a vote
                    # new_x[i, j, k] = np.sum(sample == 1) > np.sum(sample == 0)
                elif method == 'avg':
                    # the avg value in the previous cube of voxels is
                    # the value of the new voxel
                    new_x[i, j, k] = np.average(sample)
    return new_x


def coarsen_dataset(X, n_voxels, brain_dims):
    ''' coarsen across several brains
    arguments:
        X: 2D matrix of flattened brains of size (n_brains, np.product(brain_dims))
        n_voxels: how many original voxels to include in new voxel along a dimension
        brain_dims: array containing [dim0, dim1, dim2] of 3D brain (int array)
    returns:
        new_X: 2D matrix of flattened coarsened brains of size (n_brains, np.product(new_dims))
        new_dims: 1D array of new 3D dimensions of each brain
    '''
    new_dims = np.ceil(brain_dims/n_voxels).astype(int)
    new_X = np.zeros((X.shape[0], np.prod(new_dims)))
    for xi in tqdm(range(X.shape[0])):
        new_X[xi, :] = np.reshape(coarsen(X[xi, :], n_voxels, brain_dims), (np.prod(new_dims),))
    return new_X, new_dims