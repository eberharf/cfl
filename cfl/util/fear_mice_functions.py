import os

import numpy as np
import nibabel as nib

from cfl.util import brain_util as BU

def select_from_dict(dict, keywords):
    '''quick lil function to pull out only the arrays from a dict that match 
    a keyword (used to select only KO mris or only BL, etc). keywords is a list of 
    possible keywords (used to select based on an 'or' condition)
    
    Args: 
        dict (dict): a dictionary with string keys and array values 
        keywords (list of strings): list of strings that you want to find 
            in the keys of dict to return 
    Example: 
        wts, wt_names = select_from_dict(all_HM_dir, ['WT'])
    '''
    selected = []
    selected_names = []
    for key in dict: 
        for keyword in keywords: 
            if keyword in key: 
                selected.append(dict[key])
                selected_names.append(key)
    selected = np.array(selected)
    return selected, selected_names


def get_global_values(): 
    '''return mri dims, affine, and dir_labels (only
    valid for this dataset and RPS orientation!

    Intended usage: 
        mri_dir, mri_dims, affine, dir_labels = fm.get_global_values()
    
    '''

    mri_dir = r'PTSD_Data_Share\MEMRI_data'

    # load one image to get its dimensions
    img = BU.load_brain(os.path.join(mri_dir, "PTSD_KO_03_BL.nii"))
    mri_dims = img.shape

    # load one image to get affine
    nib_img = nib.load(os.path.join(mri_dir, "PTSD_KO_03_BL.nii"))
    affine = nib_img.affine

    # specify labels for plot (note the labels below are specifically for RAS orientation)
    dir_labels = { 'saggital' :   ['P', 'A', 'D', 'V'],
                'coronal' :    [' ', ' ', ' ', ' '],
                'horizontal' : ['L', 'R', 'A', 'P']} 

    return mri_dir, mri_dims, affine, dir_labels


def save_as_nifti(array, fname, mri_dims, affine):
    '''if you don't pass in a flat array, then this still should work equally 
    well. the unflatten just won't do anything'''
    if len(array.shape)==1: #if the array is flattened
        array = BU.unflatten(array, mri_dims)
    assert array.shape == mri_dims, "Array is of the wrong shape"
    array = np.nan_to_num(array) #change any NaNs in the array to 0s
    nifti_im = nib.Nifti1Image(array, affine=affine)
    nib.save(nifti_im, fname)


# function to subtract off the baseline
def remove_baseline(X, Y, id, timepoint):
    '''input: X (array of MRI images), Y (pandas dataframe with mouse ID/timepoint info),
     ID of mouse and timepoint to use ("PreF", "Fear", "D9")

    returns baseline-adjusted image
    '''
    # get image for that mouse at that timepoint
    image_index = Y.loc[(Y.ID==id) & (Y.Timepoint==timepoint)].index[0]
    image = X[image_index]

    # get image for that mouse at baseline
    baseline_index = Y.loc[(Y.ID==id) & (Y.Timepoint=="BL")].index[0]
    baseline_image = X[baseline_index]

    # subtract baseline from timepoint image
    adjusted_mri = image - baseline_image

    return adjusted_mri

def timepoint_indices_dir(Y):
    baseline_indices = Y[Y["Timepoint"]=="BL"].index.tolist()
    prefear_indices = Y[Y["Timepoint"]=="PreF"].index.tolist()
    postfear_indices = Y[Y["Timepoint"]=="Fear"].index.tolist()
    d9_indices = Y[Y["Timepoint"]=="D9"].index.tolist()
    timepoints_dir = {'BL' : baseline_indices, 'PreF': prefear_indices, "Fear": postfear_indices, 'D9': d9_indices}
    return timepoints_dir

# function to get the indices for each timepoint/genotype
def geno_time_indices_dir(Y):
    timepoints_dir = timepoint_indices_dir(Y)

    WT_indices = Y[Y["Genotype"]=="WT"].index.tolist()
    KO_indices = Y[Y["Genotype"]=="KO"].index.tolist()

    agg_dir = {}
    for timepoint in timepoints_dir:
        KO_key = timepoint + '_KO'
        KO_time_indices = list(set(KO_indices).intersection(set(timepoints_dir[timepoint])))
        agg_dir[KO_key] = KO_time_indices

        WT_key = timepoint + '_WT'
        WT_time_indices = list(set(WT_indices).intersection(set(timepoints_dir[timepoint])))
        agg_dir[WT_key] = WT_time_indices
    return agg_dir

def empty_geno_time_dir(Y, mri_dims):
    geno_time_dir = geno_time_indices_dir(Y)
    empty_dir = {}

    for key in geno_time_dir:
        empty_dir[key] = np.zeros(np.prod(mri_dims))
    return empty_dir

# function to calculate diffs?