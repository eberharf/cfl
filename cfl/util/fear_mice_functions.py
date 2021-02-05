import os

import numpy as np
import nibabel as nib

from cfl.util import brain_util as BU


def save_as_nifti(flat_array, fname, mri_dims, affine):
    flat_array = np.nan_to_num(flat_array)
    nifti_im = nib.Nifti1Image(BU.unflatten(flat_array, mri_dims), affine=affine)
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
    baseline_indices = Y[Y["Timepoint"]=="BL"].index.tolist()
    prefear_indices = Y[Y["Timepoint"]=="PreF"].index.tolist()
    postfear_indices = Y[Y["Timepoint"]=="Fear"].index.tolist()
    d9_indices = Y[Y["Timepoint"]=="D9"].index.tolist()
    timepoints_dir = {'BL' : baseline_indices, 'PreF': prefear_indices, "Fear": postfear_indices, 'D9': d9_indices}


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