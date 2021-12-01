import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from cfl.post_cfl.post_cfl_util import *
from tqdm import tqdm


def _kl_divergence(p, q):
    ''' Helper function for `discrimination_KL`. Computes the KL divergence
        in both directions and returns the mean.'''

    # TODO: document zeros_like caveat here
    kl_pq = np.sum(
        p * np.log2(np.divide(p, q, out=np.zeros_like(p), where=q != 0)))
    kl_qp = np.sum(
        q * np.log2(np.divide(q, p, out=np.zeros_like(q), where=p != 0)))
    return np.mean([kl_pq, kl_qp])


def discrimination_KL(fi_samples, fj_samples):
    ''' Compute the KL divergence between two samples.'''

    # make sure samples lie on 0-1 range for hist binning later
    assert (np.max(fi_samples) <= 1) and (np.min(fi_samples) >= 0)
    assert (np.max(fj_samples) <= 1) and (np.min(fj_samples) >= 0)

    # compute densities, add 1 to all counts to avoid zero entries in KL div
    pfi, bini = np.histogram(fi_samples, bins=20, range=(0, 1))
    pfj, binj = np.histogram(fj_samples, bins=20, range=(0, 1))
    pfi = (pfi + 1) / np.sum(pfi + 1)
    pfj = (pfj + 1) / np.sum(pfj + 1)
    assert len(pfi) == len(pfj)

    # compute kl divergence
    return _kl_divergence(pfi, pfj)


def discriminate_clusters(data, lbls, disc_func=discrimination_KL):
    ''' Compute how well each feature in data discriminates each pairwise
        class boundary using disc_func.

        Arguments:
            data (np.ndarray) : (n_samples, n_features) dataset
            lbls (np.ndarray) : (n_samples,) partition over `data`
            disc_func (function) : a function that takes two samples of 1D data
                                   and returns some distance between them
    '''

    # scale all features between 0-1 for distribution binning purposes
    data = data - np.min(data, axis=0)
    data = data / np.max(data, axis=0)

    # set variables
    n_features = data.shape[1]
    n_clusters = len(np.unique(lbls))
    disc_vals = np.zeros((n_clusters, n_clusters, n_features))

    # loop over every pair of clusters
    for ci in tqdm(range(n_clusters)):
        for cj in range(n_clusters):
            ci_samples = data[lbls == ci, :]
            cj_samples = data[lbls == cj, :]

            # if there are samples in both clusters, compute discriminability
            # of each feature
            if (ci_samples.shape[0] > 0) and (cj_samples.shape[0] > 0):
                for f in range(n_features):
                    disc_vals[ci, cj, f] = disc_func(
                        ci_samples[:, f], cj_samples[:, f])
            else:
                # mark empty cluster comparisons with -1
                disc_vals[ci, cj, :] = -1*np.ones((n_features,))
    return disc_vals


def plot_disc_vals(disc_vals, fig_path=None):

    n_features = disc_vals.shape[2]
    n_clusters = disc_vals.shape[0]
    feature_names = [f'feature {i}' for i in range(n_features)]
    cluster_names = [f'cluster {i}' for i in range(n_clusters)]
    fig, axs = plt.subplots(1, n_clusters, figsize=(n_clusters*4, 4))

    for c, ax in zip(range(n_clusters), axs.ravel()):
        dv = disc_vals[c, :, :]
        im = ax.imshow(dv, vmin=np.min(disc_vals), vmax=np.max(disc_vals))
        ax.set_xticks(range(n_features))
        ax.set_yticks(range(n_clusters))
        ax.set_xticklabels(feature_names, rotation=90)
        ax.set_yticklabels(cluster_names)
        ax.set_title(f'Cluster {c}')
    plt.colorbar(im)
    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches='tight')
    else:
        plt.show()


def compute_microvariable_importance(exp, data,
                                     dataset_name='dataset_train',
                                     cause_or_effect='cause'):
    if isinstance(data, str):
        data = np.load(data)
    macro_lbls = load_macrolbls(exp, dataset_name, cause_or_effect)
    exp_path = get_exp_path(exp)

    disc_vals = discriminate_clusters(data, macro_lbls)
    np.save(os.path.join(exp_path, dataset_name, 'microvariable_importance'),
            disc_vals)
    # plot_disc_vals(disc_vals,fig_path=os.path.join(
    #     exp_path, dataset_name, 'microvariable_importance'))
    return disc_vals
