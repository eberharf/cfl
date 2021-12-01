import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

''' Usage
from cfl.visualization_methods import macrostate_vis
data = an n_samples x an up to 3D shape for each sample
macrostate_vis(data=data, exp_id=0, cause_or_effect='cause', subtract_global_mean=True)

'''


def visualize_macrostates(exp_path, data, feature_names,
                          data_series='dataset_train', cause_or_effect='cause',
                          subtract_global_mean='True', figsize=None,
                          kwargs={}):

    if cause_or_effect == 'cause':
        fn = os.path.join(
            exp_path, f'{data_series}/CauseClusterer_results.pickle')
        with open(fn, 'rb') as f:
            lbls = pickle.load(f)['x_lbls']
    elif cause_or_effect == 'effect':
        fn = os.path.join(
            exp_path, f'{data_series}/EffectClusterer_results.pickle')
        with open(fn, 'rb') as f:
            lbls = pickle.load(f)['y_lbls']

    fig_path = os.path.join(exp_path, data_series,
                            f'{cause_or_effect}_macrostates')

    _plot(data, lbls, feature_names=feature_names,
          subtract_global_mean=subtract_global_mean,
          fig_path=fig_path, kwargs=kwargs)


def _plot(data, lbls, feature_names=None, dim_names=None,
          subtract_global_mean=True, fig_path=None, figsize=None, kwargs={}):

    u_lbls = np.unique(lbls)
    n_lbls = len(u_lbls)
    assert n_lbls > 1, 'Must have more than one macrostate'
    n_features = data.shape[1:]  # can be more than 1D
    global_mean = np.mean(data, axis=0)

    means = np.zeros(np.concatenate([[n_lbls], n_features]))
    for li in range(n_lbls):
        # compute mean and subtract global mean if specified
        mean = np.mean(data[lbls == u_lbls[li]], axis=0)
        if subtract_global_mean:
            mean = mean - global_mean
        means[li] = mean

    # compute color bounds
    if subtract_global_mean:  # symmetric colorbar centered at 0
        bound = np.max(np.abs([np.min(means), np.max(means)]))
        vmin, vmax = -bound, bound
        cmap = 'coolwarm'  # divergent
    else:  # keep colorbar constant across plots at least
        vmin, vmax = np.min(means), np.max(means)
        cmap = 'Blues'

    if len(data.shape[1:]) == 1:
        fig = _plot_1D(n_lbls, u_lbls, n_features, means, vmin, vmax, cmap,
                       feature_names, dim_names, figsize, kwargs)
    elif len(data.shape[1:]) == 2:
        fig = _plot_2D(n_lbls, u_lbls, n_features, means, vmin, vmax, cmap,
                       feature_names, dim_names, figsize, kwargs)
    elif len(data.shape[1:]) == 3:
        fig = _plot_3D(n_lbls, u_lbls, n_features, means, vmin, vmax, cmap,
                       feature_names, dim_names, figsize, kwargs)
    else:
        'No support for visualizing >3-dimensional samples'

    # save
    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches='tight')
    else:
        plt.show()


def _plot_1D(n_lbls, u_lbls, n_features, means, vmin, vmax, cmap, feature_names,
             dim_names, figsize=None, kwargs={}):
    # plot
    if figsize is None:
        figsize = (3*n_lbls, 3*np.ceil(n_features[0]/5))
    fig, ax = plt.subplots(1, n_lbls, figsize=figsize)
    for li in range(n_lbls):
        if 'cmap' not in kwargs.keys():
            im = ax[li].imshow(np.expand_dims(means[li], -1), vmin=vmin,
                               vmax=vmax, cmap=cmap, **kwargs)
        else:
            im = ax[li].imshow(np.expand_dims(means[li], -1), vmin=vmin,
                               vmax=vmax, **kwargs)
        ax[li].set_title(f'Macrostate {u_lbls[li]}')
        if dim_names is not None:
            ax[li].set_xlabel(dim_names[0])
        ax[li].set_xticks([])
        ax[li].set_yticks(range(n_features[0]))
        if feature_names is not None:
            if li == 0:
                ax[li].set_yticklabels(feature_names)
            else:
                ax[li].set_yticklabels([])

    # colorbar
    fig.subplots_adjust(right=0.9)
    # [left, bottom, width, height]
    cbar_ax = fig.add_axes([0.9, 0.3, 0.02, 0.6])
    fig.colorbar(im, cax=cbar_ax)

    return fig


def _plot_2D(n_lbls, u_lbls, n_features, means, vmin, vmax, cmap, feature_names,
             dim_names, figsize=None, kwargs={}):
    # plot
    if figsize is None:
        figsize = (3*np.ceil(n_features[1]/5)
                   * n_lbls, 3*np.ceil(n_features[0]/5))
    fig, ax = plt.subplots(1, n_lbls, figsize=figsize)
    for li in range(n_lbls):
        if 'cmap' not in kwargs.keys():
            im = ax[li].imshow(means[li], vmin=vmin, vmax=vmax, cmap=cmap,
                               **kwargs)
        else:
            im = ax[li].imshow(means[li], vmin=vmin, vmax=vmax, **kwargs)
        ax[li].set_title(f'Macrostate {u_lbls[li]}')
        if dim_names is not None:
            ax[li].set_xlabel(dim_names[1])
            ax[li].set_ylabel(dim_names[0])
        ax[li].set_xticks(range(n_features[1]))
        ax[li].set_yticks(range(n_features[0]))
        if feature_names is not None:
            ax[li].set_xticklabels(feature_names[1], rotation=45, ha='right')
            if li == 0:
                ax[li].set_yticklabels(feature_names[0])
            else:
                ax[li].set_yticklabels([])

    # colorbar
    fig.subplots_adjust(right=0.9)
    # [left, bottom, width, height]
    cbar_ax = fig.add_axes([0.9, 0.3, 0.02, 0.6])
    fig.colorbar(im, cax=cbar_ax)

    return fig


def _plot_3D(n_lbls, u_lbls, n_features, means, vmin, vmax, cmap, feature_names,
             dim_names, figsize=None, kwargs={}):
    # plot
    if figsize is None:
        figsize = (3*np.ceil(n_features[1]/5)*n_lbls,
                   3*np.ceil(n_features[0]/5)*n_features[2])
    fig, ax = plt.subplots(n_features[2], n_lbls, figsize=figsize)
    for fi in range(n_features[2]):
        for li in range(n_lbls):
            if 'cmap' not in kwargs.keys():
                im = ax[fi, li].imshow(means[li, :, :, fi], vmin=vmin, vmax=vmax,
                                       cmap=cmap, **kwargs)
            else:
                im = ax[fi, li].imshow(means[li, :, :, fi], vmin=vmin, vmax=vmax,
                                       **kwargs)
            if fi == 0:
                ax[fi, li].set_title(
                    f'Macrostate {u_lbls[li]}\n{feature_names[2][fi]}')
            else:
                ax[fi, li].set_title(feature_names[2][fi])
            if dim_names is not None:
                ax[fi, li].set_xlabel(dim_names[1])
                ax[fi, li].set_ylabel(dim_names[0])
            # TODO: how should we show the third dim_names?
            ax[fi, li].set_xticks(range(n_features[1]))
            ax[fi, li].set_yticks(range(n_features[0]))
            if feature_names is not None:
                if fi == n_features[2]-1:
                    ax[fi, li].set_xticklabels(
                        feature_names[1], rotation=45, ha='right')
                else:
                    ax[fi, li].set_xticklabels([])
                if li == 0:
                    ax[fi, li].set_yticklabels(feature_names[0])
                else:
                    ax[fi, li].set_yticklabels([])

    # colorbar
    fig.subplots_adjust(right=0.9)
    # [left, bottom, width, height]
    cbar_ax = fig.add_axes([0.9, 0.3, 0.02, 0.6])
    fig.colorbar(im, cax=cbar_ax)

    return fig
