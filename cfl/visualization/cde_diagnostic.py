'''
Contains two main functions: `pyx_scatter()` and `cde_diagnostic()` and helpers 
for those functions. 

These functions can be used to examine the quality of the CDE's learning
'''

import matplotlib.pyplot as plt
import numpy as np

def pyx_scatter(cfl_experiment, ground_truth=None, colored_by=None):
    '''
    Creates a scatter plot with a sample of points from the CDE output,
    colored by ground truth (if given). 

    Note: 
        This visualization method is only good for 1D effect data.

    Example Usage: 
    ```
        from cfl.visualization_methods import cde_diagnostic as cd 
        fig, ax = pyx_scatter(cfl_experiment, ground_truth)
        plt.show()
    ```

    Arguments: 
        cfl_experiment (cfl.experiment.Experiment): a trained CFL pipeline 
        ground_truth (np array): (Optional) an array, aligned with the CFL training data 
            that contains the ground truth macrovariable labels for the cause data.
            If provided, the points in the plot will be colored according to their
            ground truth state. Otherwise, all points will be colored the same. 

    Returns: 
        (Fig) - A `matplotlib.pyplot Figure` object that contains the scatter plot
        (Axes) - A `matplotlib.pyplot Axes` object that shows the scatter plot
    '''
    try:
        pyx = cfl_experiment.retrieve_results(
            'dataset_train')['CondDensityEstimator']['pyx']  # get training results
    except: # here for backwards compatibility with old results
        pyx = cfl_experiment.retrieve_results(
            'dataset_train')['CDE']['pyx']  # get training results

    # choose indices for a thousand (or the maximum possible) random samples from the pyx results
    n_samples = min(1000, pyx.shape[0])
    plot_idx = np.random.choice(pyx.shape[0], n_samples, replace=False)

    # make scatter plot
    fig, ax = plt.subplots(nrows=1, ncols=1)
    if ground_truth is not None:
        pyx_subset = pyx[plot_idx]
        gt_subset = ground_truth[plot_idx]
        ax = __pyx_scatter_gt_legend(ax, pyx_subset, gt_subset)
    else:
        ax.scatter(range(n_samples), pyx[plot_idx, 0], c='m')  # color magenta

    ax.set_ylabel("Expectation of Target")
    ax.set_xlabel("Sample")
    title = "Sample of predicted P(Y|X) values after CDE training"
    if colored_by is not None:
        title += f"\nColored by {colored_by}"
    ax.set_title(title)
    return fig, ax


def __pyx_scatter_gt_legend(ax, pyx, ground_truth_labels):
    '''
    Plots data from each ground_truth_class as a separate series in the
    scatter plot. Does this so that each label can be associated with a legend.
    '''
    # construct a list of all indices in the data
    all_indices = list(range(len(pyx)))

    # for each macrovariable class...
    for macrovar in np.unique(ground_truth_labels):

        # select the data points in pyx that belong to that class
        current_series = pyx[ground_truth_labels == macrovar]

        # select random values along the x-axis to plot those data against
        # (we do the selecting in this slightly convoluted way so that no x-value is chosen twice)
        current_indices = np.random.choice(
            all_indices, len(current_series), replace=False)
        indices = np.nonzero(np.in1d(all_indices, current_indices))[0]
        all_indices = np.delete(all_indices, indices)

        # scatter plot
        ax.scatter(current_indices, current_series, label=macrovar)
        ax.legend(bbox_to_anchor=(1, 1))  # place legend outside of figure
    return ax


def cde_diagnostic(cfl_experiment):
    '''
    Creates a figure to help diagnose whether the CDE is predicting the
    target variable(s) effectively or should be tuned further. 

    This function creates a figure with two subplots. The first shows the actual
    distribution of the Y variable(s), according to the data. The second shows
    the predicted distribution of the Y variable(s), as outputted by the CDE.

    If the effect data (`Y`) has type `continuous` (as specified in the
    `data_info` dictionary), then histograms showing the distribution of the
    effect variable are created. If Y is continuous
    and multidimensional, a stacked histogram with each feature is created. If Y has type
    `categorical`, bar charts with the mean values for each feature in Y are created. 

    If the CDE is doing a good job of learning the effect, the two subplots
    should contain similar or near-identical distributions.

    Note: 
        This function may not work for higher dimensional continuous Ys. 

    Arguments: 
        cfl_experiment (cfl.experiment.Experiment) - a trained CFL pipeline 
    Returns: 
        (Fig) - A `matplotlib.pyplot Figure` object that contains the diagnostic plot
        (Axes) - An array of `matplotlib.pyplot Axes` objects that are the
        subplots of the Figure object 

    Example Usage: 
    ```
        from cfl.visualization_methods import cde_diagnostic as cd 
        fig, axes = cd.cde_diagnostic(cfl_experiment)
        plt.show()
    ```
    '''

    Y = cfl_experiment.get_dataset('dataset_train').get_Y()
    pyx = cfl_experiment.retrieve_results(
        'dataset_train')['CDE']['pyx']  # get training results
    Y_type = cfl_experiment.get_data_info()['Y_type']
    assert Y_type in ['categorical', 'continuous'], \
        'There is not a graphing method defined for the Y type of this training dataset'

    if Y_type == 'continuous':
        fig, axes = __for_continuous_Y(Y, pyx)
    if Y_type == 'categorical':
        fig, axes = __for_categorical_Y(Y, pyx)

    return fig, axes


def __for_categorical_Y(Y, pyx):

    # figsize selected bc it worked well for one example plot
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(
        16, 5), sharex=True, sharey=True)

    # for both the actual (Y) and predicted (pyx) values....
    for ax, values in zip(axes, (Y, pyx)):

        # get the global average value for each sample in Y
        mean_values = values.mean(axis=0)  # axis=0 is row-wise average

        # then add the values to a bar chart
        ax.bar(range(len(mean_values)), mean_values)
        ax.set_xlabel("Questions")

    actual_mean_values = Y.mean(axis=0)
    expected_mean_values = pyx.mean(axis=0)

    # then add the values to a bar chart
    axes[2].bar(range(len(actual_mean_values)),
                expected_mean_values - actual_mean_values)
    axes[2].set_xlabel("Questions")

    axes[0].set_title("Mean values of actual effect variables\n(from data)")
    axes[1].set_title(
        "Mean values of predicted effect variables\n(output from CDE)")
    axes[2].set_title("Difference between actual and expected values")

    return fig, axes


def __for_continuous_Y(Y, pyx):
    """ 
    This method is for a Y that consists of continuous variable(s). 
    If Y contains a single variable, its distribution will be plotted as a
    single histogram. 

    If Y contains multiple variables, they will be plotted together as a
    stacked histogram. 
    """

    # figsize selected bc it worked well for one example plot
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(
        16, 5), sharex=True, sharey=True)

    # find the min and max values of both data sets (so that both histograms have
    # consistent bin boundaries)
    lower_bound = np.min(np.vstack((Y, pyx)))
    upper_bound = np.max(np.vstack((Y, pyx)))

    for ax, data in zip(axes, (Y, pyx)):
        # plot the actual distribution of the effect variable(s)
        ax.hist(data, range=(lower_bound, upper_bound),
                histtype='barstacked')  # (the default # of bins is 10)
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")

    axes[1].set_title(
        "Distribution of Predicted Effect Variable\n(output from CDE)")
    axes[0].set_title("Distribution of Actual Effect Variable\n(from data)")
    return fig, axes
