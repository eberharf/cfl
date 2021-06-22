import matplotlib.pyplot as plt
import numpy as np 


def pyx_scatter(cfl_experiment, ground_truth=None): 
    '''creates a scatter plot with a sample of points from the CDE output,
    colored by ground truth (if given). 
    and also returns the average predictions for each
    CFL macro cause class
    
    Only good for 1D effect data 

    Example Usage: 

    ```
        fig = pyx_scatter(cfl_experiment, ground_truth)
        plt.show()
    ```

    Params: 
        cfl_experiment (cfl.experiment.Experiment): a trained CFL pipeline 
        ground_truth (np array): an array, aligned with the CFL training data 
            that contains the ground truth macrovariable labels for the cause data 
''' 

    fig  = plt.figure()
    pyx = cfl_experiment.retrieve_results('dataset_train')['CDE']['pyx'] # get training results 

    #choose a thousand (or the maximum possible) random samples from the pyx results
    n_samples = min(1000, pyx.shape[0])
    plot_idx = np.random.choice(pyx.shape[0], n_samples, replace=False)

    # scatter plot 
    if ground_truth is not None: 
        plt.scatter(range(n_samples), pyx[plot_idx,0], c=ground_truth[plot_idx]) # color by ground truth
    else: 
        plt.scatter(range(n_samples), pyx[plot_idx,0], c='m') # color magenta

    plt.ylabel("Expectation of Target")
    plt.xlabel("Sample")
    return fig

def cde_diagnostic(cfL_experiment): 
    '''Creates a figure to help diagnose whether the CDE is predicting the
target variable(s) effectively or should be tuned further 


    Creates a figure with three subplots
    First: actual distribution of the Y variable(s), according to the data 
    Second: predicted distribution of the Y variable(s), as output by the CDE 
    Third: difference between subplots 1 and 2

    This function may not work for higher dimensional continuous Ys. 

    # these should ideally look fairly similar if the CDE is doing well 
    creates a histogram if Y has type 'continuous', bar chart if Y has type
    'categorical'.
    Params: 
        cfl_experiment (cfl.experiment.Experiment) - a trained CFL pipeline 
    Returns 
        (Fig) - A `matplotlib.pyplot Figure` object that contains the diagnostic plot
        (Axes) - An array of `matplotlib.pyplot Axes` objects that are the
        subplots of the Figure object 

    '''

    Y = cfL_experiment.get_training_data().get_Y()
    pyx = cfl_experiment.retrieve_results('dataset_train')['CDE']['pyx'] # get training results 
    Y_type = cfL_experiment.get_data_info().get_Y()
    assert Y_type in ['categorical', 'continuous'], \
        'There is not a graphing method defined for the Y type of this training dataset'

    fig, axes = plt.subplots(nrows= 1, ncols=3, figsize=(16, 5), sharex=True, sharey=True) #figsize selected bc it worked well for one example plot


    if Y_type == 'continuous': 
        axes = __for_continuous_Y(Y, pyx, axes)
    if Y_type == 'categorical': 
        axes = __for_categorical_Y(Y, pyx, axes)

    axes[2].set_title("Difference between actual and expected values")

    return fig, axes  

def __for_continuous_Y(Y, pyx, axes): 
    """ 
    This method is for a Y that consists of continuous variable(s). 
    If Y contains a single variable, its distribution will be plotted as a
    single histogram. 


    If Y contains multiple variables, they will be plotted together as a
    side-by-side histogram (note: this is not be the best way to display the
    data for viewing)
    """         
    
    # for both the actual (Y) and predicted (pyx) values....
    for ax, values in zip(axes, (Y, pyx)):

        # get the global average value for each sample in Y 
        mean_values = values.mean(axis=0) #axis=0 is row-wise average

        # then add the values to a bar chart 
        ax.bar(range(len(mean_values)), mean_values) 
        ax.set_xlabel("Questions")

        
    actual_mean_values = Y.mean(axis=0)
    expected_mean_values = pyx.mean(axis=0)

    # then add the values to a bar chart 
    axes[2].bar(range(len(actual_mean_values)), expected_mean_values - actual_mean_values) 
    
    axes[2].set_xlabel("Questions")

    axes[0].set_title("Mean values of actual effect variables\n(from data)")
    axes[1].set_title("Mean values of predicted effect variables\n(output from CDE)")

    return axes



def __for_categorical_Y(Y, pyx, axes): 

    # plot the actual distribution of the effect variable(s)
    axes[0].hist(Y)

    # plot the CDE's predicted distribution of the effect (HoNOSCA) given the input 
    axes[1].hist(pyx)

    # difference between plots 0 and 1 
    axes[2].hist(Y - pyx) #I think Y and pyx should have the same shape

    axes[1].set_title("Distribution of Predicted Effect Variable\n(output from CDE)")
    axes[0].set_title("Distribution of Actual Effect Variable\n(from data)")

    return axes