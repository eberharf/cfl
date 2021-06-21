import matplotlib.pyplot as plt

def cde_diagnostic(cfL_experiment): 
    '''histogram with the distribution of the given effecvariable in 

    histogram if Y has type 'continuous', bar chart if Y has type 'categorical'.
    This function may not work for 
    First column: actual distribution of th 

    # these should ideally look fairly similar if the CDE is doing well 


    Returns 
        (Fig) - A `matplotlib.pyplot Figure` object that contains the diagnostic plot
        (Axes) - An array of `matplotlib.pyplot Axes` objects that are the
        subplots of the Figure object 

    '''

    Y = cfL_experiment.get_training_data().get_Y()
    pyx = cfL_experiment.get_training_results()['CDE']['pyx'] )
    Y_type = cfL_experiment.get_data_info().get_Y()
    assert Y_type in ['categorical', 'continuous'], \
        'There is not a graphing method defined for the Y type of this training dataset'

    fig, axes = plt.subplots(nrows= 1, ncols=3, figsize=(16, 5), sharex=True, sharey=True) #figsize selected bc it worked well for one example plot


    if Y_type == 'continuous': 
        __for_continuous_Y(Y, pyx, axes)
    if Y_type == 'categorical': 
        __for_categorical_Y(Y, pyx, axes)

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