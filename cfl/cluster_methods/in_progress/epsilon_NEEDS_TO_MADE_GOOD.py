#TODO: this code was transferred from old code, not modified at all 

import sys #write to stdout
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics

from scipy.stats import norm #calculating CDF
from scipy.states import entropy #calculate KL divergence

N_CLASSES = 4 #TODO: get rid of this



################################# EPSILON CLUSTERING STUFF ######################################
def do_epsilon_clustering(data, params):
    """
    Executes clustering based on the equivalence relation x1 ~ x2 iff P(Y|x1) = P(Y|x2) +- epsilon
    where epsilon is a small value

    Parameters:
        data
        epsilon
        n_runs (int)

    Returns:
        best (np array)
    """
    for n in range(params['n_runs']):
        current = epsilon_clustering_one_time(data, params['epsilon'])

    #pick best clustering result
    best = current # TODO: implement this
    return best


def update_center(class_center, class_array, n_members):
    """
    class_center = a single instance from the class_centers array
    class_array = array of all members currently assigned to a class
    """
    new_center = (class_center * (n_members-1) + class_array) / n_members
    return new_center

def distance(pyx1, pyx2):
    '''calculates euclidean distance between E[P(Y|X=x1)] and E[P(Y|X=x2)] '''
    return np.sqrt(np.sum(np.power(pyx1 - pyx2, 2)))


def epsilon_clustering_one_time(data, epsilon):

    # classes_array = np array where first dim is # of classes,
    # and then elements of each subarray are indices of data in that class
    classes_array = -1 * np.ones((data.shape[0],))

    # shuffle index array (faster than shuffling whole data set)
    index_array = np.random.shuffle(np.arange(data.shape[0]))

    # assign first data point to the first class
    classes_array[index_array[0]] = 0
    class_centers = data[0,:]

    # define the 'center' of the class
    for di in range(data.shape[0]):
        assigned = False
        for ci in range(class_centers.shape[0]):
            dist = distance(data[index_array[di]], class_centers[ci])
            if dist < epsilon:
                classes_array[index_array[di]] = ci
                assigned = True
                n_members = np.sum(classes_array==ci) + 1
                class_centers[ci] = update_center(class_centers[ci], data[index_array[di]], n_members)
                break # idea: check the goodness of this method by seeing how often a data
                      # point *would* have been assigned to more than one cluster if we didn't
                      # break (many times = epsilon too big or center calc not great)
        if not assigned:
            classes_array[index_array[di]] = class_centers.shape[0]
            class_centers = np.vstack([class_centers, data[index_array[di]]])
    return classes_array

    # for each datum:
    #   calc dist bt datum and center of first, second, etc...., class until find a class where dist < epsilon
    #   when datum has found its class:
        #   update the center of that class
        #   assign datum to the class
    #   if no class is close enough
    #   make a new class and put that datum in it
    #   move on to next data point


################################   KL Divergence (1D ONLY!) #################################

def calc_KLDiv(pyx1, pyx2, xData, n_bins):

    discretized_pyx1, discretized_pyx2 = discretized_pyx(pxy1, pyx2, xData, n_bins)
    kl = entropy(discretized_pyx1, discretized_pyx2)
    return kl

def discretized_pyx(pyx1, pyx2, xData, n_bins):
    ''' calculates discretized pmf over P(Y|X=x) for each x in xData'''

    #find min, max of xdata
    minX = np.min(xData)
    maxX = np.max(xData)

    #calculate bin edges
    bin_edges = np.linspace(start=minX, stop=maxX, num=n_bins)

    # calculate pmf
    discretized_pyx1 = bin_probabilities(bin_edges, pyx1)
    discretized_pyx2 = bin_probabilities(bin_edges, pyx2)
    return discretized_pyx1, discretized_pyx2

def cdf_from_GMM(x, alphas, mius, sigmas):
    '''
    estimates a CDF from the input GMM, for a given x

    Only works for 1D
    alphas = vector of weights
    mius = vector of means
    sigmas = vector of covars

    Returns
    mcdf = a float
    '''
    assert alphas.ndim == 1, "Alphas should be a 1D array"
    assert mius.ndim == 1, "Mius should be a 1D array"
    assert sigmas.ndim == 1, "Sigmas should be a 1D array"
    assert alphas.shape[0] == mius.shape[0] == sigmas.shape[0], "all inputs should be same length"

    mcdf = 0.0
    for i in range(len(alphas)):
        mcdf += alphas[i] * norm.cdf(x, loc=mius[i], scale=sigmas[i])
    return mcdf

def bin_probabilities(bin_edges, parameters):
    ''' calculate P(Y|X=x) within each x bin. '''
    # unpack parameters
    n_gaussians = parameters.shape[1] / 3
    assert n_gaussians.is_int(), 'pyx should be an array of format [mius, sigmas, alphas]'
    mius = parameters[:n_gaussians]
    sigmas = parameters[n_gaussians:2*n_gaussians]
    alphas = parameters[2*n_gaussians:]

    #calculate mcdf for each bin edge
    cdfs = np.array([cdf_from_GMM(x, alphas, mius, sigmas) for x in bin_edges])
    # subtract to find prob for inside each bin
    probs_array = np.zeros(len(cdfs)-1)
    for i in range(len(probs_array)):
        probs_array[i] = cdfs[i+1] - cdfs[i]
    # return: array with p for each bin (len= 1 shorter than bin_edges)
    return probs_array
