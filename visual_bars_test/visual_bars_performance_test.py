import numpy as np 
import tensorflow as tf
from itertools import permutations
import tqdm #progress bar 

from cluster_methods import kmeans #clustering 
import core_cfl_objects.two_step_cfl as tscfl #CFL object 
from density_estimation_methods import condExp #density estimation 
import generate_visual_bars_data as vbd #visual bars data 

def find_best_unique_mapping(ground_truth, x_lbls): 
    '''checks all possible permutations of x_lbls against ground truth labels and chooses the best one 
    (highest degree of matches) as the 'correct' mapping. 
    
    NOTE: Currently labels based on visual bars data ([0, 1, 2, 3]) are hard-coded in'''
    #permute x_lbls 
    all_possible_value_orders = list(permutations([0, 1, 2, 3])) 

    #create dictionaries (mappings) with permuted x_lbls
    all_possible_mappings = [createLabelsDict(['0', '1', '2', '3'], permutation) for permutation in all_possible_value_orders] 
    bestAccuracy = 0

    #iterate over all x_lbl mapping choices and evaluate their accuracy against ground truth 
    for mapping in all_possible_mappings: 
        #find accuracy 
        currentAcc = accuracy(ground_truth, x_lbls, mapping)
        # store mapping w best accuracy 
        if currentAcc > bestAccuracy: 
            bestAccuracy = currentAcc
            bestMapping = mapping
    return bestMapping, bestAccuracy

def createLabelsDict(gt_labels, l):
    '''helper function for find_best_unique_mapping. Creates a mapping dictionary based on ''' 
    assert len(gt_labels) == len(l), "Both lists of labels must be the same length"

    D = {}
    for i in range(len(gt_labels)): 
        D[str(gt_labels[i])] = l[i] #cast gt_label as an immutable
    return D
    
    
    
def create_mapping_by_sampling(ground_truth, x_lbls, sample_len): 
    ''' by sampling the first few entries in ground_truth and x_lbls, 
    determines which class in ground_truth is most likely to correspond 
    to which class in x_lbls'''
    gtHead = ground_truth[0:sample_len]
    xHead = x_lbls[0:sample_len]

    mapping = {}
    for gt_label in [0, 1, 2, 3]: 
        # find the classes that correspond to the current ground truth label 
        currentClasses = np.asarray(gtHead==gt_label).nonzero()
        # figure out, of these, which class label in xHead is most common 
        options = [np.count_nonzero(xHead[currentClasses]==0), np.count_nonzero(xHead[currentClasses]==1), np.count_nonzero(xHead[currentClasses]==2), np.count_nonzero(xHead[currentClasses]==3)]
        mapping[str(gt_label)] = np.argmax(options)
    return mapping 

def accuracy(ground_truth, x_lbls, mapping): 
    '''checks the accuracy of the x_lbls against the ground_truth, according to the mapping bt ground_label class names and x_lbl class names'''
    #TODO: warning if not unqieu
    assert len(ground_truth) == len(x_lbls), "Ground truth and found x labels should be the same length"

    accuracy_counter = 0 
    #iterate over ground truth labels 
    for ground_truth_label in mapping.keys(): 
        ground_truth_label_int = int(ground_truth_label)
        # entries_in_gt_class = indices of all images which are a part of the current ground truth class
        entries_in_gt_class = np.where(ground_truth == ground_truth_label_int)
        #current_x_lbl = class label for x lbls that corresponds with current label for ground truth 
        current_x_lbl = mapping[ground_truth_label]

        # add up all correctly classified images in the current class 
        accuracy_counter += np.sum(x_lbls[entries_in_gt_class]==current_x_lbl)
    return accuracy_counter/len(ground_truth) #return percent images correctly classified 


def run_Visual_bars_test(sample_size, im_shape, noise_lvl, cluster_params, CDE_params): 
    vb_data = vbd.VisualBarsData(n_samples=sample_size, im_shape = im_shape, noise_lvl=noise_lvl)
    x = vb_data.getImages()
    y = vb_data.getTarget()
    
    #reformat x, y into the right shape for the neural net 
    y = np.expand_dims(y, -1)
    x = np.reshape(x, (x.shape[0], x.shape[1]*x.shape[2])) 
    data_info = {'X_dims': x.shape, 'Y_dims': y.shape} 

    # generate CDE object (with verbose mode off)
    condExp_object = condExp.CondExp(data_info, condExp_params, False)

    # generate clusterer 
    cluster_object = kmeans.KMeans(cluster_params)

    # put into a cfl core object 
    cfl_object = tscfl.Two_Step_CFL_Core(condExp_object, cluster_object)

    x_lbls, y_lbls = cfl_object.train(x, y)

    # check the results of CFL against the original 
    truth=vb_data.getGroundTruth().astype(int)
    mapping = find_best_unique_mapping(truth, x_lbls)[0]
    percent_accurate = accuracy(truth, x_lbls, mapping)
    return percent_accurate

def multiTest(n_trials, sample_sizes)  
  
    #number of trials to run for each sample 
    im_shape = (10, 10)
    noise_lvl= 0.05

    #clusterer params 
    cluster_params = {'n_Xclusters':4, 'n_Yclusters':4}

    for sample_size in sample_sizes: 
        for n in range(n_trials): 
            print("examining ", sample_size, "images for current run")
            print("trial", n+1, "of ", n_trials)
            print("image size is ", im_shape, "and noise level is", noise_lvl)

            # parameters for CDE 
            optimizer_Adam = tf.keras.optimizers.Adam(lr=1e-3)
            condExp_params = {'batch_size': 128, 'lr': 1e-3, 'optimizer': optimizer_Adam, 'n_epochs': 200, 'test_every': 10, 'save_every': 10}

            percent_accurate = run_Visual_bars_test(sample_size, im_shape, noise_lvl, cluster_params, condExp_params)
            print("percent accuracy is : ", percent_accurate)

