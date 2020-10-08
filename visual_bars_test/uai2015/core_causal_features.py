# Standard modules
import os
import sys
import pdb
import copy

# Numpy and Scipy
import numpy as np
import numpy
import scipy.optimize

# Pylearn2 and Theano
import theano
import theano.tensor as T
import pylearn2
import pylearn2.costs
import pylearn2.models.mlp as mlp
import pylearn2.train
import pylearn2.training_algorithms.learning_rule
import pylearn2.training_algorithms.sgd
import pylearn2.training_algorithms.bgd
import pylearn2.termination_criteria
import pylearn2.train_extensions.best_params
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest
from pylearn2.training_algorithms.sgd import ExponentialDecay, MonitorBasedLRAdjuster
from pylearn2.costs.cost import Cost, NullDataSpecsMixin
from pylearn2.models.mlp import Softmax

# Custom modules
import helpers

def compile_theano_costs(model, im_shape, n_classes=2, alpha=0.1, norm='L1'): 
    """ Compile the Theano manipulation cost functions 
    and their derivatives.  (compiling them here, out-of-the-loop, 
    makes the program run much faster). """
    # Defining a class object seems necessary to avoid
    # scoping problems when defining a list of functions
    # based on Theano compiled functions.
    class F(object):
        def __init__(self, theano_f):
            self.theano_f = theano_f

        def __call__(self, x, x_start):
            return self.theano_f(x.reshape(im_shape).astype('float32'),
                                 x_start.reshape(im_shape).astype('float32'))

    class dF(object):
        def __init__(self, theano_f):
            self.theano_f = theano_f

        def __call__(self, x, x_start):
            return self.theano_f(x.reshape(im_shape).astype('float32'),
                                 x_start.reshape(im_shape).astype('float32')).flatten()

    x_symb = model.input_space.make_batch_theano('xIn')
    x_start = model.input_space.make_batch_theano('xStart')
    n_pixels = np.product(im_shape)
    
    cost_list = []

    for class_id in range(n_classes):
        if norm=='L1':
            f_symb = (1-alpha)*(model.fprop(x_symb)[0, class_id]-1)**2 +\
                 alpha*((x_symb-x_start).norm(1))/n_pixels
        elif norm=='L2':
            f_symb = (1-alpha)*(model.fprop(x_symb)[0, class_id]-1)**2 +\
                 alpha*((x_symb-x_start).norm(2)**2)/n_pixels
        elif norm=='L1+L2':
            f_symb = (1-alpha)*(model.fprop(x_symb)[0, class_id]-1)**2 +\
                 alpha*(0.5*(x_symb-x_start).norm(2)**2+\
                        0.5*(x_symb-x_start).norm(1))/n_pixels
        f = theano.function(inputs=[x_symb, x_start], outputs=f_symb)
        grad_symb = theano.tensor.grad(f_symb, x_symb)
        grad = theano.function(inputs=[x_symb, x_start], 
                                outputs=grad_symb)

        cost_list.append((F(f), dF(grad)))

    return cost_list

def causal_coarsening(model_obs, data_obs_train, data_obs_valid, 
                                agent, n_queries=1, coarsen_thr=0.7, 
                                    train_params=None, img_shape=(10,10)):
    """ 
    Learn the causal coarsening of observational data, 
    and re-train an observational model on the causal data.

    Inputs:
    -------
    'model_obs': pylearn2 model trained on 'data_obs'
    'data_obs_train': pylearn2 dataset with observational labels
    'data_obs_valid': as above, valida tion set
    'agent': function to query behavior of an agent on an image
    'n_queries': number of queries to use from each class for each
                 inter-class boundary.
    'coarsen_thr': if more than this fraction out of 'n_queries' randomly
                   chosen points from two observational classes have 
                   the same causal labels, consider the classes causally
                   equivalent.   
    'train_params': dictionary of parameters for the train_mlp
    'img_shape': topological image size

    Outputs:
    --------
    'model_csl': pylearn2 model trained on causal data
    'data_csl_train': same data as data_obs_train, but with causal labels.
    'data_csl_valid': same as above but for data_obs_valid.
    """
    x_trn = data_obs_train.X
    y_trn = data_obs_train.y
    csl_classes = {}
    x_val = data_obs_valid.X
    y_val = data_obs_valid.y
    obs_labels = range(data_obs_train.y_labels)
    
    # Check if classes label_1 and label_2 are still separete.
    for label_id, y1 in enumerate(obs_labels[:-1]):
        for y2 in obs_labels[label_id+1:]:
            # Choose n_queries pts in class label_1.
            y1_csl = np.array([agent.behave(x_trn[i].\
                     reshape(img_shape)) for i in 
                     np.random.choice(np.where(y_trn==y1)[0], 
                                     n_queries)])
            y2_csl = np.array([agent.behave(x_trn[i].\
                     reshape(img_shape)) for i in 
                     np.random.choice(np.where(y_trn==y2)[0], 
                                      n_queries)])
            # If more than coarsen_thr queries have same labels, 
            # join the two classes.
            if (y1_csl == y2_csl).sum()/float(n_queries) > coarsen_thr:
                if y1 not in csl_classes:
                    csl_classes[y1] = y1
                if y2 not in csl_classes:
                    csl_classes[y2] = csl_classes[y1]
    
    # Assign the causal labels to the data.
    for label in csl_classes.keys():
        y_trn[y_trn==label] = csl_classes[label]
        y_val[y_val==label] = csl_classes[label]

    # Make sure the labels are consecutive integers.
    labels = np.sort(np.unique(y_trn))
    for lbl_id, lbl in enumerate(labels):
        y_trn[y_trn==lbl] = lbl_id
        y_val[y_val==lbl] = lbl_id
    data_train = pylearn2.datasets.DenseDesignMatrix(X=x_trn, y=y_trn,
                 y_labels=y_trn.max()+1)
    data_valid = pylearn2.datasets.DenseDesignMatrix(X=x_val, y=y_val,
                 y_labels=y_val.max()+1)
        
    # Retrain the model.
    train_params['data_train'] = data_train
    train_params['data_valid'] = data_valid
    train_params['data_test'] = copy.deepcopy(data_valid)
    train_params['model_fname'] =\
                        train_params['model_fname'][:-4]+'_csl.pkl'
    train_params['model'] = None #pylearn2.monitor.push_monitor(
    #    copy.deepcopy(model_obs), 'monitor', transfer_experience=True)
    
    if not os.path.isfile(train_params['model_fname']):
        train_mlp(**train_params)
    model_csl = pylearn2.utils.serial.load(train_params['model_fname'])

    print('Causal model trained. Validation set error {:.2g}'.\
          format(float(model_csl.monitor.channels['valid_y_misclass'].\
                       val_record[-1])))
    
    return (model_csl, data_train, data_valid)

def train_mlp(data_train, data_valid, data_test,
              model_fname='EXPERIMENTS/best_binary_mlp.pkl',
              clip_size=None, arch=[500,500], model=None,
              learning_rate=0.01, max_iters=250, 
              batch_size=100, max_after_best=10):
    """ Train a multilayer perceptron using MSE and weight decay. """
    print('Retraining the classifier...')
    in_dim = data_train.X.shape[1]
    
    monitoring_dataset = {'train':data_train, 
                          'valid':data_valid, 
                          'test':data_test}

    # Set up the learning machine or start from 
    # an already-trained one or load a pre-trained model.
    if model is None:
        layer_list = [mlp.RectifiedLinear(
            layer_name='h'+str(i), dim=arch[i], 
            sparse_init=max(1, min(in_dim, int(arch[i]/10))))
                                    for i in range(len(arch))]
        model = mlp.MLP(layers = layer_list + 
                        [mlp.Softmax(layer_name='y', 
                                     n_classes=data_train.y_labels, 
                                     irange=0.)],
                        nvis = data_train.X[0].size)
    else:
        del(model.layers[-1])
        model.layer_names.remove('y')
        model.add_layers([mlp.Softmax(layer_name='y', 
                                     n_classes=data_train.y_labels, 
                                     irange=0.)])

    # Construct the cost function.
    learning_cost = pylearn2.costs.cost.SumOfCosts(
        costs = [pylearn2.costs.mlp.Default(),
                 pylearn2.costs.mlp.WeightDecay(
                     coeffs=[.00005]*(len(arch)+1))])
    
    momentum = pylearn2.\
               training_algorithms.\
               learning_rule.\
               Momentum(init_momentum=0.5)

    # Construct the termination criterion.
    termination_criterion=pylearn2.\
        termination_criteria.And(criteria=[
            pylearn2.termination_criteria.MonitorBased(
                channel_name='valid_y_misclass', 
                prop_decrease=0., N=max_after_best),
            pylearn2.termination_criteria.EpochCounter(
                max_epochs=max_iters)])

    # Set up the optimization algorithm.
    algorithm = pylearn2.\
                training_algorithms.sgd.\
                SGD(batch_size=batch_size, 
                    learning_rate=learning_rate,
                    monitoring_batches=1,
                    monitoring_dataset=monitoring_dataset,
                    cost=learning_cost,
                    learning_rule=momentum,
                    termination_criterion=termination_criterion)

    # Choose extensions to the optimization algorithm.
    extensions = [
        pylearn2.train_extensions.best_params.\
            MonitorBasedSaveBest(channel_name='valid_y_misclass', 
                                 save_path=model_fname),
        pylearn2.training_algorithms.learning_rule.\
            MomentumAdjustor(start=1, saturate=10, final_momentum=0.99)]

    # Put everything together in a pylearn2 system.
    trainer = pylearn2.train.Train(dataset=data_train,
                                 model=model,
                                 algorithm=algorithm,
                                 extensions=extensions,
                                 save_freq=0)
    trainer.main_loop()
    print('Done.')

def manipulate_img(model, x_start, ftr_val, im_shape,
    starter_helper=None, augment_thr=0.1, augment_coeff=2., img_id=0, 
    manip_cost=None, D_manip_cost=None):
    """ 
    Manipulate image in x_start to have value 1 on the positive class
    (if ftr_val==1) or value 1 on the negative class (if ftr_val==0).
    This assumes net is a binary classifier with two softmax outputs.
    """
    # Theano expressions for the cost function and its gradient.
    x_start = x_start.flatten()
    x_symb = model.input_space.make_batch_theano('xIn')
    F = lambda x: manip_cost(x, x_start).astype('float64')
    dF = lambda x: D_manip_cost(x, x_start).astype('float64')

    # l-bfgs optimization with scipy.
    if starter_helper is None:
        #x0 = x_start
        x0 = np.random.rand(*x_start.shape).astype('float32')
    else:
        x0 = starter_helper.flatten()
    x_best = scipy.optimize.fmin_l_bfgs_b(func=F, x0=x0, fprime=dF,
                        bounds=[(0,1)]*x_start.shape[0], disp=0)[0].astype('float32')


    # Augment pixel changes larger than the threshold; supress all other changes.
    if augment_thr>0:
        delta = x_start-x_best
        small_change_ids = numpy.abs(delta)<augment_thr
        conf = model.fprop(theano.shared(x_best.reshape(im_shape))).eval()[0, ftr_val]
        print('Pre-augmentation activation {}'.format(conf))
        if augment_coeff < np.inf:
            x_best[delta>augment_thr]=x_best[delta>augment_thr]/augment_coeff
            x_best[delta<-augment_thr]=x_best[delta<-augment_thr]*augment_coeff
        else:
            x_best[delta>augment_thr]=0.
            x_best[delta<-augment_thr]=1.
        x_best[x_best>1] = 1.
        x_best[small_change_ids] = x_start[small_change_ids]

    conf = model.fprop(theano.shared(x_best.reshape(im_shape))).eval()[0, ftr_val]
    print('Manipulated image {:d}... Post-manipulation'.format(img_id) +
          ' activation {:.3g}\r'.format(conf)) 
    return (img_id, (x_best, conf, ftr_val))

def multiprocess_img_manipulations(model, im_shape, data_x, data_y, target_y, 
    starter_helper, absolute_ids, augment_threshold, augment_coeff, cost_list):
    """ 
    Perform image manipulations in parallel on a multicore machine.
    Return a dictionary of pool results.
    """
    print('\nManipulating images [multiprocessing]...')
    
    # Prepare the data.
    n_data = data_x.shape[0]
    manipulation_specs = []
    if isinstance(starter_helper, pylearn2.datasets.Dataset):
        flat_ys = np.where(starter_helper.y==1)[1]
    for img_id in range(n_data):
        if starter_helper is None:
            x_init = None
        elif isinstance(starter_helper, pylearn2.datasets.Dataset):
            x_init = _closest_from(data_x[img_id],
                starter_helper.X[np.where(flat_ys==target_y[img_id])[0]]).astype('float32')

        manipulation_specs.append({'model': model,
                             'im_shape': im_shape,
                             'x_start': data_x[img_id],
                             'ftr_val': target_y[img_id],
                             'starter_helper': x_init,
                             'augment_thr': augment_threshold,
                             'augment_coeff': augment_coeff,
                             'img_id': absolute_ids[img_id],
                             'manip_cost': cost_list[target_y[img_id]][0],
                             'D_manip_cost': cost_list[target_y[img_id]][1]})
    pool_results = [manipulate_img(**manip_spec) for manip_spec in manipulation_specs]
    
    return dict(pool_results)

def update_dataset(data_x, data_y, data_valid=None, data_train=None):
    """
    Create a Pylearn2 dataset consisting of causally annotated 
    manipulation results and an equal number of causal base points.
    """
    old_train_n = data_train.y.shape[0]
    old_valid_n = data_valid.y.shape[0]
    # Assemble data arrays from causal queries.
    data_x = np.concatenate((data_x, data_train.X))
    data_y = np.concatenate((data_y, data_train.y))

    if data_valid is not None:
        data_x = np.concatenate((data_x, data_valid.X))
        data_y = np.concatenate((data_y, data_valid.y))

    # Remove duplicates using Stackoverflow tricks (thanks Jaime)
    data_x_cpy = np.ascontiguousarray(data_x).view(
        np.dtype((np.void, data_x.dtype.itemsize * data_x.shape[1])))
    _, ids = np.unique(data_x_cpy, return_index=True)

    data_x = data_x[ids]
    data_y = data_y[ids]

    # Reshuffle.
    n_data = data_y.shape[0]
    ids = np.random.permutation(n_data)
    data_x = data_x[ids]
    data_y = data_y[ids]
    
    # Create Pylearn2 training and validation sets.
    data_train.set_design_matrix(data_x[:int(n_data*0.8)])
    data_train.y = data_y[:int(n_data*0.8)]
    data_valid.set_design_matrix(data_x[int(n_data*0.8):])
    data_valid.y = data_y[int(n_data*0.8):]
    print('Updated the data. Old data size (train/valid): {}/{}. New data size: {}/{}.'.\
        format(old_train_n, old_valid_n, data_train.y.shape[0], data_valid.y.shape[0]))
    return (data_train, data_valid)
        
def prepare_img_queries(data_train, model, n_queries, im_shape, alpha=0.1, augment_thr=0, augment_coeff=2.):
    """
    Choose images to manipulate and their target labels.
    """
    n_classes = data_train.y_labels
    q_per_class = int(float(n_queries)/n_classes)
    Xs = []
    ys = []
    target_ys = []
    results_dict = {}
    abs_ids = []
    for c_id in range(n_classes):
        if data_train.y.shape[1]==data_train.y_labels:
            # Data is in the one_hot format.
            flat_ys = np.where(data_train.y==1)[1]
        else:
            flat_ys = data_train.y
        ids = np.random.permutation(np.where(flat_ys==c_id)[0])[:q_per_class]
        abs_ids = abs_ids + list(ids)

        Xs.append(data_train.X[ids])
        ys.append(np.ones(q_per_class)*c_id)
        target_ys.append(np.random.choice(range(0,c_id)+range(c_id+1,n_classes), q_per_class))
    Xs = np.concatenate(Xs, axis=0)
    ys = np.concatenate(ys, axis=0)
    target_ys = np.concatenate(target_ys, axis=0)

    # Manipulate the queries.
    print('Compiling Theano costs...')
    cost_list = compile_theano_costs(model, im_shape, n_classes=n_classes, alpha=alpha, norm='L2')
    print('Done. Starting the manipulations pool!')
    manipulations_pool = multiprocess_img_manipulations(
        model, im_shape=im_shape, data_x=Xs, data_y=ys, target_y=target_ys, starter_helper=None, 
        absolute_ids = abs_ids,
        augment_threshold=augment_thr, augment_coeff=augment_coeff,
        cost_list=cost_list)
    print('Done')
    return manipulations_pool

def _closest_from(I, Is):
    """
    Choose the element of Is closest to I w.r.t. L1 norm.
    """
    closest_id = np.linalg.norm(Is-I, 1, axis=1).argmin()
    return Is[closest_id]

