""" Machine Learning and Causal Discovery routines. 
Krzysztof Chalupka, July 2016.
"""
import sys
import joblib
import numpy as np
import theano
import theano.tensor as T
import lasagne

sys.setrecursionlimit(10000)
BATCH_SIZE = 32
def logsumexp(x, axis=None, keepdims=True):
    xmax = T.max(x, axis=axis, keepdims=True)
    preres = T.log(T.sum(T.exp(x-xmax), axis=axis, keepdims=keepdims))
    return preres+xmax.reshape(preres.shape)

def softplus(x, bias=0):
    return np.log(1+np.exp(x-bias))

def softmax(x, bias=0):
    return np.exp(x*(1+bias))/np.exp(x*(1+bias)).sum()

def train_mixture_density_network(X_tr, Y_tr, X_ts, Y_ts, n_components=10,
                                  save_fname='net_params/net'):
    """ Train a mixture density network. 

    The idea is described in detail and with great clarity in [Bishop 1995].
    
    Args:
      X_tr - numpy array of shape (n_train_data, n_dim_in)
      Y_tr - numpy array of shape (n_train_data, n_dim_in)
      X_ts - numpy array of shape (n_valid_data, n_dim_out)
      Y_ts - numpy array of shape (n_valid_data, n_dim_out)
      n_components - number of components in the Gaussian mixtures
      save_fname - the network weights go here.

    Returns:
      None - you'll need to load the network from file using joblib.
    """
    nonlin = lasagne.nonlinearities.rectify
    dropout = lasagne.layers.dropout

    net = dropout(lasagne.layers.InputLayer(shape=(None, 1)), p=0)
    net = dropout(lasagne.layers.DenseLayer(net, num_units=64, nonlinearity=nonlin), p=0)
    net = dropout(lasagne.layers.DenseLayer(net, num_units=32, nonlinearity=nonlin), p=0)
    net = lasagne.layers.DenseLayer(net, num_units=3*n_components,
        nonlinearity=lasagne.nonlinearities.identity)

    input_var = lasagne.layers.get_all_layers(net)[0].input_var
    target_var = T.matrix('target_output')

    # The cost function, log likelihood of the data under the mixture model.
    tr_out = lasagne.layers.get_output(net, deterministic=False)
    tr_alphas = T.nnet.softmax(tr_out[:,:n_components])
    tr_means = tr_out[:,n_components:n_components*2]
    tr_sigmas = 1e-7+T.nnet.softplus(tr_out[:,n_components*2:n_components*3])
    tr_Eq = T.log(tr_alphas/(T.sqrt(2*np.pi)*tr_sigmas))-\
            (T.tile(target_var[:,0:1], (1, n_components))-tr_means)**2/(2*tr_sigmas**2)
    tr_cost = -logsumexp(tr_Eq, axis=1).sum()
    l2_penalty = lasagne.regularization.regularize_network_params(
        net, lasagne.regularization.l2)*.0001
    tr_cost = tr_cost + l2_penalty

    val_out = lasagne.layers.get_output(net, deterministic=True)
    val_alphas = T.nnet.softmax(val_out[:,:n_components])
    val_means = val_out[:,n_components:n_components*2]
    val_sigmas = 1e-7+T.nnet.softplus(val_out[:,n_components*2:n_components*3])
    val_Eq = T.log(val_alphas/(T.sqrt(2*np.pi)*val_sigmas))-\
            (T.tile(target_var[:,0:1], (1, n_components))-val_means)**2/(2*val_sigmas**2)
    val_cost = -logsumexp(val_Eq, axis=1).sum()

    learning_rate = T.scalar(name='learning_rate')
    params = lasagne.layers.get_all_params(net, trainable=True)
    updates = lasagne.updates.adam(tr_cost, params, learning_rate)
    train = theano.function([input_var, target_var, learning_rate],
        tr_cost, updates=updates, allow_input_downcast=True)
    validate = theano.function([input_var, target_var],
        val_cost, allow_input_downcast=True)

    # Train the neural net.
    batches_per_epoch = int(np.ceil(X_tr.shape[0]/float(BATCH_SIZE)))
    best_loss = np.inf
    val_loss = validate(X_ts, Y_ts)
    print('Validation loss before training: {}.'.format(val_loss))
    print('Starting training...')
    for epoch_id in range(1000):
        try:
            tr_loss = 0
            for batch_id in range(batches_per_epoch):
                tr_loss += train(X_tr[batch_id*BATCH_SIZE:(batch_id+1)*BATCH_SIZE],
                                 Y_tr[batch_id*BATCH_SIZE:(batch_id+1)*BATCH_SIZE],
                                 1e-3)
            val_loss = validate(X_ts, Y_ts)

            if np.isfinite(val_loss) and val_loss < best_loss:
                joblib.dump(net, save_fname)
                best_loss = val_loss

            sys.stdout.write('\rEpoch {}. Valid loss {:.4g} [{:.4g}]. Train loss {:.4g}.'.format(
                    epoch_id, float(val_loss), float(best_loss), tr_loss/float(batches_per_epoch)))
            sys.stdout.flush()
        except KeyboardInterrupt:
            print('Training interrupted.')
            break

def eval_gaussian_mixture(params, bias=0, density_grid=np.linspace(-10, 10, 1000)):
    """ Evaluate a Gaussian mixture on a 1d grid. 

    Args:
      params - parameters of the mixture, e.g. as returned by a mixture
               density network evaluated on an input point.
      bias - positive float, the higher the more bias towards the heaviest 
             component in our sampler.
      density_grid - where to evaluate the density.
    
    Returns:
      density - mixture density on the density_grid.
    """
    density_grid = np.atleast_2d(density_grid).T
    n_components = params.size/3
    alphas = softmax(params.flatten()[:n_components], bias)
    means = params.flatten()[n_components:n_components*2]
    sigmas = 1e-5+softplus(params.flatten()[n_components*2:n_components*3], bias)
    sigmas[np.where(sigmas>10)[0]]=10.

    component_wise = alphas * np.exp(-.5 * (density_grid - means)**2 / sigmas**2) / (sigmas * np.sqrt(2 * np.pi))
    return component_wise.sum(axis=1)
