''' Iman Wahle, 2020 '''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def calc_pdf(y, mu, var):
    """Calculate component density"""
    value = tf.subtract(y, mu)**2
    value = (1/tf.math.sqrt(2 * np.pi * var)) * tf.math.exp((-1/(2*var)) * value)
    return value

# def logsumexp(x, axis=None, keepdims=True):
#     xmax = tf.max(x, axis=axis, keepdims=True)
#     preres = tf.log(tf.sum(tf.exp(x-xmax), axis=axis, keepdims=keepdims))
#     return preres+xmax.reshape(preres.shape)

def mdn_loss(model, x_true, y_true, training):
    """MDN Loss Function """
    alpha, mu, var = model(x_true, training=training)
    out = calc_pdf(y_true, mu, var)
    # multiply with each alpha and sum it
    out = tf.multiply(out, alpha)
    out = tf.reduce_sum(out, 1, keepdims=True)
    out = -tf.math.log(out + 1e-10)
    return tf.reduce_mean(out)    


@tf.function
def train_step(model, optimizer, train_x, train_y):
    # GradientTape: Trace operations to compute gradients
    with tf.GradientTape() as tape:
        # calculate loss
        loss = mdn_loss(model, train_x, train_y, training=True)
    # compute and apply gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

    
def get_model(n_features, n_components):
    # Network
    input_layer = tf.keras.Input(shape=(n_features,), name='input_layer')
    layer = tf.keras.layers.Dense(64, activation='relu', name='nn_layer1',
                                 activity_regularizer=tf.keras.regularizers.l2(0.0001))(input_layer)
    layer = tf.keras.layers.Dense(32, activation='relu', name='nn_layer2',
                                 activity_regularizer=tf.keras.regularizers.l2(0.0001))(layer)
    mu = tf.keras.layers.Dense((n_features * n_components), activation=None, name='mu_layer',
                              activity_regularizer=tf.keras.regularizers.l2(0.0001))(layer)
    # variance (should be greater than 0 so we exponentiate it)
    var_layer = tf.keras.layers.Dense(n_components, activation=None, name='dense_var_layer',
                                     activity_regularizer=tf.keras.regularizers.l2(0.0001))(layer)
    var = tf.keras.layers.Lambda(lambda x: tf.math.exp(x), output_shape=(n_components,), name='var_layer',
                                activity_regularizer=tf.keras.regularizers.l2(0.0001))(var_layer)
    # mixing coefficient should sum to 1.0
    alpha = tf.keras.layers.Dense(n_components, activation='softmax', name='alpha_layer',
                                 activity_regularizer=tf.keras.regularizers.l2(0.0001))(layer)
    
    model = tf.keras.models.Model(input_layer, [alpha, mu, var])
    return model


def train_mixture_density_network(X_tr, Y_tr, X_ts, Y_ts, n_components=10, n_epochs=1000,
                                  lr=1e-3, verbose=False, save_fname='net_params/net'):
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
    # Setup
    n_examples_tr = X_tr.shape[0]
    n_examples_ts = X_ts.shape[0]
    n_features = X_tr.shape[1]
    batch_size = 32
    
    model = get_model(n_features, n_components)
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    if verbose:
        model.summary()
    
                     
    # Construct train and test datasets (load, shuffle, set batch size)
    dataset_tr = tf.data.Dataset.from_tensor_slices((X_tr, Y_tr)).shuffle(n_examples_tr).batch(batch_size)
    dataset_ts = tf.data.Dataset.from_tensor_slices((X_ts, Y_ts)).shuffle(n_examples_ts).batch(batch_size)
    
    
    train_losses = []
    test_losses = []
    test_every = int(0.1 * n_epochs)
    save_every = int(0.1 * n_epochs)


    # Start training
    print('Test every {} epochs'.format(test_every))
    tl_idx = 0
    for i in range(n_epochs):

        # train
        train_loss = tf.keras.metrics.Mean()
        for train_x, train_y in dataset_tr:
            train_loss(train_step(model, optimizer, train_x, train_y))
        train_losses.append(train_loss.result())
        
        # test
        if i % test_every == 0:
            test_loss = tf.keras.metrics.Mean()
            for test_x, test_y in dataset_ts:
                test_loss(mdn_loss(model, test_x, test_y, training=False))
            test_losses.append(test_loss.result())
            tl_idx+=1
                                 
            print('Epoch {}/{}: train_loss: {}, test_loss: {}'.format(
                i, n_epochs, train_losses[-1], test_losses[-1])) 
        
        if i % save_every == 0:
            print("Saving weights to ", save_fname.format(i))
            model.save_weights(save_fname.format(i))
            
    if verbose:
        print(len(train_losses))
        print(len(test_losses))
        plt.plot(range(len(train_losses)), train_losses)
        plt.plot(np.linspace(0,len(train_losses),len(test_losses)).astype(int), test_losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss')
        plt.legend(['Train', 'Test'])
        plt.show()
        
    return model, dataset_tr, dataset_ts
                                 
                               
def softplus(x, bias=0):
    return tf.math.log(1+tf.math.exp(x-bias))

def softmax(x, bias=0):
    return tf.reduce_sum(tf.math.exp(x*(1+bias))/tf.math.exp(x*(1+bias)))

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
    n_components = params[0].shape
    alpha,mean,var = params
    component_wise = alpha * np.exp(-.5 * (density_grid - mean)**2 / var) / (np.sqrt(2 * np.pi * var))
    return component_wise.sum(axis=1)                                 
                                 
                                 
                                 
                                 
                                 
                                 
                                 
                                 
                                 
                                 
                                 
                                 
                                 
                                 
                                 
                                 
                                 
                                 
    
    