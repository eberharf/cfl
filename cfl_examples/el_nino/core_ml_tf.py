''' Iman Wahle, 2020 '''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# def calc_pdf(y, mu, var):
#     """Calculate component density"""
#     value = tf.subtract(y, mu)**2
#     value = (1/tf.math.sqrt(2 * np.pi * var)) * tf.math.exp((-1/(2*var)) * value)
#     return value

# def logsumexp(x, axis=None, keepdims=True):
#     xmax = tf.max(x, axis=axis, keepdims=True)
#     preres = tf.log(tf.sum(tf.exp(x-xmax), axis=axis, keepdims=keepdims))
#     return preres+xmax.reshape(preres.shape)

def compute_loss(model, x_true, y_true, training=False):
    """MDN Loss Function """
    y_hat = model(x_true, training=training)
    cost = tf.keras.losses.MSE(y_true, y_hat)
    return tf.reduce_mean(cost)    


@tf.function
def train_step(model, optimizer, train_x, train_y):
    # GradientTape: Trace operations to compute gradients
    with tf.GradientTape() as tape:
        # calculate loss
        loss = compute_loss(model, train_x, train_y, training=True)
    # compute and apply gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

    
def get_model(n_in, n_out):
    # Network
    input_layer = tf.keras.Input(shape=(n_in,), 
                                 name='nn_input_layer')
    layer = tf.keras.layers.Dropout(
                                 rate=0.2, 
                                 activity_regularizer=tf.keras.regularizers.l2(0.0001),
                                 name='nn_dropout1')(input_layer)
    layer = tf.keras.layers.Dense(
                                 units=1024, 
                                 activation='linear',
                                 kernel_initializer='he_normal',
                                 activity_regularizer=tf.keras.regularizers.l2(0.0001),
                                 name='nn_dense1')(layer)
    layer = tf.keras.layers.Dropout(
                                 rate=0.5, 
                                 activity_regularizer=tf.keras.regularizers.l2(0.0001),
                                 name='nn_dropout2')(layer)
    layer = tf.keras.layers.Dense(
                                 units=1024, 
                                 activation='linear',
                                 kernel_initializer='he_normal',
                                 activity_regularizer=tf.keras.regularizers.l2(0.0001),
                                 name='nn_layer2')(layer)
    layer = tf.keras.layers.Dropout(
                                 rate=0.5, 
                                 activity_regularizer=tf.keras.regularizers.l2(0.0001),
                                 name='nn_dropout3')(layer)
    output_layer = tf.keras.layers.Dense(
                                 units=n_out, 
                                 activation='linear',
                                 kernel_initializer='he_normal',
                                 activity_regularizer=tf.keras.regularizers.l2(0.0001),
                                 name='nn_output_layer')(layer)
    model = tf.keras.models.Model(input_layer, output_layer)
    return model


def train_network(X_tr, Y_tr, X_ts, Y_ts, n_epochs=1000,
                  lr=1e-3, verbose=False, save_fname='net_params/net'):
    
    # Setup
    n_examples_tr = X_tr.shape[0]
    n_examples_ts = X_ts.shape[0]
    n_in = X_tr.shape[1]
    n_out = Y_tr.shape[1]
    batch_size = 128
    
    model = get_model(n_in, n_out)
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
                test_loss(compute_loss(model, test_x, test_y, training=False))
            test_losses.append(test_loss.result())
                                 
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
                       

    