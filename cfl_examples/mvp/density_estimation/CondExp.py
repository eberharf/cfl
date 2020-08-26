'''
Iman Wahle
Created August 14, 2020
A class for performing estimating E[P(Y | X)] from X and Y samples.
'''

import os
# TODO: add GPU support
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class CondExp():
    ''' A class for training/predicting for a neural network that learns
        P(Y | X) given two coordinated datasets X and Y. 
    '''

    def __init__(self, n_xfeatures, n_yfeatures, verbose=False):
        ''' Initialize model and define network. 
            Arguments: 
                n_xfeatures : dimensionality of X dataset (int)
                n_yfeatures : dimensionality of Y dataset (int)
                verbose : whether to print out model information (boolean)
            Returns: None
        '''
        self.n_xfeatures = n_xfeatures
        self.n_yfeatures = n_yfeatures
        self.model = self.build_model()
        self.verbose = verbose

    def compute_loss(self, x_true, y_true, training=False):
        ''' Compute the mean squared error (MSE) between ground truth and prediction.
            Arguments:
                x_true : a batch of observations of X (tf.Tensor)
                y_true : a batch of observations of Y (tf.Tensor)
                training : whether to backpropagate gradient (boolean)
            Returns: the average MSE for this batch (float)
        '''
        
        y_hat = self.model(x_true, training=training)
        cost = tf.keras.losses.MSE(y_true, y_hat)
        return tf.reduce_mean(cost)    

    @tf.function
    def train_step(self, optimizer, train_x, train_y):
        ''' train model with loss function defined in compute_loss and passed-in optimizer.
            Arguments:
                optimizer : the optimizer for the model (tf.keras.optimizers object)
                train_x : a batch of observations of X (tf.Tensor)
                train_y : a batch of observations of Y (tf.Tensor)
            Returns: the loss for this batch (float)
        '''

        # GradientTape: Trace operations to compute gradients
        with tf.GradientTape() as tape:
            # calculate loss
            loss = self.compute_loss(train_x, train_y, training=True)
        # compute and apply gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def build_model(self):
        ''' Define the neural network based on dimensions passed in during initialization.
            Eventually, this architecture will have to become more dynamic (TODO).
            Arguments: None
            Returns: the model (tf.keras.models.Model object)
        '''

        # Network
        input_layer = tf.keras.Input(shape=(self.n_xfeatures,), 
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
                                    units=self.n_yfeatures, 
                                    activation='linear',
                                    kernel_initializer='he_normal',
                                    activity_regularizer=tf.keras.regularizers.l2(0.0001),
                                    name='nn_output_layer')(layer)
        model = tf.keras.models.Model(input_layer, output_layer)
        return model


    def train_model(self, X_tr, Y_tr, X_ts, Y_ts, n_epochs, save_fname='net_params/net'):
        ''' Full training loop. Constructs t.data.Dataset for training and testing, 
            updates model weights each epoch and evaluates on test set periodically.
            Saves model weights as checkpoints. 
            Arguments:
                X_tr : X training set of dimensions [# training observations, # features] (np.array)
                Y_tr : Y training set of dimensions [# training observations, # features] (np.array)
                X_ts : X test set of dimensions [# test observations, # features] (np.array)
                Y_ts : Y test set of dimensions [# test observations, # features] (np.array)
                n_epochs : number of epochs to train model for (int)
                save_fname : file path to save checkpoints to (string)
            Returns: None
        '''
        # TODO: make validation set optional (is this really helpful?)
        # TODO: standardize save path structure
        # TODO: save each checkpoint to different name

        # Setup
        self.n_xfeatures = X_tr.shape[1]
        self.n_yfeatures = Y_tr.shape[1]
        batch_size = 128
        lr = 1e-3
        optimizer = tf.keras.optimizers.Adam(lr=lr)
        if self.verbose:
            self.model.summary()
        
        # Construct train and test datasets (load, shuffle, set batch size)
        dataset_tr = tf.data.Dataset.from_tensor_slices((X_tr, Y_tr)).shuffle(X_tr.shape[0]).batch(batch_size)
        dataset_ts = tf.data.Dataset.from_tensor_slices((X_ts, Y_ts)).shuffle(X_ts.shape[0]).batch(batch_size)
        
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
                train_loss(self.train_step(optimizer, train_x, train_y))
            train_losses.append(train_loss.result())
            
            # test
            if i % test_every == 0:
                test_loss = tf.keras.metrics.Mean()
                for test_x, test_y in dataset_ts:
                    test_loss(self.compute_loss(test_x, test_y, training=False))
                test_losses.append(test_loss.result())
                                    
                print('Epoch {}/{}: train_loss: {}, test_loss: {}'.format(
                    i, n_epochs, train_losses[-1], test_losses[-1])) 
            
            if i % save_every == 0:
                print("Saving weights to ", save_fname.format(i))
                self.model.save_weights(save_fname.format(i))
                
        if self.verbose:
            print(len(train_losses))
            print(len(test_losses))
            plt.plot(range(len(train_losses)), train_losses)
            plt.plot(np.linspace(0,len(train_losses),len(test_losses)).astype(int), test_losses)
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training and Test Loss')
            plt.legend(['Train', 'Test'])
            plt.show()

        return None                        

    
    def predict(self, X):
        ''' Given a set of observations X, get neural network output.
            Arguments:
                X : model input of dimensions [# observations, # features] (np.array)
            Returns: model prediction (np.array) (TODO: check if this is really np.array)
        '''
        return self.model.predict(X)

    def score(self, X, Y): 
        # TODO: implementation
        pass

    def get_model(self):
        ''' Return model object.
            Arguments: None
            Returns: network model (tf.keras.models.Model object)
        '''
        return self.model

    def load_weights(self, weights_path):
        ''' Load model weights from saved checkpoint into current model.
            Arguments: 
                weights_path : path to checkpoint (string)
            Returns: None
        '''
        self.model.load_weights(weights_path)
        return None