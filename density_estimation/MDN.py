'''
Iman Wahle
Created August 14, 2020
A class for performing density estimation with MDN
'''

import os
# TODO: add GPU support
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class MDN():
    
    def __init__(self, n_xfeatures, n_yfeatures, verbose=False):
        self.n_xfeatures = n_xfeatures
        self.n_yfeatures = n_yfeatures
        self.model = self.build_model()
        self.verbose = verbose

    def compute_loss(self, x_true, y_true, training=False):
        """MDN Loss Function """
        # TODO: documentation
        y_hat = self.model(x_true, training=training)
        cost = tf.keras.losses.MSE(y_true, y_hat)
        return tf.reduce_mean(cost)    

    @tf.function
    def train_step(self, optimizer, train_x, train_y):
        # TODO: documentation

        # GradientTape: Trace operations to compute gradients
        with tf.GradientTape() as tape:
            # calculate loss
            loss = self.compute_loss(train_x, train_y, training=True)
        # compute and apply gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def build_model(self):
        # TODO: documentation

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


    def train_model(self, X_tr, Y_tr, X_ts, Y_ts, n_epochs=1000, save_fname='net_params/net'):
        # TODO: documentation
        # TODO: make validation set optional (is this really helpful?)
        # TODO: standardize save path structure

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

    
    def predict(self, X, Y):
        # TODO: implementation
        pass

    def score(self, X, Y): 
        # TODO: implementation
        pass

    def get_model(self):
        return self.model
    
    def get_n_xfeatures(self):
        return self.n_xfeatures

    def get_n_yfeatures(self):
        return self.n_yfeatures