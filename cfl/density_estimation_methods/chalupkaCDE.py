import os
# TODO: add GPU support
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from density_estimation_methods.cde import CDE #base class

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt



example_params = {'batch_size': 128, 'lr': 1e-3, 'optimizer': tf.keras.optimizers.Adam(lr=1e-3), 'n_epochs': 100, 'test_every': 10, 'save_every': 10}

class ChalupkaCDE(CDE):

    def __init__(self, data_info, model_params, verbose):
        ''' Initialize model and define network.
            Arguments:
                data_info : a dictionary containing information about the data that will be passed in
                model_params : dictionary containing parameters for the model
                verbose : whether to print out model information (boolean)
        '''

        self.data_info = data_info #TODO: check that data_info is correct format

        self.model_params = model_params
        #TODO: need to pass in the optimizer as a string, and then create the object - passing in the object is annoying
        self.verbose = verbose
        self.model = self.build_model()

    # def train(self, Xtr, Ytr, Xts, Yts, save_dir):
    def train(self, Xtr, Ytr, Xts, Yts, save_path): # TODO: figure out saving conventions later
        ''' Full training loop. Constructs t.data.Dataset for training and testing,
            updates model weights each epoch and evaluates on test set periodically.
            Saves model weights as checkpoints.
            Arguments:
                Xtr : X training set of dimensions [# training observations, # features] (np.array)
                Ytr : Y training set of dimensions [# training observations, # features] (np.array)
                Xts : X test set of dimensions [# test observations, # features] (np.array)
                Yts : Y test set of dimensions [# test observations, # features] (np.array)
                save_dir : directory path to save checkpoints to (string)
            Returns: None
        '''
        #TODO: do a more formalized checking that actual dimensions match expected 
        assert self.data_info['X_dims'][1] == Xtr.shape[1] == Xts.shape[1], "Expected X-dim do not match actual X-dim"

        self.model.compile(optimizer=self.model_params['optimizer'],
                    loss=tf.keras.losses.MeanSquaredError(),
                    metrics=['accuracy'])

        history = self.model.fit(Xtr, Ytr, batch_size=self.model_params['batch_size'], epochs=self.model_params['n_epochs'], 
                            validation_data=(Xts, Yts))

        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        # plt.ylim([0.5, 1])
        plt.legend(loc='lower right')
        plt.show()

        test_loss, test_acc = self.model.evaluate(Xts,  Yts, verbose=2)
        return history.history['loss'], history.history['val_loss']


    def graph_results(self, train_losses, test_losses, save_path=None):
        '''graphs the training vs testing loss across all epochs of training'''
        plt.plot(range(len(train_losses)), train_losses)
        plt.plot(np.linspace(0,len(train_losses),len(test_losses)).astype(int), test_losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss')
        plt.legend(['Train', 'Test'])
        # plt.savefig(save_path)
        # plt.close()
        plt.show()

    def predict(self, X, Y=None): #put in the x and y you want to predict with
        ''' Given a set of observations X, get neural network output.
            Arguments:
                X : model input of dimensions [# observations, # x_features] (np.array)
                Y : model input of dimensions [# observations, # y_features] (np.array)
                    note: this derivation of CDE doesn't require Y for prediction.
            Returns: model prediction (np.array) (TODO: check if this is really np.array or tf.Tensor)
        # '''
        # if Y is not None:
        #     raise RuntimeWarning("Y was passed as an argument, but is not being used for prediction.")
        return self.model.predict(X)


    def evaluate(self, X, Y, training=False):
        ''' Compute the mean squared error (MSE) between ground truth and prediction.
            Arguments:
                X : a batch of true observations of X (tf.Tensor)
                Y : a batch of true observations of Y, corresponding to X (tf.Tensor)
                training : whether to backpropagate gradient (boolean)
            Returns: the average MSE for this batch (float)
        '''

        Y_hat = self.model(X, training=training)
        cost = tf.keras.losses.MSE(Y, Y_hat)
        return tf.reduce_mean(cost)


    def load_parameters(self, file_path):
        ''' Load model weights from saved checkpoint into current model.
            Arguments:
                file_path : path to checkpoint file (string)
            Returns: None
        '''
        print("Loading parameters from ", file_path)
        self.model.load_weights(file_path)
        return None


    def save_parameters(self, file_path):
        print("Saving parameters to ", file_path)
        self.model.save_weights(file_path)

    def build_model(self):
        ''' Define the neural network based on dimensions passed in during initialization.
            Eventually, this architecture will have to become more dynamic (TODO).

            Right now the architecture is optimized for visual bars 1000 10x10 images 
            Arguments: None
            Returns: the model (tf.keras.models.Model object)
        '''

        model = models.Sequential()
        model.add(layers.Dense(50, activation='relu', input_shape=(100,)))
        model.add(layers.Dense(50, activation='relu')) # TODO: currently not matching initialization rule but i don't think this will cause issues at our level of comparison
        model.add(layers.Dense(1, activation='sigmoid'))

        return model


