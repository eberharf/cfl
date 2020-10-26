import os
# TODO: add GPU support
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import random as rn
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from cfl.util.data_processing import standardize_train_test

from cfl.density_estimation_methods.cde import CDE #base class

# example_params = {'batch_size': 128, 'lr': 1e-3, 
#       'optimizer': tf.keras.optimizers.Adam(lr=1e-3), 'n_epochs': 100, 
#       'test_every': 10, 'save_every': 10}

class CondExpBase(CDE):

    def __init__(self, data_info, model_params, random_state=None):
        ''' Initialize model and define network.
            Arguments:
                data_info : a dictionary containing information about the data 
                    that will be passed in. Should contain 'X_dims' and 'Y_dims' as keys
                model_params : dictionary containing parameters for the model
                random_state (int): Used to set a random seed to create reproducible results
        '''
        #globally set the random seed to a reproducible value before creating/training the model 
        self._set_random_state(random_state)

        # set attributes
        self.data_info = data_info #TODO: check that data_info is correct format
        self.model_params = model_params
        self.check_save_model_params()
        self.trained = False # keep track of training status

        self.model = self.build_model()
        
        # load model weights if specified
        if self.model_params['weights_path'] is not None:
            self.load_parameters(self.model_params['weights_path'])
    

    def train(self, dataset, standardize):
        ''' Full training loop. Constructs t.data.Dataset for training and testing,
            updates model weights each epoch and evaluates on test set periodically.
            Saves model weights as checkpoints.
            Arguments:
                Xtr : X training set of dimensions [# training observations, # features] (np.array)
                Ytr : Y training set of dimensions [# training observations, # features] (np.array)
                Xts : X test set of dimensions [# test observations, # features] (np.array)
                Yts : Y test set of dimensions [# test observations, # features] (np.array)
            Returns: None
        '''
        #TODO: do a more formalized checking that actual dimensions match expected 
        #TODO: say what expected vs actual are 


        # train-test split
        split_data = train_test_split(dataset.X, dataset.Y, shuffle=True, train_size=0.75)
        
        # standardize if specified
        if standardize:
            split_data = standardize_train_test(split_data)
        Xtr, Xts, Ytr, Yts = split_data


        # build optimizer
        optimizer = tf.keras.optimizers.get({ 'class_name' : self.model_params['optimizer'],
                                              'config' : self.model_params['opt_config']})

        # compile model
        self.model.compile(
            loss=self.model_params['loss'],
            optimizer=optimizer,
        )

        # specify checkpoint save callback
        if dataset.to_save:
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=dataset.saver.get_save_path(
                    'checkpoints/weights_epoch_{epoch:02d}_val_loss_{val_loss:.2f}'),
                save_weights_only=True,
                monitor='val_loss',
                mode='min',
                save_best_only=True)
            callbacks = [model_checkpoint_callback]
        else:
            callbacks = []

        # train model
        history = self.model.fit(
            Xtr, Ytr,
            batch_size=self.model_params['batch_size'],
            epochs=self.model_params['n_epochs'],
            validation_data=(Xts,Yts),
            callbacks=callbacks
        )


        # handle results
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']

        if dataset.to_save:
            self.graph_results(train_loss, val_loss, save_path=dataset.saver.get_save_path('train_val_loss'))
            np.save(dataset.saver.get_save_path('train_loss'), train_loss)
            np.save(dataset.saver.get_save_path('val_loss'), val_loss)
        else:
            self.graph_results(train_loss, val_loss, save_path=None)

        self.trained = True
        return train_loss, val_loss


    def graph_results(self, train_loss, val_loss, save_path):
        '''graphs the training vs testing loss across all epochs of training'''
        plt.plot(range(len(train_loss)), train_loss, label='train_loss')
        plt.plot(np.linspace(0,len(train_loss),len(val_loss)).astype(int), val_loss, label='val_loss')
        plt.xlabel('Epochs')
        plt.ylabel(self.model_params['loss'])
        plt.title('Training and Test Loss')
        plt.legend(loc='upper right')
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()


    def predict(self, dataset): #put in the x and y you want to predict with
        # TODO: deal with Y=None weirdness
        ''' Given a set of observations X, get neural network output.
            Arguments:
                X : model input of dimensions [# observations, # x_features] (np.array)
                Y : model input of dimensions [# observations, # y_features] (np.array)
                    note: this derivation of CDE doesn't require Y for prediction.
            Returns: model prediction (np.array) (TODO: check if this is really np.array or tf.Tensor)
        '''
        # if Y is not None:
        #     raise RuntimeWarning("Y was passed as an argument, but is not being used for prediction.")

        assert self.trained, "Remember to train the model before prediction."
        dataset.pyx = self.model.predict(dataset.X)
        if dataset.to_save:
            np.save(dataset.saver.get_save_path('pyx'), dataset.pyx)
        return dataset.pyx

    def evaluate(self, X, Y):
        ''' Compute the mean squared error (MSE) between ground truth and prediction.
            Arguments:
                X : a batch of true observations of X (tf.Tensor)
                Y : a batch of true observations of Y, corresponding to X (tf.Tensor)
                training : whether to backpropagate gradient (boolean)
            Returns: the average MSE for this batch (float)
        '''
        
        assert self.trained, "Remember to train the model before evaluation."

        Y_hat = self.predict(X)
        loss_fxn = tf.keras.losses.get(self.model_params['loss'])
        cost = loss_fxn(Y, Y_hat) 
        return tf.reduce_mean(cost)


    def load_parameters(self, file_path):
        ''' Load model weights from saved checkpoint into current model.
            Arguments:
                file_path : path to checkpoint file (string)
            Returns: None
        '''

        assert hasattr(self, 'model'), 'Build model before loading parameters.'

        print("Loading parameters from ", file_path)
        self.model.load_weights(file_path)
        self.trained = True
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
        reg = tf.keras.regularizers.l2(0.0001)
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(self.data_info['X_dims'][1],)),
            tf.keras.layers.Dropout(rate=0.2, activity_regularizer=reg), 
            tf.keras.layers.Dense(units=50, activation='linear', 
                kernel_initializer='he_normal', activity_regularizer=reg), 
            tf.keras.layers.Dropout(rate=0.5, activity_regularizer=reg),
            tf.keras.layers.Dense(units=10, activation='linear', 
                kernel_initializer='he_normal', activity_regularizer=reg), 
            tf.keras.layers.Dropout(rate=0.5, activity_regularizer=reg),
            tf.keras.layers.Dense(units=self.data_info['Y_dims'][1], activation='linear', 
                kernel_initializer='he_normal', activity_regularizer=reg), 
        ])

        return model


    def check_save_model_params(self):
        default_params = {  'batch_size'  : 32, 
                            'n_epochs'    : 20,
                            'optimizer'   : 'adam',
                            'opt_config'  : {},
                            'verbose'     : True,
                            'dense_units' : [50, self.data_info['Y_dims'][1]],
                            'activations' : ['relu', 'linear'],
                            'dropouts'    : [0, 0],
                            'weights_path': None,
                            'loss'        : 'mean_squared_error'
                        }
        
        for k in default_params.keys():
            if k not in self.model_params.keys():
                print('{} not specified in model_params, defaulting to {}'.format(k, default_params[k]))
                self.model_params[k] = default_params[k]

        # if dataset.to_save:
        #     dataset.saver.save_parameters(self.model_params, 'CDE_params')
        
    def _set_random_state(self, random_state):
        '''
        sets a random seed in order to generate reproducible results. 

        sets a global random seed at the numpy, python, and tensorflow level, 
        as per the instructions for generating reproducible keras results here: 
        https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development 
        ''' 
        
        np.random.seed(random_state)
        rn.seed(random_state)
        tf.random.set_seed(random_state)
