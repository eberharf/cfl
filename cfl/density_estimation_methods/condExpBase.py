import os
# TODO: add GPU support
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from cfl.density_estimation_methods.cde import CDE #base class

# example_params = {'batch_size': 128, 'lr': 1e-3, 
#       'optimizer': tf.keras.optimizers.Adam(lr=1e-3), 'n_epochs': 100, 
#       'test_every': 10, 'save_every': 10}

class CondExpBase(CDE):

    def __init__(self, data_info, model_params):
        ''' Initialize model and define network.
            Arguments:
                data_info : a dictionary containing information about the data 
                    that will be passed in. Should contain 'X_dims' and 'Y_dims' as keys
                model_params : dictionary containing parameters for the model
        '''

        # set attributes
        self.data_info = data_info #TODO: check that data_info is correct format
        self.model_params = model_params
        self.trained = False # keep track of training status

        #TODO: need to pass in the optimizer as a string, and then create the object - passing in the object is annoying
        self.model = self.build_model()
        
        # load model weights if specified
        if 'weights_path' in self.model_params.keys():
            self.load_parameters(self.model_params['weights_path'])
    

    def train(self, Xtr, Ytr, Xts, Yts, saver=None):
        ''' Full training loop. Constructs t.data.Dataset for training and testing,
            updates model weights each epoch and evaluates on test set periodically.
            Saves model weights as checkpoints.
            Arguments:
                Xtr : X training set of dimensions [# training observations, # features] (np.array)
                Ytr : Y training set of dimensions [# training observations, # features] (np.array)
                Xts : X test set of dimensions [# test observations, # features] (np.array)
                Yts : Y test set of dimensions [# test observations, # features] (np.array)
                saver : Saver to pull save paths from (Saver object)
            Returns: None
        '''
        #TODO: do a more formalized checking that actual dimensions match expected 
        #TODO: say what expected vs actual are 
        #TODO: I got confused that it was Xtr Ytr Xts Yts, can there be an 
        #      option where you put in just X, Y and it splits for you? I liked that more
        assert self.data_info['X_dims'][1] == Xtr.shape[1] == Xts.shape[1], "Expected X-dim do not match actual X-dim"
        if 'loss' not in self.model_params.keys():
            print('No loss function specified in model_params, defaulting to mean_squared_error.')
            self.model_params['loss'] = 'mean_squared_error' # TODO: we should resave parameters if we update them

        self.model.compile(
            loss=self.model_params['loss'],
            optimizer=self.model_params['optimizer'],
        )

        if saver is not None:
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=saver.get_save_path(
                    'checkpoints/weights_epoch_{epoch:02d}_val_loss_{val_loss:.2f}'),
                save_weights_only=True,
                monitor='val_loss',
                mode='min',
                save_best_only=True)
            callbacks = [model_checkpoint_callback]
        else:
            callbacks = []

        history = self.model.fit(
            Xtr, Ytr,
            batch_size=self.model_params['batch_size'],
            epochs=self.model_params['n_epochs'],
            validation_data=(Xts,Yts),
            callbacks=callbacks
        )


        train_loss = history.history['loss']
        val_loss = history.history['val_loss']

        if saver is not None:
            self.graph_results(train_loss, val_loss, save_path=saver.get_save_path('train_val_loss'))
            np.save(saver.get_save_path('train_loss'), train_loss)
            np.save(saver.get_save_path('val_loss'), val_loss)
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


    def predict(self, X, Y=None, saver=None): #put in the x and y you want to predict with
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
        pyx = self.model.predict(X)
        if saver is not None:
            np.save(saver.get_save_path('pyx'), pyx)
        return pyx

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


