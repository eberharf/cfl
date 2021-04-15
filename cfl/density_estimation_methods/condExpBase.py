from abc import abstractmethod
import os
import shutil
# TODO: add GPU support
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from cfl.density_estimation_methods.cde_interface import Block #base class

# Things that descend from this class should have a self.name attribute but 
# this class doesn't since CondExpBase objects are not supposed to be created 
# by the user 


class CondExpBase(Block):
    # TODO: update Class docstring
    ''' A class to define, train, and perform inference with conditional density
    estimators that fall under the "conditional expectation" umbrella. This
    subset of conditional density estimators (referred to as 'CondExp') learns
    E[P(Y|X)] instead of the full conditional distribution. This base class
    implements all functions needed for training and predictiion, and supplies
    a model architecture that can be overridden by children of this class. In
    general, if you would like to use a CondExp CDE for your CFL pipeline, it is
    easiest to either 1) inherit this class and override the build_model
    function, which defines the architecture, or 2) use the condExpMod child
    class which allows you to pass in limited architecture specifications
    through the params attribute.

    Attributes:
        name : name of the model so that the model type can be recovered from
               saved parameters (str)
        data_info : dict with information about the dataset shape (dict)
        default_params : default parameters to fill in if user doesn't provide
                         a given entry (dict)
        params : parameters for the CDE that are passed in by the user and
                 corrected by check_save_model_params (dict)
        trained : whether or not the modeled has been trained yet. This can
                  either happen by defining by instantiating the class and
                  calling train, or by passing in a path to saved weights from
                  a previous training session through params['weights_path'].
                  (bool)
        weights_loaded : whether or not weights were loaded from
                         params['weights_path]. (bool)
        model : tensorflow model for this CDE (tf.keras.Model.Sequential)


    Methods:
        train : train the neural network on a given Dataset
        graph_results : helper function to graph training and validation loss
        predict : once the model is trained, predict for a given Dataset
        evaluate : return the model's prediction loss on a Dataset
        load_model : load tensorflow model weights from a file into
                          self.model
        save_model : save the current weights of self.model
        build_model : create and return a tensorflow model
        _check_model_params : fill in any parameters that weren't provided in
                             params with the default value, and discard any
                             unnecessary paramaters that were provided.
    '''

    def __init__(self, data_info, params):
        ''' 
        Initialize model and define network.

        Arguments:
            data_info (dict) : a dictionary containing information about the 
                data that will be passed in. Should contain 'X_dims',
                'Y_dims', and 'Y_type' as keys.
            params (dict) : dictionary containing parameters for the model.
            model (str) : name of the model so that the model type can be
                recovered from saved parameters.
        Returns: 
            None
        '''

        super().__init__(data_info=data_info, params=params)

        # self.params = self._check_model_params(params)

        # set object attributes
        self.weights_loaded = False
        self.model = self._build_model()

        # load model weights if specified
        if self.params['weights_path'] is not None:
            self.load_model(self.params['weights_path'])
            self.weights_loaded = True
            self.trained = True


    def load_block(self, path):
        ''' 
        Load model saved at path into this model.
        Arguments:
            path (str) : path to saved weights.
        Returns: 
            None
        '''

        self.load_model(path)
        self.trained = True

    def save_block(self, path):
        ''' 
        Save trained model to specified path.
        
        Arguments:
            path (str) : path to save to.
        Returns: 
            None
        '''

        self.save_model(path)

    def train(self, dataset, prev_results=None):
        ''' 
        Full training loop. Constructs t.data.Dataset for training and
        testing, updates model weights each epoch and evaluates on test set
        periodically.

        Arguments:
            dataset (Dataset): Dataset object containing X and Y data for this
                training run.
            best (bool) : whether to use weights from epoch with best test-loss,
                or from most recent epoch for future prediction.
        Returns:
            dict : dictionary of CDE training results. Specifically, this will 
                contain `pyx`, the predicted conditional probabilites for the 
                training dataset. 
        '''
        #TODO: do a more formalized checking that actual dimensions match 
        # expected
        #TODO: say what expected vs actual are

        if self.weights_loaded:
            print('No need to train CDE, specified weights loaded already.')
            return {'pyx' : self.model.predict(dataset.X)}

        # train-test split
        dataset.split_data = train_test_split(dataset.X, dataset.Y, 
                                              shuffle=True, train_size=0.75)

        Xtr, Xts, Ytr, Yts = dataset.split_data

        # build optimizer
        optimizer = tf.keras.optimizers.get(
            { 'class_name' : self.params['optimizer'],
              'config' : self.params['opt_config']})

        # compile model
        self.model.compile(
            loss=self.params['loss'],
            optimizer=optimizer,
        )

        # specify checkpoint save callback
        callbacks = []
        if self.params['best']:
            # handle case where user interrupted previous training session
            # before tmp_checkpoints clean-up code was executed.
            if os.path.exists('tmp_checkpoints'):
                if self.params['verbose']>0:
                    print('Warning: deleting tmp_checkpoints directory.')
                shutil.rmtree('tmp_checkpoints')
            os.mkdir('tmp_checkpoints')
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath='tmp_checkpoints/best_weights',
                save_weights_only=True,
                monitor='val_loss',
                mode='min',
                save_best_only=True)
            callbacks = [model_checkpoint_callback]


        # train model
        history = self.model.fit(
            Xtr, Ytr,
            batch_size=self.params['batch_size'],
            epochs=self.params['n_epochs'],
            validation_data=(Xts,Yts),
            callbacks=callbacks,
            verbose=self.params['verbose']
        )

        # handle results
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        fig = self._graph_results(train_loss, val_loss, 
            show=self.params['show_plot'])
        pyx = self.model.predict(dataset.X)

        # load in best weights if specified
        if self.params['best']:
            self.load_model('tmp_checkpoints/best_weights')
            shutil.rmtree('tmp_checkpoints')


        results_dict = {'train_loss' : train_loss,
                        'val_loss' : val_loss,
                        'loss_plot' : fig,
                        'model_weights' : self.model.get_weights(),
                        'pyx' : pyx}


        self.trained = True
        return results_dict


    def _graph_results(self, train_loss, val_loss, show=True):
        '''
        Graph training and testing loss across training epochs.

        Arguments:
            train_loss (np.ndarray) : (n_epochs,) array of training losses per 
                epoch.
            val_loss (np.ndarray) : (n_epochs,) array of validation losses per 
                epoch.
            show (bool) : displays figure if show=True. Defaults to True. 
        Returns:
            plt.figure : figure object.
        '''
        fig,ax = plt.subplots()
        ax.plot(range(len(train_loss)), train_loss, label='train_loss')
        ax.plot(np.linspace(0,len(train_loss),len(val_loss)).astype(int), 
            val_loss, label='val_loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel(self.params['loss'])
        ax.set_title('Training and Test Loss')
        plt.legend(loc='upper right')

        if show:
            plt.show()
        else:
            plt.close()
        return fig

    def predict(self, dataset, prev_results=None):
        ''' 
        Given a Dataset of microvariable observations, estimate macrovariable
        states.

        Arguments:
            dataset (Dataset): Dataset object containing X and Y data to
                estimate macrovariable states for.
        Returns:
            dict : dictionary of prediction results. Specifically, this dictionary will
                contain `pyx`, the predicted conditional probabilites for the 
                given Dataset. 
        '''
        
        assert self.trained, "Remember to train the model before prediction."
        pyx = self.model.predict(dataset.X)

        results_dict = {'pyx' : pyx}
        return results_dict

    def evaluate(self, dataset):
        ''' 
        Compute the loss specified in self.params['loss] between ground truth 
        and model prediction.

        Arguments:
            dataset (Dataset): Dataset object containing X and Y data for this
                        training run
        Returns: 
            float : the average loss for this batch
        '''

        assert self.trained, "Remember to train the model before evaluation."

        Y_hat = self.predict(dataset)
        loss_fxn = tf.keras.losses.get(self.params['loss'])
        cost = loss_fxn(dataset.Y, Y_hat)
        return tf.reduce_mean(cost)


    def load_model(self, file_path):
        ''' 
        Load model weights from saved checkpoint into current model.

        Arguments:
            file_path (str) : path to checkpoint file
        Returns: 
            None
        '''

        assert hasattr(self, 'model'), 'Build model before loading parameters.'

        if self.params['verbose']>0:
            print("Loading parameters from ", file_path)
        self.model.load_weights(file_path)

        #TODO: does tensorflow keep track of if model is trained? 
        self.trained = True

    def save_model(self, file_path):
        ''' 
        Save model weights from current model.

        Arguments:
            file_path (str) : path to checkpoint file
        Returns: 
            None
        '''
        # TODO : add check to only save trained models? (bc of load model 
        # setting train to true )
        if self.params['verbose']>0:
            print("Saving parameters to ", file_path)
        self.model.save_weights(file_path)

    @abstractmethod
    def _build_model(self):
        ''' 
        Define the neural network based on specifications in self.params.

        Arguments:
            None
        Returns: 
            tf.keras.models.Model : untrained model specified in self.params.
        '''
        ...
