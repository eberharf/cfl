from abc import abstractmethod
import os
import shutil
import datetime  # for creating ID
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from cfl.dataset import Dataset
from cfl.cond_density_estimation.cde_model import CDEModel

class CondExpBase(CDEModel):
    ''' 
    A class to define, train, and perform inference with conditional density
    estimators that fall under the "conditional expectation" umbrella. This
    subset of conditional density estimators (referred to as 'CondExp') learns
    E[P(Y|X)] instead of the full conditional distribution. This base class
    implements all functions needed for training and prediction, and supplies
    a model architecture that can be overridden by children of this class. In
    general, if you would like to use a CondExp CDE for your CFL pipeline, it is
    easiest to either 1) use the CondExpDIY child class of CondExpBase that
    allows you to define your network through a function specified in 
    `model_params`, 2) use the condExpMod child class which allows you to pass 
    in limited architecture specifications through the params attribute, or
    3) inherit this class and override the methods you would like to modify. 

    Attributes:
        name (str) : name of the model so that the model type can be recovered 
            from saved parameters (str)
        data_info (dict) : dict with information about the dataset shape
        default_params (dict) : default parameters to fill in if user doesn't 
            provide a given entry
        model_params (dict) : parameters for the CDE that are passed in by the 
            user and corrected by check_save_model_params
        trained (bool) : whether or not the modeled has been trained yet. This 
            can either happen by defining by instantiating the class and
            calling train, or by passing in a path to saved weights from
            a previous training session through model_params['weights_path'].
        model (tf.keras.Model.Sequential) : tensorflow model for this CDE

    Methods:
        get_model_params : return self.model_params
        load_model : load everything needed for this CondExpBase model
        save_model : save the current state of this CondExpBase model
        train : train the neural network on a given Dataset
        _graph_results : helper function to graph training and validation loss
        predict : once the model is trained, predict for a given Dataset
        load_network : load tensorflow network weights from a file into
            self.network
        save_network : save the current weights of self.network
        _build_network : create and return a tensorflow network
        _check_format_model_params : check dimensionality of provided 
            parameters and fill in any missing parameters with defaults.
    '''

    def __init__(self, data_info, model_params):
        ''' 
        Initialize model and define network.

        Arguments:
            data_info (dict) : a dictionary containing information about the 
                data that will be passed in. Should contain 'X_dims',
                'Y_dims', and 'Y_type' as keys.
            model_params (dict) : dictionary containing parameters for the model.
            model (str) : name of the model so that the model type can be
                recovered from saved parameters.
        Returns: 
            None
        '''
        self.name = 'CondExpBase'
        self.data_info = data_info
        self.model_params = model_params

        # set object attributes
        self.network = self._build_network()
        self.trained = False
        
        # load model weights if specified
        if self.model_params['weights_path'] is not None:
            self.load_network(self.model_params['weights_path'])
            self.trained = True

    def get_model_params(self):
        ''' Get parameters for this CDE model.
            Arguments: None
            Returns: 
                dict: dictionary of parameter names (keys) and values (values)
        '''

        return self.model_params

    def load_model(self, path):
        ''' 
        Load model saved at path into this model.
        Arguments:
            path (str) : path to saved weights.
        Returns: 
            None
        '''

        assert isinstance(path, str), 'path should be a str of path to block.'
        self.load_network(path)
        self.trained = True

    def save_model(self, path):
        ''' 
        Save trained model to specified path.

        Arguments:
            path (str) : path to save to.
        Returns: 
            None
        '''
        assert isinstance(path, str), 'path should be a str of path to block.'
        self.save_network(path)

    def train(self, dataset, prev_results=None):
        ''' 
        Full training loop. Constructs t.data.Dataset for training and
        testing, updates model weights each epoch and evaluates on test set
        periodically.

        Arguments:
            dataset (Dataset): Dataset object containing X and Y data for this
                training run.
            prev_results (dict): dictionary that contains any results generated
                in previous Block in Experiment pipeline (usually none for CDE).
        Returns:
            dict : dictionary of CDE training results. Specifically, this will 
                contain `pyx`, the predicted conditional probabilites for the 
                training dataset. 
        '''

        assert isinstance(dataset, Dataset), 'dataset is not a Dataset.'
        assert isinstance(prev_results, (type(None), dict)),\
            'prev_results is not NoneType or dict'
        if self.trained:
            print('Model has already been trained, will return predictions ' +
                  'on training data.')
            return {'pyx': self.network.predict(dataset.X)}

        # train-test split
        if dataset.get_in_sample_idx() is None:
            Xtr, Xva, Ytr, Yva, in_sample_idx, out_sample_idx = \
                train_test_split(dataset.X, dataset.Y,
                                 range(dataset.X.shape[0]), shuffle=True,
                                 train_size=0.75)
            dataset.set_in_sample_idx(in_sample_idx)
            dataset.set_out_sample_idx(out_sample_idx)

        else:
            Xtr = dataset.X[dataset.get_in_sample_idx()]
            Ytr = dataset.Y[dataset.get_in_sample_idx()]
            Xva = dataset.X[dataset.get_out_sample_idx()]
            Yva = dataset.Y[dataset.get_out_sample_idx()]

        # build optimizer
        optimizer = tf.keras.optimizers.get(
            {'class_name': self.model_params['optimizer'],
             'config': self.model_params['opt_config']})

        # compile model
        self.network.compile(
            loss=self.model_params['loss'],
            optimizer=optimizer,
        )

        # log GPU device if available
        device_name = tf.test.gpu_device_name()
        if self.model_params['verbose'] > 0:
            if device_name is not '':
                print('Using GPU device: ', device_name)
            else:
                print('No GPU device detected.')

        try:
            # specify checkpoint save callback
            callbacks = []

            # if we want to return the best weights (rather than the weights at the
            # end of training)
            if self.model_params['best']:

                # give the checkpoints path a unique ID (so that it doesn't get
                # confused with other CFL runs)
                now = datetime.datetime.now()
                # this creates a string based on the current date and time up 
                # to the second (NOTE: if you create a bunch of CFLs all at 
                # once maybe you'd need a more precise ID)
                dt_id = now.strftime("%d%m%Y%H%M%S")
                checkpoint_path = self.model_params['checkpoint_name']+dt_id
                os.mkdir(checkpoint_path)

                # ModelCheckpoint saves network checkpoints to specified path 
                # during training
                best_path = os.path.join(checkpoint_path, 'best_weights')
                network_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                    filepath=best_path,
                    save_weights_only=True,
                    monitor='val_loss',
                    mode='min',
                    save_best_only=True)
                callbacks = [network_checkpoint_callback]

            if self.model_params['tb_path'] is not None:
                tb_callback = tf.keras.callbacks.TensorBoard(
                    log_dir=self.model_params['tb_path'])
                callbacks = [tb_callback] + callbacks

            if self.model_params['optuna_callback'] is not None:
                callbacks = [self.model_params['optuna_callback']] + callbacks

            if self.model_params['early_stopping']:
                es_callback = tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=20)
                callbacks = [es_callback] + callbacks

            # train network
            history = self.network.fit(
                Xtr, Ytr,
                batch_size=self.model_params['batch_size'],
                epochs=self.model_params['n_epochs'],
                validation_data=(Xva, Yva),
                callbacks=callbacks,
                verbose=self.model_params['verbose']
            )

            # handle results
            train_loss = history.history['loss']
            val_loss = history.history['val_loss']
            fig = self._graph_results(train_loss, val_loss,
                                      show=self.model_params['show_plot'])
            pyx = self.network.predict(dataset.X)

            # load in best weights if specified
            if self.model_params['best']:
                # TODO: this is where the error is jenna
                self.load_network(best_path)

            results_dict = {'train_loss': train_loss,
                            'val_loss': val_loss,
                            'loss_plot': fig,
                            'network_weights': self.network.get_weights(),
                            'pyx': pyx}

            self.trained = True

        # we want to delete the checkpoints directory at the end, even if 
        # something messed up during training
        finally:
            if self.model_params['best']:
                shutil.rmtree(checkpoint_path)
        return results_dict

    def _graph_results(self, train_loss, val_loss, show=True):
        '''
        Graph training and validation loss across training epochs.

        Arguments:
            train_loss (np.ndarray) : (n_epochs,) array of training losses per 
                epoch.
            val_loss (np.ndarray) : (n_epochs,) array of validation losses per 
                epoch.
            show (bool) : displays figure if show=True. Defaults to True. 
        Returns:
            matplotlib.pyplot.figure : figure object.
        '''
        fig, ax = plt.subplots()
        ax.plot(range(len(train_loss)), train_loss, label='train_loss')
        ax.plot(range(len(val_loss)), val_loss, label='val_loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel(self.model_params['loss'])
        ax.set_title('Training and Validation Loss')
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
            prev_results (dict): dictionary that contains any results generated
                in previous Block in Experiment pipeline (usually none for CDE).
        Returns:
            dict : dictionary of prediction results. Specifically, this 
                dictionary will contain `pyx`, the predicted conditional 
                probabilites for the given Dataset. 
        '''

        assert isinstance(dataset, Dataset), 'dataset is not a Dataset.'
        assert isinstance(prev_results, (type(None), dict)),\
            'prev_results is not NoneType or dict'

        assert self.trained, "Remember to train the network before prediction."
        pyx = self.network.predict(dataset.X)

        results_dict = {'pyx': pyx}
        return results_dict


    def load_network(self, file_path):
        ''' 
        Load network weights from saved checkpoint into current network.

        Arguments:
            file_path (str) : path to checkpoint file
        Returns: 
            None
        '''

        assert hasattr(self, 'network'), 'Build network before loading parameters.'

        if self.model_params['verbose'] > 0:
            print("Loading parameters from ", file_path)
        try:
            # specify "expect_partial()" to let tf know that we won't be using
            # all vars that were saved from training bc we are just doing
            # prediction now. Alternative is to not save the extra params in 
            # the first place, but will currently avoid that to encourage
            # reproducibility. 
            # https://stackoverflow.com/questions/58289342/tf2-0-translation-model-error-when-restoring-the-saved-model-unresolved-objec
            self.network.load_weights(file_path).expect_partial()
        except:
            raise ValueError('path does not exist.')

        # TODO: does tensorflow keep track of if network is trained?
        self.trained = True

    def save_network(self, file_path):
        ''' 
        Save network weights from current network.

        Arguments:
            file_path (str) : path to checkpoint file
        Returns: 
            None
        '''
        # TODO : add check to only save trained networks? (bc of load network
        # setting train to true )
        if self.model_params['verbose'] > 0:
            print("Saving parameters to ", file_path)
        try:
            self.network.save_weights(file_path)
        except:
            raise ValueError('path does not exist.')

    @abstractmethod
    def _build_network(self):
        ''' 
        Define the neural network based on specifications in self.model_params.

        Arguments:
            None
        Returns: 
            tf.keras.models.Model : untrained network specified in self.model_params.
        '''
        ...

    @abstractmethod
    def _check_format_model_params(self):
        '''
        Make sure all required model_params are specified and of appropriate 
        dimensionality. Replace any missing model_params with defaults,
        and resolve any simple dimensionality issues if possible.
        
        Arguments:
            None
        Returns:
            None
        Raises:
            AssertionError : if params are misspecified and can't be 
                             automatically fixed.
        '''
        ...
    
    @abstractmethod
    def _get_default_model_params(self):
        ''' 
        Returns the default parameters specific to this type of model.

        Arguments: None
        Returns:
            dict : dictionary of default parameters
        '''
        ...