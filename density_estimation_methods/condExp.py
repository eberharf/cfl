import os
# TODO: add GPU support
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from density_estimation_methods.cde import CDE #base class


#TODO: this code hasn't been transferred from CFL.py yet 
# #shuffles and splits data into training and testing sets
#         self.X_tr, self.X_ts, self.Y_tr, self.Y_ts = \
#             train_test_split(X, Y, shuffle=True, train_size=training_data_size, random_state=RANDOM_SEED)
#
#         self.normalizeData() #normalize each chunk of data to have mean of 0 and var 1
#
# def normalizeData(self):
#         """
#         independently normalizes each training and testing set to have mean 0 and var 1.
#         The scaling vector is calculated separately for training and testing so that there is
#         no leak of information about the testing data into the training data
#         """
#         self.X_tr = self.normalizeHelper(self.X_tr)
#         self.X_ts = self.normalizeHelper(self.X_ts)
#         self.Y_tr = self.normalizeHelper(self.Y_tr)
#         self.Y_ts = self.normalizeHelper(self.Y_ts)
#
#     def normalizeHelper(self, data):
#         """for a inputted data set, calculated the scaler to normalize the data and applies the transformation"""
#         scaler = StandardScaler().fit(data)
#         normalized = scaler.transform(data).astype('float32')
#         return normalized




example_params = {'batch_size': 128, 'lr': 1e-3, 'optimizer': tf.keras.optimizers.Adam(lr=1e-3), 'n_epochs': 100, 'test_every': 10, 'save_every': 10}

class CondExp(CDE):

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

    def train(self, Xtr, Ytr, Xts, Yts, save_dir):
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
        # TODO: make validation set optional (is this really helpful?)
        # TODO: standardize save path structure
        # TODO: save each checkpoint to different name

        # Setup
         #TODO: instead of individually assigning everything here,
         # check that dictionary only contains valid parameters and
         # then assign everything in dict to a variable
         # or something
        batch_size = self.model_params['batch_size']
        lr = self.model_params['lr']
        optimizer = self.model_params['optimizer']
        n_epochs = self.model_params['n_epochs']
        test_every = self.model_params['test_every']
        save_every = self.model_params['save_every']

        if self.verbose:
            self.model.summary()

        # Construct train and test datasets (load, shuffle, set batch size)
        dataset_tr = tf.data.Dataset.from_tensor_slices((Xtr, Ytr)).shuffle(Xtr.shape[0]).batch(batch_size)
        dataset_ts = tf.data.Dataset.from_tensor_slices((Xts, Yts)).shuffle(Xts.shape[0]).batch(batch_size)

        train_losses = []
        test_losses = []

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
                    test_loss(self.evaluate(test_x, test_y, training=False))
                test_losses.append(test_loss.result())

                print('Epoch {}/{}: train_loss: {}, test_loss: {}'.format(
                    i, n_epochs, train_losses[-1], test_losses[-1]))

            if i % save_every == 0:
                self.save_parameters(save_dir + "_{}".format(i))

        if self.verbose:
            self.graph_results(train_losses, test_losses)

        return None


    def graph_results(self, train_losses, test_losses):
        '''graphs the training vs testing loss across all epochs of training'''
        plt.plot(range(len(train_losses)), train_losses)
        plt.plot(np.linspace(0,len(train_losses),len(test_losses)).astype(int), test_losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss')
        plt.legend(['Train', 'Test'])
        plt.show()

    def predict(self, X, Y=None): #put in the x and y you want to predict with
        ''' Given a set of observations X, get neural network output.
            Arguments:
                X : model input of dimensions [# observations, # x_features] (np.array)
                Y : model input of dimensions [# observations, # y_features] (np.array)
                    note: this derivation of CDE doesn't require Y for prediction.
            Returns: model prediction (np.array) (TODO: check if this is really np.array or tf.Tensor)
        '''
        if Y:
            raise RuntimeWarning("Y was passed as an argument, but is not being used for prediction.")
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
            Arguments: None
            Returns: the model (tf.keras.models.Model object)
        '''

        # Network
        input_layer = tf.keras.Input(shape=(self.data_info['X_dims'][1],),
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
                                    units=self.data_info['Y_dims'][1],
                                    activation='linear',
                                    kernel_initializer='he_normal',
                                    activity_regularizer=tf.keras.regularizers.l2(0.0001),
                                    name='nn_output_layer')(layer)
        model = tf.keras.models.Model(input_layer, output_layer)
        return model