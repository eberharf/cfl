# Standard modules
import time
import logging
import copy
import cPickle as pkl

# Numpy and Scipy
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# Pylearn2 and Theano
import theano
import pylearn2
from pylearn2.datasets import preprocessing

# Custom modules
import core_causal_features
import dataset_binary_gratings
import ai_gratings
import helpers

# Seed the rng.
np.random.seed(1423)
def prepare_datasets(agent, n_training=10000, 
                     im_shape=(10,10), noise_lvl=0):
    print('Creating an observational gratings dataset...')
    data_train = dataset_binary_gratings.\
                 GRATINGS(n_samples=n_training*5/4, 
                          agent=agent, im_shape=im_shape, 
                          noise_lvl=noise_lvl)

    # Normalize the data.
    # preprocessor = preprocessing.Standardize()
    # data_train.apply_preprocessor(preprocessor, can_fit=True)
    # data_train.X = data_train.X-0.5

    data_valid = copy.deepcopy(data_train)
    split_id = int(4*np.floor(data_train.X.shape[0]/5)) 
    data_train.X = data_train.X[:split_id]
    data_train.y = data_train.y[:split_id]
    data_valid.X = data_valid.X[split_id:]
    data_valid.y = data_valid.y[split_id:]
    data_test = dataset_binary_gratings.\
                GRATINGS(n_samples=100, agent=agent, 
                         im_shape=im_shape, noise_lvl=noise_lvl)
    # data_test.apply_preprocessor(preprocessor, can_fit=False)
    # data_test.X = data_test.X-0.5

    print('Done. Training set size {0}. Validation set size {1}.\n'.\
                             format(data_train.y.size,data_valid.y.size))
    return (data_train, data_valid, data_test)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# START MAIN SCRIPT
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 1. Set file names.
curT = time.localtime()
MODEL_FNAME = 'EXPERIMENTS/GRATINGS_MLP.pkl'.\
              format(curT.tm_year, curT.tm_mon, curT.tm_mday, 
                     curT.tm_hour, curT.tm_min, curT.tm_sec)

DATA_FNAME = 'EXPERIMENTS/Data_GRATING_{0}_{1}_{2}_{3}_{4}_{5}.pkl'.\
              format(curT.tm_year, curT.tm_mon, curT.tm_mday, 
                     curT.tm_hour, curT.tm_min, curT.tm_sec)

REPORT_FNAME = 'EXPERIMENTS/Report_GRATING_{0}_{1}_{2}_{3}_{4}_{5}.pdf'.\
               format(curT.tm_year, curT.tm_mon, curT.tm_mday, 
                      curT.tm_hour, curT.tm_min, curT.tm_sec)

LOGS_FNAME = 'EXPERIMENTS/Logs_GRATING_{0}_{1}_{2}_{3}_{4}_{5}.log'.\
              format(curT.tm_year, curT.tm_mon, curT.tm_mday, 
                     curT.tm_hour, curT.tm_min, curT.tm_sec)
#%%%%%%%%
# 2. Configure logging.
logging.getLogger('pylearn2').setLevel(logging.ERROR)

#%%%%%%%%
# 3. Set other parameters.
IM_SHAPE = (10, 10)
N_TRAIN = 10000
N_BATCHES = 10
QUERY_BATCH_SIZE = 100
RETRAIN = 0
DATA_NOISE = 0.03
MANIP_THR = 0.05
MANIP_ALPHA = 0.5
agent = ai_gratings.Ai_Causal(visual_reshape=IM_SHAPE)

#%%%%%%%%
# 4. Create the observational dataset.
(data_train, data_valid, data_test) =\
    prepare_datasets(agent, n_training=N_TRAIN, 
                     im_shape=IM_SHAPE, noise_lvl=DATA_NOISE)

#%%%%%%%%
# 5. Train an observational MLP or load a pre-trained model.
train_params = {'data_train': data_train,
                'data_valid': data_valid,
                'data_test': data_test,
                'model_fname': MODEL_FNAME,
                'arch': [50, 50],
                'learning_rate': 0.001,
                'max_iters': 250,
                'max_after_best': 100,
                'batch_size': 10}
if RETRAIN:
    core_causal_features.train_mlp(**train_params)
    model_obs = pylearn2.utils.serial.load(MODEL_FNAME)
else:
    model_obs = pylearn2.utils.\
                serial.load('EXPERIMENTS/GRATINGS_MLP.pkl')

print('Observational model trained. Validation error {:.2g}'.\
      format(float(model_obs.monitor.channels['valid_y_misclass'].\
                   val_record[-1])))
data_obs = data_train

#%%%%%%%%
# 6. Retrain the model using the Causal Coarsening Theorem
(model_csl, data_csl_train, data_csl_valid) = core_causal_features.\
    causal_coarsening(model_obs=model_obs, data_obs_train=data_obs, 
                      data_obs_valid=data_valid, agent=agent, 
                      n_queries=10, train_params=train_params)

#%%%%%%%%
# 8. Learn the manipulator.
for batch_id in range(N_BATCHES):
    print('\n\nCausal query batch {}/{}'.format(batch_id, N_BATCHES))
    # 8.2. Perform the causal queries on current batch, including
    #    manipulated versions on the batch.
    manip_pool = core_causal_features.prepare_img_queries(data_train=data_csl_train,
            model=model_csl, n_queries=QUERY_BATCH_SIZE, im_shape=[1,100],
            alpha=MANIP_ALPHA, augment_thr=MANIP_THR, augment_coeff=np.inf)
    
    # 8.2a. Annotate the queries using a simulated agent and incorporate them into the data.
    (data_train, data_valid) = helpers.incorporate_queries(manip_pool=manip_pool, agent=agent,
            data_train=data_csl_train, data_valid=data_csl_valid) 

    # 8.5. Continue training the classifier, starting from previous
    #     weights, using the current full causal dataset.
    train_params['data_train'] = data_train
    train_params['data_valid'] = data_valid
    train_params['model'] = None 
    core_causal_features.train_mlp(**train_params)
    model_csl = pylearn2.utils.serial.load(train_params['model_fname'])
