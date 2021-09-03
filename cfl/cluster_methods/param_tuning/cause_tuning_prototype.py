
import numpy as np
import os
from optuna.visualization import plot_slice
import optuna
from sklearn.cluster import KMeans
from cfl.experiment import Experiment
from sklearn.linear_model import LinearRegression as LR
from cfl.util.data_processing import one_hot_encode
import plotly.graph_objects as go
import matplotlib.pyplot as plt


BASE_PATH = '/Users/imanwahle/Desktop/cfl/cfl/cluster_methods/param_tuning'
DATA_DIR = 'toy_data_7_3'
CDE_PATH = os.path.join(BASE_PATH, f'{DATA_DIR}/experiment0000/trained_blocks/CDE')
RESULTS_PATH = os.path.join(BASE_PATH, f'{DATA_DIR}/cfl_results')

# load data
data = np.load(os.path.join(BASE_PATH, f'{DATA_DIR}/data.npz'), allow_pickle=True)
X,Y,Xbar,Ybar,params = data['X'],data['Y'],data['Xbar'],data['Ybar'],data['params']
print(X.shape, Y.shape)

def run_cfl_helper(n_cause_clusters, n_effect_clusters=None):

    data_info = {'X_dims' : X.shape, 'Y_dims' : Y.shape, 'Y_type' : 'continuous'}
    cde_params = {  'dense_units' : [300, 300, 300, data_info['Y_dims'][1]], 
                    'activations' : ['relu', 'relu', 'relu', 'linear'],
                    'dropouts'    : [0, 0, 0, 0],
                    'batch_size'  : 128,
                    'n_epochs'    : 100,
                    'optimizer'   : 'adam',
                    'opt_config'  : {'lr' : 1e-3},
                    'loss'        : 'mean_squared_error',
                    'best'        : True,
                    'verbose'     : 1, 
                    'show_plot'   : False,
                    'weights_path' : CDE_PATH}
    if n_effect_clusters is None:
        cluster_params = {'x_model' : KMeans(n_clusters=n_cause_clusters),
                          'y_model' : None}
    else:
        cluster_params = {'x_model' : KMeans(n_clusters=n_cause_clusters),
                          'y_model' : KMeans(n_clusters=n_effect_clusters)}
    block_names = ['CondExpMod', 'Clusterer']
    block_params = [cde_params, cluster_params]
    my_exp = Experiment(X_train=X, 
                        Y_train=Y, 
                        data_info=data_info, 
                        block_names=block_names, 
                        block_params=block_params, 
                        results_path=RESULTS_PATH)
    return my_exp.train()

def score(true, pred):
    return np.mean(np.power(true-pred, 2))

def compute_predictive_error(Xlr, Ylr, n_iter=100):
    
    # reshape data if necessary
    if Xlr.ndim==1:
        Xlr = np.expand_dims(Xlr, -1)
    if Ylr.ndim==1:
        Ylr = np.expand_dims(Ylr, -1)

    n_samples = Xlr.shape[0]
    errs = np.zeros((n_iter,))
    for ni in range(n_iter):
        
        # choose samples to withhold
        in_sample_idx = np.zeros((n_samples,))
        in_sample_idx[np.random.choice(n_samples, int(n_samples*0.95), replace=False)] = 1
        in_sample_idx = in_sample_idx.astype(bool)

        # get predictive error
        reg = LR().fit(Xlr[in_sample_idx], Ylr[in_sample_idx])
        pred = reg.predict(Xlr[~in_sample_idx])
        errs[ni] = score(Ylr[~in_sample_idx], pred)

    return np.mean(errs)


def cause_tuning(n_cause_clusters):
    # run cfl
    results = run_cfl_helper(n_cause_clusters)
    
    # compute error measure (error in predicting pyx from Xbarhat)
    err = compute_predictive_error(
        one_hot_encode(results['Clusterer']['x_lbls'], range(n_cause_clusters)), 
        results['CDE']['pyx'])  
    return err



# compute errors over varied # clusters
n_cause_clusters_range = range(2,20)
errs = np.zeros((len(n_cause_clusters_range),))
for ncci,ncc in enumerate(n_cause_clusters_range):
    errs[ncci] = cause_tuning(ncc)

# save errors
np.savez(os.path.join(BASE_PATH, DATA_DIR, 'cause_errors'),
         n_cause_clusters_range=n_cause_clusters_range,
         errs=errs)

# plot errors
fig,ax = plt.subplots()
ax.scatter(n_cause_clusters_range, errs)
ax.set_xticks(n_cause_clusters_range)
ax.set_xlabel('n_cause_clusters')
ax.set_ylabel('predictive_error')
ax.set_title('n_cause_clusters tuning curve')

if not os.path.exists(os.path.join(BASE_PATH, DATA_DIR, 'figures')):
    os.mkdir(os.path.join(BASE_PATH, DATA_DIR, 'figures'))
plt.savefig(os.path.join(BASE_PATH, DATA_DIR, 'figures/cause_errors'))
plt.show()

