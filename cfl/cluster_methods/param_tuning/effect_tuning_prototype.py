
import numpy as np
import os
from optuna.visualization import plot_slice
import optuna
from scipy.sparse import base
from sklearn.cluster import KMeans
from cfl.experiment import Experiment
from sklearn.linear_model import LinearRegression as LR
from cfl.util.data_processing import one_hot_encode
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

BASE_PATH = '/Users/imanwahle/Desktop/cfl/cfl/cluster_methods/param_tuning'
DATA_DIR = 'toy_data_4_6'
n_cause_clusters = 4
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

def effect_tuning(n_cause_clusters, n_effect_clusters):
    # run cfl
    results = run_cfl_helper(n_cause_clusters, n_effect_clusters)
    
    # compute error measure (error in predicting y_probs from Ybarhat)
    err = compute_predictive_error(
        one_hot_encode(results['Clusterer']['y_lbls'], range(n_effect_clusters)), 
        results['Clusterer']['y_probs'])  
    return err



# make fig dir
if not os.path.exists(os.path.join(BASE_PATH, DATA_DIR, 'figures')):
    os.mkdir(os.path.join(BASE_PATH, DATA_DIR, 'figures'))


############################ DEBUGGING PLOTS ##################################

# plot y_probs corner plot
results = run_cfl_helper(n_cause_clusters, 3)
y_probs = results['Clusterer']['y_probs']
fig,ax = plt.subplots(n_cause_clusters, n_cause_clusters, 
                      figsize=(2*n_cause_clusters,2*n_cause_clusters),
                      sharex=True, sharey=True)
for i in range(n_cause_clusters):
    for j in range(n_cause_clusters):
        ax[j,i].scatter(y_probs[:,i], y_probs[:,j], s=2)
        if j==n_cause_clusters-1:
            ax[j,i].set_xlabel(f'P(Y=y | Xmacro={i})')
        if i==0:
            ax[j,i].set_ylabel(f'P(Y=y | Xmacro={j})')

plt.savefig(os.path.join(BASE_PATH, DATA_DIR, 'figures/y_probs_dist'))
# plt.show()

# plot y_probs projection
pca = PCA(n_components=2).fit(y_probs)
proj = pca.transform(y_probs)
exp_var = pca.explained_variance_ratio_
fig,ax = plt.subplots()
ax.scatter(proj[:,0], proj[:,1], s=2)
ax.set_xlabel('PC0')
ax.set_ylabel('PC1')
ax.set_title(f'y_probs 2 PCs\n% var. exp. = {exp_var}')
plt.savefig(os.path.join(BASE_PATH, DATA_DIR, 'figures/y_probs_pcs'))
# plt.show()

# plot y_probs for each Ybar
uYbars = np.unique(Ybar)
fig,axs = plt.subplots(1, len(uYbars), figsize=(18,4), sharex=True, sharey=True)
for ax,yb in zip(axs, np.unique(uYbars)):
    ax.errorbar(range(y_probs.shape[1]), 
                np.mean(y_probs[Ybar==yb],axis=0),
                np.std(y_probs[Ybar==yb],axis=0)
    )
    ax.set_xlabel('Xmacro class')
    ax.set_ylabel('f(P(Y=y|Xmacro))')
    ax.set_title(f'Ymacro = {yb}')
    ax.set_xticks(range(y_probs.shape[1]))
plt.savefig(os.path.join(BASE_PATH, DATA_DIR, 'figures/y_probs_per_Ybar'))
# plt.show()

# plot predicted Xbar
fig,ax = plt.subplots()
Xbarhat = results['Clusterer']['x_lbls']
for i in np.unique(Xbarhat):
    ax.scatter(X[Xbarhat==i], Y[Xbarhat==i], label=i, s=2, alpha=0.8)
ax.legend()
ax.set_title('Microvariables colored by Xbarhat')
ax.set_xlabel('X micro')
ax.set_ylabel('Y micro')

plt.savefig(os.path.join(BASE_PATH, DATA_DIR, 'figures/Xbarhat'))
# plt.show()


###############################################################################

# compute errors over varied # clusters
n_effect_clusters_range = range(2,20)
errs = np.zeros((len(n_effect_clusters_range),))
for ncci,ncc in enumerate(n_effect_clusters_range):
    errs[ncci] = effect_tuning(n_cause_clusters, ncc)

# save errors
np.savez(os.path.join(BASE_PATH, DATA_DIR, 'effect_errors'),
         n_effect_clusters_range=n_effect_clusters_range,
         errs=errs)

# plot errors
fig,ax = plt.subplots()
ax.scatter(n_effect_clusters_range, errs)
ax.set_xticks(n_effect_clusters_range)
ax.set_xlabel('n_effect_clusters')
ax.set_ylabel('predictive_error')
ax.set_title('n_effect_clusters tuning curve')


plt.savefig(os.path.join(BASE_PATH, DATA_DIR, 'figures/effect_errors'))
# plt.show()


