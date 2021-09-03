import numpy as np
import os
from optuna.visualization import plot_slice
import optuna
from sklearn.cluster import KMeans
from cfl.experiment import Experiment
from sklearn.linear_model import LinearRegression as LR
from sklearn.datasets import make_blobs
from cfl.util.data_processing import one_hot_encode
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from tqdm import tqdm

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

n_iter = 10
n_centers_range = range(2,11)
n_cause_clusters_range = range(2,20)
errs = np.zeros((n_iter, len(n_centers_range), len(n_cause_clusters_range)))
for ni in range(n_iter):
    for i,n_centers in tqdm(enumerate(n_centers_range)):
        # construct dataset
        data,true_lbls = make_blobs(n_samples=1000, n_features=2, centers=n_centers)

        # compute errors over varied # clusters
        for j,ncc in enumerate(n_cause_clusters_range):
            lbls = KMeans(n_clusters=ncc).fit_predict(data)
            errs[ni,i,j] = compute_predictive_error(one_hot_encode(lbls, range(ncc)), data)

# plot errors
fig,axs = plt.subplots(2,len(n_centers_range)//2,figsize=(30,8),sharey=True)
for (i,n_centers),ax in zip(enumerate(n_centers_range),axs.ravel()):
    ax.errorbar(n_cause_clusters_range, np.sum(errs[:,i,:], axis=0), np.std(errs[:,i,:],axis=0))
    ax.set_xticks(n_cause_clusters_range)
    ax.set_xlabel('n_clusters')
    ax.set_ylabel('predictive_error')
    ax.set_title(f'g.t. {n_centers} clusters tuning curve')

plt.savefig('/Users/imanwahle/Desktop/cfl/cfl/cluster_methods/param_tuning/cluster_tuning_sanity_check.png')
plt.show()




