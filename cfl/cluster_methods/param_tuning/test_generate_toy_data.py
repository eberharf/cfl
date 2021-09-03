

import numpy as np
import matplotlib.pyplot as plt
from generate_toy_data import *

def test_shapes():
    X, Y, Xbar, Ybar, params = create_dataset()
    
    assert X.shape[0]==Y.shape[0]==Xbar.shape[0]==Ybar.shape[0], \
        'should all have n_samples'
    
    pyx = params['pyx']
    assert len(np.unique(Xbar))==pyx.shape[0], \
        f'{len(np.unique(Xbar))}, {pyx.shape[0]}'
    assert len(np.unique(Ybar))==pyx.shape[1], \
        f'{len(np.unique(Ybar))}, {pyx.shape[1]}'

def test_plot_data():

    X, Y, Xbar, Ybar, pyx = create_dataset(n_samples=10000)

    fig,axs = plt.subplots(1, 2, figsize=(12,4))
    for ax,macro,name in zip(axs.ravel(), [Xbar,Ybar], ['Xbar','Ybar']):
        for i in np.unique(macro):
            ax.scatter(X[macro==i], Y[macro==i], label=i, s=2, alpha=0.4)
        ax.legend()
        ax.set_title(f'Microvariables colored by {name}')
        ax.set_xlabel('X micro')
        ax.set_ylabel('Y micro')
    
    plt.savefig('/Users/imanwahle/Desktop/cfl/cfl/cluster_methods/param_tuning/toy_data_7_3')
    # plt.show()

def test_create_dataset_and_cde_results():
    dpath = '/Users/imanwahle/Desktop/cfl/cfl/cluster_methods/param_tuning/toy_data_7_3'
    create_dataset_and_cde_results(n_Xbar=7, n_Ybar=3, dataset_path=dpath)