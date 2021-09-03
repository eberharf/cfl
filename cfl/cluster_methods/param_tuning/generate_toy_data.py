import numpy as np
import os
from cfl.experiment import Experiment
import matplotlib.pyplot as plt

def plot_dataset(X, Y, Xbar, Ybar, fig_path):
    fig,axs = plt.subplots(1, 2, figsize=(12,4))
    for ax,macro,name in zip(axs.ravel(), [Xbar,Ybar], ['Xbar','Ybar']):
        for i in np.unique(macro):
            ax.scatter(X[macro==i], Y[macro==i], label=i, s=2, alpha=0.8)
        ax.legend()
        ax.set_title(f'Microvariables colored by {name}')
        ax.set_xlabel('X micro')
        ax.set_ylabel('Y micro')

    plt.savefig(os.path.join(fig_path, 'data_dists'))
    plt.show()

def create_dataset(n_samples=1000, n_Xbar=4, n_Ybar=3, pyx=None):
    ''' This file generates a toy dataset to test cluster tuning. At the
        microvariable level, this dataset has a one-dimensional cause and
        one-dimensional effect. There are n_Xbar ground-truth cause macrostates,
        and n_Ybar ground-truth effect macrostates. The transition probabilities
        between cause and effect macrostates can be set using pyx. 
    '''

    if pyx is None:
        if n_Xbar==2 and n_Ybar==2:
            pyx = np.array([[0.0,1.0], 
                            [1.0,0.0]])
        if n_Xbar==3 and n_Ybar==2:
            pyx = np.array([[0.0,1.0], 
                            [1.0,0.0],
                            [0.5,0.5]])
        elif n_Xbar==4 and n_Ybar==3:
            # pyx = np.array([[0.8,0.1,0.1], 
            #                 [0.1,0.8,0.1], 
            #                 [0.1,0.1,0.8], 
            #                 [0.4,0.3,0.3]])
            pyx = np.array([[0.8,0.1,0.1], 
                            [0.1,0.8,0.1], 
                            [0.01,0.01,0.98], 
                            [0.1,0.3,0.6]])  
        elif n_Xbar==4 and n_Ybar==4:
            pyx = np.array([[0.8,0.05,0.1,0.05], 
                            [0.05,0.35,0.05,0.55], 
                            [0.01,0.01,0.01,0.97], 
                            [0.1,0.2,0.6,0.1]])                
        elif n_Xbar==4 and n_Ybar==5:
            pyx = np.array([[0.1, 0.3, 0.5, 0.07,0.03], 
                            [0.6, 0.1, 0.01,0.05,0.24], 
                            [0.01,0.16,0.01,0.8, 0.02], 
                            [0.05,0.05,0.1, 0.2, 0.6]])
        elif n_Xbar==4 and n_Ybar==6:
            pyx = np.array([[0.1, 0.8, 0.07,0.01,0.01,0.01], 
                            [0.1, 0.1, 0.01,0.05,0.73,0.01], 
                            [0.01,0.16,0.01,0.8, 0.01,0.01], 
                            [0.09,0.01,0.1, 0.2, 0.01, 0.59]])
        elif n_Xbar==5 and n_Ybar==3:
            pyx = np.array([[0.98,0.01,0.01],
                            [0.6,0.3,0.1], 
                            [0.1,0.8,0.1], 
                            [0.01,0.01,0.98], 
                            [0.1,0.3,0.6]])
        elif n_Xbar==6 and n_Ybar==3:
            pyx = np.array([[0.98,0.01,0.01],
                            [0.6,0.3,0.1], 
                            [0.2,0.75,0.05], 
                            [0.05,0.75,0.2], 
                            [0.01,0.01,0.98], 
                            [0.1,0.3,0.6]])
        elif n_Xbar==7 and n_Ybar==3:
            # pyx = np.array([[0.1,0.1,0.8], 
            #                 [0.1,0.8,0.1], 
            #                 [0.8,0.1,0.1], 
            #                 [0.6,0.2,0.2],
            #                 [0.2,0.6,0.2],
            #                 [0.2,0.2,0.6],
            #                 [0.5,0.4,0.1]])
            # pyx = np.array([[0.01,0.01,0.98], 
            #                 [0.24,0.75,0.01], 
            #                 [0.98,0.01,0.01], 
            #                 [0.6,0.2,0.2],
            #                 [0.02,0.6,0.38],
            #                 [0.1,0.2,0.7],
            #                 [0.15,0.7,0.15]])
            pyx = np.array([[0.98,0.01,0.01],
                            [0.6,0.3,0.1], 
                            [0.24,0.75,0.01], 
                            [0.15,0.7,0.15], 
                            [0.01,0.01,0.98], 
                            [0.25,0.25,0.5],
                            [0.01,0.39,0.6]])
        elif n_Xbar==7 and n_Ybar==7:
            pyx = np.array([[1.0,0.0,0.0,0.0,0.0,0.0,0.0], 
                            [0.0,1.0,0.0,0.0,0.0,0.0,0.0], 
                            [0.0,0.0,1.0,0.0,0.0,0.0,0.0], 
                            [0.0,0.0,0.0,1.0,0.0,0.0,0.0],
                            [0.0,0.0,0.0,0.0,1.0,0.0,0.0],
                            [0.0,0.0,0.0,0.0,0.0,1.0,0.0],
                            [0.0,0.0,0.0,0.0,0.0,0.0,1.0]])
        elif n_Xbar==10 and n_Ybar==3:
            pyx = np.array([[0.0,0.9,0.1], 
                            [0.1,0.8,0.1], 
                            [0.2,0.7,0.1], 
                            [0.3,0.6,0.1],
                            [0.4,0.5,0.1],
                            [0.5,0.3,0.2],
                            [0.6,0.2,0.2],
                            [0.7,0.1,0.2],
                            [0.8,0.0,0.2],
                            [0.9,0.0,0.1]])
        else:
            print('Default pyx only available for default n_Xbar, n_Ybar values.')
            return

    X = np.zeros((n_samples,1))
    Y = np.zeros((n_samples,1))
    Xbar = np.zeros((n_samples,), dtype=int)
    Ybar = np.zeros((n_samples,), dtype=int)
    X_interval = 1 / n_Xbar
    Y_interval = 1 / n_Ybar
    for si in range(n_samples):
        # randomly select cause macrostate
        Xbar[si] = np.random.choice(n_Xbar)

        # generate cause microvariable value
        X[si] = (np.random.random_sample() / n_Xbar) + (Xbar[si] / n_Xbar)

        # generate effect macrostate from pyx table
        Ybar[si] = np.random.choice(n_Ybar, p=pyx[Xbar[si]])

        # generate effect microvariable value
        Y[si] = (np.random.random_sample() / n_Ybar) + (Ybar[si] / n_Ybar)

    params = {'n_Xbar' : n_Xbar, 'n_Ybar' : n_Ybar, 'pyx' : pyx}
    return X, Y, Xbar, Ybar, params



def create_dataset_and_cde_results(n_samples=1000, n_Xbar=4, n_Ybar=3, pyx=None,
    dataset_path='/Users/imanwahle/Desktop/cfl/cfl/cluster_methods/param_tuning/toy_data'):
    
    # make directory for this dataset and results
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    else:
        print('choose a different dataset_path')
        return
    
    # generate data
    X,Y,Xbar,Ybar,params = create_dataset(n_samples=n_samples, 
                                       n_Xbar=n_Xbar, 
                                       n_Ybar=n_Ybar, 
                                       pyx=pyx)
    
    # plot data
    plot_dataset(X, Y, Xbar, Ybar, fig_path=dataset_path)

    # save data
    np.savez(os.path.join(dataset_path, 'data'),
             X=X,
             Y=Y,
             Xbar=Xbar,
             Ybar=Ybar,
             params=params)

    # run CFL
    data_info = {   'X_dims' : X.shape,
                    'Y_dims' : Y.shape,
                    'Y_type' : 'continuous'
                }
    cde_params = {  # model creation parameters
                    'dense_units' : [300, 300, 300, data_info['Y_dims'][1]], 
                    'activations' : ['relu', 'relu', 'relu', 'linear'],
                    'dropouts'    : [0, 0, 0, 0],

                     # parameters for training
                    'batch_size'  : 128,
                    'n_epochs'    : 1200,
                    'optimizer'   : 'adam',
                    'opt_config'  : {'lr' : 1e-4},
                    'loss'        : 'mean_squared_error',
                    'best'        : True,

                    # amount of output to print
                    'verbose'     : 1, 
                    'show_plot'   : True,
            }
    block_names = ['CondExpMod']
    block_params = [cde_params]

    my_exp = Experiment(X_train=X, 
                        Y_train=Y, 
                        data_info=data_info, 
                        block_names=block_names, 
                        block_params=block_params, 
                        results_path=dataset_path)

    results = my_exp.train()
    plot_pyx_dist(results['CDE']['pyx'], fig_path=dataset_path)


    
def plot_pyx_dist(pyx, fig_path):
    fig,ax = plt.subplots()
    ax.hist(pyx, bins=100)
    ax.set_xlim((0,1))
    ax.set_xlabel('E[P(Y|X=x)]')
    ax.set_ylabel('Count')
    ax.set_title('E[P(Y|X=x)] Distribution')
    if not os.path.exists(os.path.join(fig_path, 'figures')):
        os.mkdir(os.path.join(fig_path, 'figures'))
    plt.savefig(os.path.join(fig_path, 'figures/pyx_dist'))