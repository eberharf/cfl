
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

def compute_cond_prob(xlbls, ylbls):

    # handle noise clusters
    xlbls_pos = xlbls - np.min(xlbls)
    ylbls_pos = ylbls - np.min(ylbls)

    # compute P(y_lbl | x_lbl)
    P_XmYm = np.array([np.bincount(ylbls_pos.astype(int)[xlbls_pos==xlbl], 
        minlength=ylbls_pos.max()+1).astype(float) for xlbl in np.sort(np.unique(xlbls_pos))])
    P_XmYm = P_XmYm/P_XmYm.sum()
    P_Ym_given_Xm = P_XmYm/P_XmYm.sum(axis=1, keepdims=True)

    return P_Ym_given_Xm

def visualize_cond_prob(P_Ym_given_Xm, uxlbls, uylbls, fig_path=None):
    fig,ax = plt.subplots()
    im = ax.imshow(P_Ym_given_Xm, vmin=0, vmax=1)
    ax.set_xticks(range(len(uxlbls)))
    ax.set_yticks(range(len(uylbls)))
    ax.set_xticklabels(uxlbls)
    ax.set_yticklabels(uylbls)
    ax.set_xlabel('X macrostates')
    ax.set_ylabel('Y macrostates')
    ax.set_title('P(Ymacrostate | Xmacrostate)')
    fig.colorbar(im)
    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches='tight')
    plt.show()


def macro_cond_prob(exp_path, dataset='dataset_train'):
    xpath = os.path.join(exp_path, dataset, 'CauseClusterer_results.pickle')
    with open(xpath, 'rb') as f:
        xlbls = pickle.load(f)['x_lbls']
    ypath = os.path.join(exp_path, dataset, 'EffectClusterer_results.pickle')
    with open(ypath, 'rb') as f:
        ylbls = pickle.load(f)['y_lbls']
    P_Ym_given_Xm = compute_cond_prob(xlbls, ylbls)
    save_path = os.path.join(exp_path,dataset, 'macro_cond_prob')
    np.save(save_path, P_Ym_given_Xm)
    visualize_cond_prob(P_Ym_given_Xm, np.unique(xlbls), np.unique(ylbls), 
                        save_path)
    return P_Ym_given_Xm

