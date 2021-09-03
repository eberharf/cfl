import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import os

BASE_PATH = '/Users/imanwahle/Desktop/cfl/cfl/cluster_methods/param_tuning'

fig,ax = plt.subplots()
n_effect_gt_range = range(3,7)
cmap = get_cmap('YlOrBr')
colors = cmap(np.linspace(0.5,1,len(n_effect_gt_range)))
for i,ei in enumerate(n_effect_gt_range):
    z = np.load(f'toy_data_4_{ei}/effect_errors.npz') 
    n_effect_clusters_range,errs = z['n_effect_clusters_range'], z['errs']
    ax.plot(n_effect_clusters_range, errs, color=colors[i], label=f'GT#C={ei}', 
            marker='o', alpha=0.6)
    ax.set_xticks(n_effect_clusters_range)
    ax.legend()
    ax.set_xlabel('n_effect_clusters')
    ax.set_ylabel('predictive_error')
plt.savefig(os.path.join(BASE_PATH, 'overlaid_effect_errors'))
# plt.show()
