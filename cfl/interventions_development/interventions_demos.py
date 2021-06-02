import cfl.interventions_development.interventions_prototype as IP
from sklearn.datasets import make_blobs
from visual_bars.generate_visual_bars_data import VisualBarsData as VB
import numpy as np
import matplotlib.pyplot as plt

X, y = make_blobs(n_samples=1000, centers=3, n_features=2, random_state=0)
IP.main(X, y, k_samples=30, eps=0.5, to_plot=True, series='blobs3_100HC')
IP.main(X, y, k_samples=100, eps=0.5, to_plot=True, series='blobs3_500HC')


# X, y = make_blobs(n_samples=10000, centers=10, n_features=2, random_state=0)
# IP.main(X, y, k_samples=30, eps=0.95, to_plot=True, series='blobs10_100HC')
# IP.main(X, y, k_samples=1000, eps=0.95, to_plot=True, series='blobs10_3000HC')


# vb_data = VB(n_samples=5000, noise_lvl=0.05, set_random_seed=0)
# X = vb_data.getImages()
# X_flat = np.reshape(X, (X.shape[0], np.product(X.shape[1:])))
# y = vb_data.getGroundTruth()
# mask = IP.main(X_flat, y, k_samples=500, eps=0.01, to_plot=False, 
#                series='vb_3000HC')
# n_clusters = 4
# n_examples = 20
# fig,ax = plt.subplots(n_clusters, n_examples, figsize=(n_examples*2,10))
# for i in range(n_clusters):
#     cluster_mask = np.where((y==i) & (mask==1))[0]
#     print(cluster_mask)
#     if len(cluster_mask) > 0:
#         idx = np.random.choice(cluster_mask, 
#                             n_examples, replace=False)
#         for j in range(n_examples):
#             ax[i,j].imshow(X[idx[j]])
#             ax[i,j].axis('off')
# plt.show()


