import cfl.interventions.interventions_prototype as IP
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=1000, centers=3, n_features=2, random_state=0)
IP.main(X, y, k_samples=100, eps=0.5, to_plot=True, series='blobs3_100HC')
IP.main(X, y, k_samples=500, eps=0.5, to_plot=True, series='blobs3_500HC')


X, y = make_blobs(n_samples=10000, centers=10, n_features=2, random_state=0)
IP.main(X, y, k_samples=100, eps=0.95, to_plot=True, series='blobs10_100HC')
IP.main(X, y, k_samples=1000, eps=0.95, to_plot=True, series='blobs10_1000HC')
IP.main(X, y, k_samples=3000, eps=0.95, to_plot=True, series='blobs10_1000HC')

