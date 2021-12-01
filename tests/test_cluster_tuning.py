


from cfl.clustering.effect_clusterer import EffectClusterer
import numpy as np
from cfl.clustering import CauseClusterer
from cfl.dataset import Dataset

n_samples,nx,ny = 1000,4,3
X = np.random.uniform(size=(n_samples, nx))
Y = np.random.uniform(size=(n_samples,ny))
dataset = Dataset(X, Y)
pyx = np.random.uniform(size=(n_samples,ny))
cause_prev_results = {'pyx' : pyx}
effect_prev_results = {'x_lbls' : np.random.randint(low=0, high=3, size=(n_samples,))}
data_info = {   'X_dims'    : (1000,4), 
                'Y_dims'    : (1000,3), 
                'Y_type'    : 'continuous'}


def test_cause_kmeans_no_tuning():
    params = {  'model'      : 'KMeans', 
                'n_clusters' : 3,
                'tune'      : False}
    clusterer = CauseClusterer(data_info, params)
    print(clusterer.train(dataset, cause_prev_results))
    print(clusterer.params)

def test_cause_kmeans_tuning():
    params = {  'model'      : ['KMeans'], 
                'n_clusters' : range(2,4),
                'tune'      : True}
    clusterer = CauseClusterer(data_info, params)
    print(clusterer.train(dataset, cause_prev_results))
    print(clusterer.params)  

def test_cause_dbscan_tuning():
    params = {  'model'      : ['DBSCAN'], 
                'eps' : [0.001,0.01,0.1,1],
                'min_samples' : range(3,5),
                'tune'      : True}
    clusterer = CauseClusterer(data_info, params)
    print(clusterer.train(dataset, cause_prev_results))
    print(clusterer.params) 


def test_effect_dbscan_tuning():
    params = {  'model'      : ['DBSCAN'], 
                'eps' : [0.01,0.1,1],
                'min_samples' : range(3,5),
                'tune'      : True}
    clusterer = EffectClusterer(data_info, params)
    print(clusterer.train(dataset, effect_prev_results))
    print(clusterer.params) 