import io
import cProfile
import pstats

import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

from cfl.cluster_methods.snn_helper import SNN as espin_SNN
from cfl.cluster_methods.snn_vectorized import SNN as vector_SNN


# create some of the data from the paper
# blobs
n_samples = 5000
random_state = 170
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)

x, y = blobs

# normalize dataset for easier parameter selection
# ^TODO: note this, add it into pipeline
X = StandardScaler().fit_transform(x)

# algorithms
snn = vector_SNN(neighbor_num=20, min_shared_neighbor_proportion=0.5, eps=0.5)


# profile the snn fitting
pr = cProfile.Profile()
pr.enable()

snn.fit(X, num=2)

pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
ps.print_stats()


with open('test_big.txt', 'w+') as f:

    f.write(s.getvalue())