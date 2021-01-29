# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import matplotlib.pyplot as plt
import joblib

# I keep having import issues with plotly so now I will just install it from whatever environment I'm currently in
# import sys
# !conda install --yes --prefix {sys.prefix} plotly
import plotly.graph_objects as go

from cfl.cluster_methods.kmeans import KMeans
from cfl.experiment import Experiment
from visual_bars import generate_visual_bars_data as vbd
from cfl.util.data_processing import one_hot_encode
from cfl.dataset import Dataset
from cfl.visualization_methods import clustering_to_sankey as sk


# %%
# create a visual bars data set

n_samples = 1000
noise_lvl = 0.03
im_shape = (10, 10)
random_seed = 143
print('Generating a visual bars dataset with {} samples at noise level {}'.format(n_samples, noise_lvl))

vb_data = vbd.VisualBarsData(n_samples=n_samples, im_shape = im_shape, noise_lvl=noise_lvl, set_random_seed=random_seed)

ims = vb_data.getImages()
y = vb_data.getTarget()

# format data
x = np.expand_dims(ims, -1)

# y = one_hot_encode(y, unique_labels=[0,1])
y = np.expand_dims(y, -1)


# %%
y.shape
y[:10]


# %%
data_info = {'X_dims': x.shape,
             'Y_dims': y.shape,
             'Y_type': 'categorical'}

# # paragmeters for CDE
CNN_params = { # parameters for model creation
                    'filters'         : [32, 64],
                    'input_shape'     : (10, 10, 1),
                    'kernel_size'     : [(3, 3)] *2,
                    'pool_size'       : [(2, 2)] *2,
                    'padding'         : ['same'] *2,
                    'conv_activation' : ['softmax']*2,
                    'dense_units'     : 64,
                    'dense_activation' : 'softmax',
                    'output_activation': None,

                    # parameters for training
                    'batch_size'  : 32,
                    'n_epochs'    : 100,
                    'optimizer'   : 'adam',
                    'opt_config'  : {},
                    'verbose'     : 2,
                    'weights_path': None,
                    'loss'        : 'mean_squared_error',
                    'show_plot'   : True,
                    'standardize' : False,
                    'best'        : True,
                    }


block_names = ['CondExpCNN']
block_params = [CNN_params]

# save_path = '/Users/imanwahle/Desktop/cfl/examples/exp_results'
save_path = 'C:/Users/yumen/Documents/Schmidt Academy/cfl/cnn_test'
my_exp = Experiment(X_train=x, Y_train=y, data_info=data_info, block_names=block_names, block_params=block_params, blocks=None, results_path=save_path)


# %%
my_exp.blocks[0].model.summary()


# %%
cde_results_dict = joblib.load(os.path.join('/', save_path, 'experiment0032/dataset_train/CondExpCNN_results.pickle'))

# 29 is a good one, 32 is about the same
pyx = cde_results_dict['pyx']
##### TRY THE CONVOLUTIONAL NEURAL NET


# %%
truth = vb_data.getGroundTruth()

#choose a thousand random samples from the pyx results
plot_idx = np.random.choice(pyx.shape[0], 1000, replace=False)

# plot them
plt.scatter(range(1000), pyx[plot_idx,0], c=truth[plot_idx])
plt.ylabel("Probability that target = 1")
plt.xlabel("Sample")
plt.show()

for i in range(4):
    print('Average prediction for x-class {}: {:.2f}'.format(i, np.mean(pyx[truth==i,0])))


# %%
cluster_params = {'n_Xclusters': 4, 'n_Yclusters': 2}
kmeans_obj = KMeans('Kmeans', data_info, cluster_params, random_state=143)
res = kmeans_obj.train(Dataset(x, y), cde_results_dict)


# %%
from sklearn.metrics import adjusted_rand_score

adjusted_rand_score(truth, res['x_lbls'])


# %%
# # this graph is the reverse of the above graph
# plt.scatter(range(1000), pyx[plot_idx,0], c=truth[plot_idx])
# plt.ylabel("Probability that target = 0")
# plt.xlabel("Sample")
# plt.show()

# for i in range(4):
#     print('Average prediction for x-class {}: {:.2f}'.format(i, np.mean(pyx[truth==i,0])))


# %%
# show all the probabilities that y=1
plt.hist(pyx[:, 1], bins=25)
plt.show()

#same info as above, in a hist


# %%
# create a number of Kmeans objects with different Ks
k_range = range(2, 8)

kmeans_l = []
for n_clusters in k_range:
    params = {'n_Xclusters': n_clusters, 'n_Yclusters': 2}
    kmeans_obj = KMeans('Kmeans', data_info, params, random_state=143)
    kmeans_l.append(kmeans_obj)

data = Dataset(x, y)

# trained all the data on the kmeanss
x_lbls_L = []
for kmeans_obj in kmeans_l:
  cluster_results = kmeans_obj.train(data, cde_results_dict)
  x_lbls_L.append(cluster_results['x_lbls'])


# %%
link, label = sk.convert_lbls_to_sankey_nodes(x_lbls_L)
# plot
fig = go.Figure(data=
          [go.Sankey(node = dict(pad = 15, thickness=20, label = label, color =  "blue"),
                     link = link)])

fig.update_layout(title_text="Visual Bars Clustering, no noise, with 2 to 7 Clusters", font_size=10)
fig.show()


# %%
# x_lbls_L[2][plot_idx].shape


# %%
truth = vb_data.getGroundTruth()


# plot them
markers = ["." , "," , "o" , "v" ]
colors = ['g','b', 'm', 'y']

#choose a thousand random samples from the pyx results
plot_idx = np.random.choice(pyx.shape[0], 1000, replace=False)

# for each sample
for i in range(1000):
    pi = pyx[plot_idx[i]] # plot probability
    mi = markers[x_lbls_L[2][plot_idx[i]]] #marker based on cluster assignment
    ci = colors[truth[plot_idx[i]]] #color based on ground truth class

    plt.scatter(i,pi, marker=mi, color=ci)
plt.show()


# %%
from cfl.visualization_methods import general_vis as vis

vis.view_class_examples(ims, im_shape, 10, x_lbls_L[0])

# %% [markdown]
#

# %%
vis.view_class_examples(ims, im_shape, 10, x_lbls_L[1])


# %%
vis.view_class_examples(ims, im_shape, 10, x_lbls_L[2])


# %%
# vis.view_class_examples(ims, im_shape, 10, x_lbls_L[3])


# %%
# vis.view_class_examples(ims, im_shape, 10, x_lbls_L[4])


# %%
# vis.view_class_examples(ims, im_shape, 10, x_lbls_L[5])


# %%
# truth = vb_data.getGroundTruth()


# # plot them
# markers = ["." , "," , "o" , "v" , "^" , "<", ">"]
# colors = ['c','m', 'y', 'k', 'r', 'g','b']


# #choose a thousand random samples from the pyx results
# plot_idx = np.random.choice(pyx.shape[0], 1000, replace=False)

# # for each sample
# fig = plt.figure(figsize=(10, 10))
# for i in range(1000):
#     pi = pyx[plot_idx[i],1] # plot probability
#     mi = markers[truth[plot_idx[i]]] #marker based on ground truth
#     ci = colors[x_lbls_L[5][plot_idx[i]]] #color based on cluster assignment

#     plt.scatter(i,pi, marker=mi, color=ci)
# plt.show()


