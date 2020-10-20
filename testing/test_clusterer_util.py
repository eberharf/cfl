import tensorflow as tf 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans as sKMeans

#testing module 
from hypothesis import given
from hypothesis.strategies import integers, decimals, tuples

#local modules 
import generate_visual_bars_data as g
import cfl 
from cfl.cluster_methods.clusterer_util import getYs
from cfl.cluster_methods.kmeans_helper import cond_prob_of_Y

#try out diff visual bars data sizes, shapes, noise levels, random seeds 
# training epochs? 

# and make sure new and old files equal each other 



@given(integers(min_value=0, max_value=10000), tuples(integers(min_value=0, max_value=100), integers(min_value=0, max_value=100)), \
    decimals(min_value=0, max_value=1), integers(), integers(min_value=0, max_value=10000)) 
def test_things(n_images, img_shape, noise_lvl, random_seed, training_epochs): 
    # generate and reshape data to pass into neural net
    n_images =100 
    img_shape = (10, 10)
    noise_lvl = 0.03
    random_seed = 143 

    training_epochs = 100 

    v = g.VisualBarsData(n_images, img_shape, noise_lvl, random_seed)
    X = v.getImages()
    Y = v.getTarget()

    Y = np.expand_dims(Y, -1)
    X = np.reshape(X, (X.shape[0], X.shape[1]*X.shape[2])) 
    data_info = {'X_dims': X.shape, 'Y_dims': Y.shape} 

    #create training/testing sets 
    Xtr, Xts, Ytr, Yts, Itr, Its= train_test_split(X, Y, range(X.shape[0]), shuffle=True, train_size=0.85, random_state=random_seed)


    #create denisty estimator 
    density_estimator = cfl.density_estimation_methods.condExp.CondExp(data_info, \
    {'batch_size': 128, 'lr': 1e-3, 'optimizer': tf.keras.optimizers.Adam(lr=1e-3), 'n_epochs': training_epochs, 'test_every': 10, 'save_every': 10, 'verbose': False})

    saver = cfl.saver.Saver("whatver")
    saver.set_save_mode("train")
    # train density estimator 
    density_estimator.train(Xtr, Ytr, Xts, Yts, saver)
    # predict probability distribution 
    pyx = density_estimator.predict(X)

    #create clusterer object 
    cluster_params = {'n_Xclusters':4, 'n_Yclusters':2}

    #(this is just running the code inside kmeans outside of it) (which is not robust but will do for now)
    xkmeans = sKMeans(cluster_params['n_Xclusters'])
    x_lbls = xkmeans.fit_predict(pyx)  


    y_distributionOLD = getYs(Y, x_lbls) #y_distribution = P(y|Xclass)
    y_distributionNEW = cond_prob_of_Y(Y, x_lbls)

    assert y_distributionOLD == y_distributionNEW
