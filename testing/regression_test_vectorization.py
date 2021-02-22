'''this code is designed to test that the new vectorized form of the SNN code
produces the same results as the original version (it does)

adapted from
https://github.com/albert-espin/snn-clustering/blob/master/SNN/main.py'''


import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

from cfl.cluster_methods.snn_helper import SNN as espin_SNN
from cfl.cluster_methods.snn_vectorized import SNN as vector_SNN

print("imports done")

def main():

    """Main function"""

    # seed for reproducibility of results
    np.random.seed(0)

    # simple synthetic data
    n_samples = 1500
    circles = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.1)
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
    no_structure = np.random.rand(n_samples, 2), None

    print("simple data created")

    # anisotropic distributed data
    random_state = 170
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)

    print("ansio data done")

    # blobs with varied variances
    varied = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5],  random_state=random_state)

    # blobs with very different densities
    diff_density_blobs = datasets.make_blobs(n_samples=[n_samples//2, n_samples//5, n_samples//4, n_samples//6, n_samples//5, n_samples//3], cluster_std=[.1, .5, .2, .3, 2, 6], random_state=6)
    blobs0 = datasets.make_blobs(n_samples=n_samples, cluster_std=1., random_state=1)
    blobs1 = datasets.make_blobs(n_samples=n_samples, cluster_std=0.2, random_state=2)
    blobs2 = datasets.make_blobs(n_samples=n_samples//3, cluster_std=12, random_state=1)
    diff_density_blobs1 = (np.concatenate((blobs0[0], blobs1[0], blobs2[0])), np.concatenate((blobs0[1], blobs1[1]+3, blobs2[1]+3)))

    print('varied density data done')

    # real datasets
    iris = datasets.load_iris(return_X_y=True)
    faces = datasets.fetch_lfw_people(min_faces_per_person=70, resize=0.4, return_X_y=True)
    breast = datasets.load_breast_cancer(return_X_y=True)

    print('real data loaded')

    default_base = {'eps': .3,
                    'n_neighbors': 20,
                    'min_shared_neighbor_proportion': 0.5
                   }


    test_datasets = [('Circles', circles, {}),
        ('Circles (noisy)', noisy_circles, {'eps': .15}),
        ('Moons', noisy_moons, {}),
        ('Varied', varied, {'eps': .18}),
        ('Ansiotropic', aniso, {'eps': .15}),
        ('Blobs', blobs, {}),
        ('Square', no_structure, {}),

        ('Different density', diff_density_blobs, {'eps': .15}),
        ('Different density (II)', diff_density_blobs1, {'eps': .15}),

        ("Iris", iris, {'eps': .8}),
        ('Breast cancer', breast, {'eps': 2, 'n_neighbors': 55, 'min_shared_neighbor_proportion': 0.5}),
        ('Faces', faces, {'eps': 25, 'n_neighbors': 10, 'min_shared_neighbor_proportion': 0.5})
    ]

    #     plt.figure(num='Comparison of clustering algorithms ({})'.format(i+1), figsize=(40, 30))
    #     plot_num = 1

    counter = 0
    for i_dataset, (data_name, dataset, algo_params) in enumerate(test_datasets):
        print('moving onto dataset {}'.format(data_name))
        # update parameters with dataset-specific values
        params = default_base.copy()
        params.update(algo_params)

        X, y = dataset

        # normalize dataset for easier parameter selection
        X = StandardScaler().fit_transform(X)

        # algorithms
        esp_snn = espin_SNN(neighbor_num=params['n_neighbors'], min_shared_neighbor_proportion=params['min_shared_neighbor_proportion'])
        vec_snn = vector_SNN(neighbor_num=params['n_neighbors'], min_shared_neighbor_proportion=params['min_shared_neighbor_proportion'])

        clustering_algorithms = (
            ('esp_SNN', esp_snn),
            ('vec_SNN', vec_snn)

        )

        warnings.simplefilter("ignore")

        results = []
        for name, algorithm in clustering_algorithms:

            algorithm.fit(X, counter)
            print('finished fitting {}'.format(algorithm))

            y_pred = algorithm.labels_.astype(np.int)
            results.append(y_pred)

            # evaluate the results
            mutual_info = None
            rand_index = None
            calinski_score = None
            if len(np.unique(y_pred)) > 1 and len(np.unique(y)) > 1:
                mutual_info = adjusted_mutual_info_score(y, y_pred, average_method='arithmetic')
                rand_index = adjusted_rand_score(y, y_pred)
                calinski_score = calinski_harabasz_score(X, y_pred)

        og_mat = np.load('d_mat_og_' + str(counter)+ '.npy')
        new_mat = np.load('d_mat_vect_' + str(counter)+ '.npy')
        assert(np.all(og_mat == new_mat)), "Distance matrices for {} not the same".format(data_name)

        assert(np.all(results[0] == results[1])), "Predictions are not the same for {}".format(data_name)

        results.clear()
        counter += 1

if __name__ == "__main__":
    main()