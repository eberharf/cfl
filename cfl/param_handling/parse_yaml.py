import yaml

if __name__ == '__main__':

    stream = open("cfl/param_handling/yaml_prototype.yaml", 'r')
    dictionary = yaml.load(stream)
    print(dictionary)


    {'data_info': {'X_shape': '(100, 2)', 'Y_shape': '(100, 4)', 'Y_type': 'continuous'}, 
    'cde_params': {'n_epochs': 50, 'dense_units': [20], 'verbose': 1}, 
    'clusterer_params': {'x_model': 'KMeans', 'x_params': {'n_clusters': 2}, 'y_model': 'None'}}