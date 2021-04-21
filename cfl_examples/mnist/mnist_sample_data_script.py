''' a little script that generates our mnist causal set up''' 

from tensorflow.keras.datasets import mnist
import numpy as np

import random


# Load causal (X) data
(X_train, Y_train), (_, _) = mnist.load_data()

# Keep only data 1-9 
X_train = X_train[Y_train != 0]

# We convert the image values from [0, 255] to [0, 1] to speed up training
MAX_RGB = 255
X_train = np.true_divide(X_train, MAX_RGB)




# Generate target (Y) data. Here 'a' and 'b' represent alpha and beta
targets = ['alpha', 'beta']
class_A = [1, 2, 3]
class_B = [4, 5, 6]
class_C = [7, 8, 9]
distributions = [[0.95, 0.05], [0.05, 0.95], [0.5, 0.5]]

'''
Returns a distribution depending on the causal class of the input value
'''
def get_distribution(val):
    if val in class_A:
        return distributions[0]
    elif val in class_B:
        return distributions[1]
    else:
        return distributions[2]

'''
For every data point given, draw from its distribution and return an array of target variables
'''
def generate_target(data):
    target = []
    for val in data:
        target += (random.choices(targets, get_distribution(val)))
    return np.array(target)

