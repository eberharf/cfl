import numpy as np
import sys
# sys.path.append('/home/ec2-user/cfl')
from cfl.clustering.Y_given_Xmacro import _continuous_Y
from time import time

n_samples = 13000
Y_data = np.random.randint(low=0,high=100,size=(n_samples,150*150*2))
x_lbls = np.random.randint(low=0,high=4,size=(n_samples,))


st = time()
print(f'Running _continuous_Y for Y_data of shape {Y_data.shape} and x_lbls of shape {x_lbls.shape}')
_continuous_Y(Y_data, x_lbls)
print(f'Total run time: {time()-st}')