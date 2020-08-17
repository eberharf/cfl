# convert the el nino pkl file to csv files

import numpy
import joblib
import os

if __name__ == "__main__":
    l = joblib.load('elnino_data.pkl') #l = a list of 2 numpy arrays and a dict
    x, y, coords = l

    if not os.path.exists('x.csv'):
        numpy.savetxt('x.csv', x, delimiter=',') #x data file

    if not os.path.exists('y.csv'):
        numpy.savetxt('y.csv', y, delimiter=',') #y data file

    if not os.path.exists('coords.csv'):
        with open('coords.csv', 'w') as f:  #coords data file
            for key in coords.keys():
                f.write("%s,%s\n" % (key,coords[key]))
