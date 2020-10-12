A Python implementation of Algorithms 1 and 2 from "Visual Causal Feature Learning", Krzysztof Chalupka, Frederick Eberhardt, Pietro Perona, UAI 2015.

Requirements:
Python 2.7.*
Numpy, Scipy
Theano, Pylearn2

Usage:
The script experiment_grating.py reproduces the GRATING experiment from the paper. The script learns the causal coarsening using Algorithm 1 and then proceeds to train the manipulator function using Algorithm 2. File data_binary_gratings.py generates the GRATINGS dataset. File ai_gratings.py contains a class that simulates an agent that implements behavior T (see article). We leave visualization of the results to the user; the script displays only the causal prediction and manipulation errors throughout training.

