# Set-up instructions

## Short instructions: 

With Python 3.6-3.8: 

```
pip install cfl
```

## Long instructions: 


 We recommend installing `cfl` in a virtual environment to prevent unintended
 interactions between different packages. If you don't already have a virtual
 environment system, follow steps 1 and 2. Otherwise, skip to step 3. 

**1. Install Anaconda**

We use the `conda` environment management system system. 
You can install `conda`
[here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/). For
our purposes, either the full Anaconda distribution or the smaller Miniconda
should work fine.

**2. Create a conda environment**

With `conda` installed, open a terminal window and run the command: 

```
conda create -n cfl-env python=3.8
```

where `cfl-env` can be replaced with any name of your choice. 

This will create a fresh environment, named `cfl-env`, that contains the version
of Python we specified (`cfl` was developed with Python 3.8, so we're using that).

Then activate the environment: 

```
conda activate cfl-env
```

If no error messages result from this command, then you have successfully activated the new environment.


**3. Pip install `cfl`**

With your cfl virtual environment active, run the command: 

```
pip install cfl
```

The installation may take a few minutes, especially if `tensorflow` is not already
installed. 

To check that the installation was successful, open a Python interpreter (type
`python` into the terminal). Then, from within Python, run the command `import cfl` and check the version:

```
python
>>> import cfl
>>> cfl.__version__
```
The version number of `cfl` should print.
If this command executes with no errors, then you are now ready to use `cfl`!


### Troubleshooting 

1. "`No matching distribution found for tensorflow>=2.4.0`" 

Check that the version of Python you are using is supported by the current
version of Tensorflow
(see https://www.tensorflow.org/install). If not, upgrade/downgrade your Python
version to fit.