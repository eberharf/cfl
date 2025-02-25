# Install CFL

## Short instructions: 

With Python 3.7-3.8: 

```
pip install cfl
```

## Disclaimer:

The existing implementation of CFL uses methods and functions that are deprecated and 
only exist in past versions of certain libraries.

To ensure functionality of CFL, follow the long instructions below (with or without Anaconda).

## Long instructions (Anaconda): 


 We recommend installing `cfl` in a virtual environment to prevent unintended
 interactions between different packages. If you don't already have a virtual
 environment system, follow steps 1 and 2. Otherwise, skip to step 3. 

**1. Install Anaconda**

We recommend using the `conda` environment management system system. You can
install `conda`
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
of Python we specified (`cfl` was developed with Python 3.8, so we're using
that).

Then activate the environment: 

```
conda activate cfl-env
```

If no error messages result from this command, then you have successfully
activated the new environment.

**3. Pip install `cfl`**

With your cfl virtual environment active, run the command: 

```
pip install cfl
```

The installation may take a few minutes, especially if `tensorflow` is not
already installed. 

To check that the installation was successful, open a Python interpreter (type
`python` into the terminal). Then, from within Python, run the command `import
cfl` and check the version:

```
python
>>> import cfl
>>> cfl.__version__
```
The version number of `cfl` should print. If this command executes with no
errors, then you are now ready to use `cfl`!

## Long instructions (Native Python):

**1. Creating and activating virtual environment**

Create the virtual environment with the command:

```
python -m venv cfl-env
```

and activate the virtual environment with the following command (depending on OS):

```
# for MacOS and Linux:
source cfl-env/bin/activate 

# for Windows:
cfl-env\Scripts\activate
```

This will create and activate an environment with the version of Python associated 
with the `python` terminal command

**2. Install `cfl` and other dependencies:**

The existing implementation of CFL uses methods and functions that are deprecated and
only exist in past versions of certain libraries.

To ensure the correct versions of libraries are used and full functionality of CFL,
run the following commands in the virtual environment:

```
pip install cfl
pip install optuna
pip install tensorflow==2.15.0
pip install keras==2.15.0
```

The installation may take a few minutes, especially if `tensorflow` is not
already installed. 

To check that the installation was successful, open a Python interpreter (type
`python` into the terminal). Then, from within Python, run the command `import
cfl` and check the version:

```
python
>>> import cfl
>>> cfl.__version__
```
The version number of `cfl` should print. If this command executes with no
errors, then you are now ready to use `cfl`!

<!-- ### Troubleshooting 

1. "`No matching distribution found for tensorflow>=2.4.0`" 

Check that the version of Python you are using is supported by the current
version of Tensorflow (see https://www.tensorflow.org/install). If not,
upgrade/downgrade your Python version to fit. -->