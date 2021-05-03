# Set-up instructions

## Short instructions: 

```
pip install cfl
```


## Long instructions: 


 We recommend installing `cfl` in a virtual environment to prevent unintended
 interactions between different packages. If don't already have a virtual
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

This will create a fresh environment, named `cfl-env` that contains the version
of Python we specify (`cfl` was developed with Python 3.8)

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

To check that the installation was successful, open a Python interpreter (type
`python`). Then, from within Python, run the command `import cfl` and check the version:

```
python
>>> import cfl
>>> cfl.__version__
```
The version number of `cfl` should print.
If this command executes with no errors, then you are now ready to use `cfl`!


**Optional: Add the cfl environment to the Jupyter notebook kernel**

For running Jupyter notebooks from within a `conda` virtual environment: 

In order to also be able to run CFL inside of a Jupyter notebook, we need to add `cfl-env` (the `conda` environment which contains the dependencies for `cfl`) to the iPython kernel. This will allow Jupyter notebooks to access the packages we installed for `cfl`. Add `cfl-env` to the Jupyter kernel by running the following command:

```
 ipython kernel install --name cfl-env --user
```

If this step is sucessful, it will generate the message

```
Installed kernelspec cfl-env in C:/some/directory/
```

You can also test the success of this step by downloading a notebook from the
`examples` folder on GitHub, `cd`ing into the folder containing that notebook, and starting a Jupyter Notebook server:

```
jupyter notebook
```

Open one the notebook. Select `cfl-env` as the kernel if prompted. Run the first few code blocks in the notebook. If the import statements in the notebook can be run without errors, then setup has been successful!