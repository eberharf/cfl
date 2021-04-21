## Set-up instructions


### Clone the repository

First we will download a local copy of the `cfl` repository. Before starting this step, make sure you have [Git](https://git-scm.com/) installed on your computer.

Open a terminal window. Navigate to the location where you would like the repository to be located, and clone the repository onto your computer:

```
git clone https://github.com/eberharf/cfl.git
```

You should now see a folder named `cfl` on your computer that contains all of the code for this project.


### Install Dependencies

This package was developed with Python 3.7.4 and requires a compatible version of Python.

We recommend following the instructions below to create a `conda` virtual environment with all the required dependencies. However, if you wish, you may also manually install the required packages (full list of requirements in [`requirements.yml`](https://github.com/eberharf/cfl/blob/master/requirements.yml)).


#### Install Anaconda 

If you do not already have `conda` installed, see [this documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) for information on how to install it. For our purposes, either the full Anaconda distribution or the smaller Miniconda should work fine.


#### Create a conda environment

These instructions are for use with conda 4.8.4 or a compatible version.

We will create a conda virtual environment that contains the dependencies for `cfl` from the file `requirements.yml`. This will create a fresh environment that contains no packages aside from those purposefully installed. Using isolated environments can help prevent unintended interactions between packages or versions.

Use the terminal to navigate into the root directory of `cfl` and run the command:
```
conda env create -f requirements.yml
```

This command uses the information in the `requirements.yml` file to generate a new `coonda` environment. It may take a few minutes to execute. You should see progress information output onto the screen as packages are successfully downloaded. At the end, you should see the output:

```
#
# To activate this environment, use
#
#     $ conda activate cfl-env
#
# To deactivate an active environment, use
#
#     $ conda deactivate
```

If you see this output, the new environment, named `cfl-env` has been successfully created.


To use `cfl-env`, we must activate it:

```
conda activate cfl-env
```

If no error messages result from this command, then you have successfully activated the new environment.


### Add `cfl` to path
Before trying to run any code, add the path to the respository to your computer's `PYTHONPATH` variable. This will allow you to easily import the `cfl` package to use in any file, regardless of the location of that file.

To do this, consult the Internet for system-specific instructions on how to [set your `PYTHONPATH` variable](https://bic-berkeley.github.io/psych-214-fall-2016/using_pythonpath.html), and add the path to the repository's root directory to the `PYTHONPATH`.

You must restart your session after modifying the `PYTHONPATH` to see the change take effect.

You should now be ready to run `cfl`!

Check that your installation has been successful by opening a new terminal window. Do _not_ navigate to the `cfl` directory. Make sure that the cfl `conda` environment is active, and open a Python interpreter (type `python`). Then, from within Python, run the command `import cfl`:

```
python
>>> import `cfl`
>>>
```

If this command executes with no errors, then you are now ready to use `cfl`.

#### Alternative: Temporarily adding directory to path 

If you have difficulty permanently modifying the `PYTHONPATH` variable, use the following workaround.

Add this block of code to the top of any file where you want to use `cfl`:

```
import sys
sys.path.append('/Users/path/to/cfl')
```

where `path/to/cfl` is the path to the root directory of the repository.


### Add the cfl environment to the Jupyter notebook kernel

Last step! In order to also be able to run CFL inside of a Jupyter notebook, we need to add `cfl-env` (the virtual environment which contains the dependencies for `cfl`) to the iPython kernel. This will allow Jupyter notebooks to access the packages we installed for `cfl`. Add `cfl-env` to the Jupyter kernel by running the following command:

```
 ipython kernel install --name cfl-env --user
```

If this step is sucessful, it will generate the message

```
Installed kernelspec cfl-env in [C:/some/directory/]
```

You can also test the success of this step by `cd`ing into the `examples` folder and opening one of the Jupyter notebooks. Select `cfl-env` as the kernel if prompted. If the import statements in the notebook can all be run without errors, then the setup has been successful!