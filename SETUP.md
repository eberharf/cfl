## Set-up instructions

### Clone the repository

Git clone this repository onto your computer:
```
git clone https://github.com/eberharf/cfl.git
```

### Install Dependencies

Any version of Python compatible with 3.7.4
View full requirements (with the version we used) in the `requirements.yml` file.
You may either manually install the required packages or follow the instructions below to generate a conda virtual environment with all the required dependencies from file.


#### Create a conda environment
To create a conda virtual environment with the required dependencies for `cfl` from the file `requirements.yml`, navigate into the root directory of `cfl` and run the command:
```
conda env create -f requirements.yml
```

Then activate the newly created environment:
```
conda activate cfl-env
```

(These instructions use Anaconda 4.8.4)

### Add the cfl-env environment to the Jupyter notebook kernel

In order to be able to access the
```
 ipython kernel install --name cfl-env --user
```

### Add `cfl` to path
Before running this code, add the path to the location of the respository to your **`PYTHONPATH`** variable. This will allow you to easily import the `cfl` package into any other file (regardless of the location of that file) using the statement `import cfl`.

For example, on my windows machine I would add
```
C:\Users\Jenna\Documents\Schmidt\cfl
```
to the PYTHONPATH variable in my system environment variables.

On mac, open ~/.bash_profile with a text editor (i.e. `vim ~/.bash_profile` from terminal), and add the following lines to the end of the file:

```
PYTHONPATH=/path/to/cfl
export PYTHONPATH
```

Consult Google for system-specific instructions on how to modify your environment variables.


You should now be ready to run `cfl`.
Check that your installation has been successful by opening a Python terminal from the cfl conda environment (or whatever environment you're using) and typing `import cfl`.
