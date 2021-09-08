# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from recommonmark.parser import CommonMarkParser # for parsing Markdown files

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

# set path to root directory 
# don't set it to the main cfl code directory (ie '../../cfl'), because that will make readthedocs not work
sys.path.insert(0, os.path.abspath('../..')) 

# further modules needed for autodoc
import tensorflow

# -- Project information -----------------------------------------------------
import cfl

project = 'cfl'
copyright = cfl.__credits__
author = cfl.__author__

# The full version, including alpha/beta/rc tags
release = cfl.__version__ 


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', #for auto-using docstrings
              'sphinx.ext.napoleon', #for converting Numpy/Google style-doc strings to reST
              'nbsphinx', # for incorporating jupyter notebooks into docs
              'recommonmark', # for parsing markdown files
              'sphinx.ext.mathjax' #nbsphinx asks for this 
            ]

## Napoleon settings (for docstring formatting)
napoleon_google_docstring=True
napoleon_include_init_with_doc=True
napoleon_include_private_with_doc=True
napoleon_use_param=True

source_parsers = {'.md': CommonMarkParser}

# source_suffix = {'.rst': 'restructuredtext',
#                  '.md':  'markdown',
#                  '.ipynb': 'jupyter' #This line doesn't work 
#                  }   

source_suffix = ['.rst', '.md', '.iypnb']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []