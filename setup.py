from setuptools import setup, find_packages
import cfl

VERSION = cfl.__version__
AUTHORS = cfl.__author__
CONTACT = 'imanwahle@gmail.com'

# to create a new distribution for cfl
# run the commands:
#       python setup.py sdist bdist_wheel
#       twine upload dist/*
# SOURCE: see
# https://betterscientificsoftware.github.io/python-for-hpc/tutorials/python-pypi-packaging/#creating-a-python-package
# for more information on how to package code for pyPI

# TODO: right now, only cfl/ and visual_bars/ are packaged for installation
# eventually, we may want to package more things (docs and/or examples and/or
# unit tests )

setup(
    name='cfl',
    version=VERSION,
    # TODO: one sentence summary
    description='Causal Feature Learning (CFL) is an unsupervised algorithm designed to construct macro-variables from low-level data, while maintaining the causal relationships between these macro-variables. ',
    long_description='See cfl.readthedocs.io for a full description',
    url='https://github.com/eberharf/cfl',
    author=AUTHORS,
    author_email=CONTACT,
    # package main cfl package and visual bars (find_packages() returns a list of any folder with an __init__ file)
    packages=find_packages(),
    # TODO: ^ do we package other stuff too?
    python_requires=">=3.7",  # TODO: good? it probably works with 3.6 too ...
    install_requires=[  # TODO: this list contains semi-redundant information with requirements.yml
        'tqdm',
        'matplotlib',
        'tensorflow>=2.4.0',
        'numpy>=1.19.2',
        'scikit-learn>=0.23',
        'jupyter',  # for jupyter notebooks
        'ipykernel',
        'joblib>=0.16.0'
    ],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: Free for non-commercial use',
        # 'Operating System  :: MacOS :: MacOS X',
        # 'Operating System :: Microsoft :: Windows :: Windows 10',
        'Programming Language :: Python :: 3.6',  # TODO: other versions?
        'Programming Language :: Python :: 3.7',  # TODO: other versions?
        # NOt with python 3.9 at the moment bc of Tensorflow
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        # TODO: are these topics appropriate classifiers ?
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
)
