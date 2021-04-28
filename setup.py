from setuptools import setup, find_packages

setup(
    name='cfl', 
    version='0.1.3', # version <1.0 indicates 'pre-release' 
    description='Causal Feature Learning (CFL) is an unsupervised algorithm designed to construct macro-variables from low-level data, while maintaining the causal relationships between these macro-variables. ', #TODO: one sentence summary 
    url='https://github.com/eberharf/cfl',
    author='Jenna Kahn and Iman Wahle',
    author_email = 'iwahle@caltech.edu',
    packages=find_packages(where='cfl'), #TODO: add visual bars? what else do we package? 
    package_dir={'': 'cfl'},
    python_requires = ">=3.7",  #TODO: good? it probably works with 3.6 too idk. guess that means do testing
    install_requires=['tqdm','matplotlib' ,'tensorflow==2', 'numpy>=1.19.2', 'scikit-learn>=0.24'], 

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: Free for non-commercial use',
        # 'Operating System  :: MacOS :: MacOS X',
        # 'Operating System :: Microsoft :: Windows :: Windows 10',
        'Programming Language :: Python :: 3.7', # TODO: other versions? 
        'Programming Language :: Python :: 3.8', # TODO: other versions? 
        'Programming Language :: Python :: 3.9', # TODO: other versions? 
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence' #TODO: are these topics appropriate classifiers ?
    ],
)