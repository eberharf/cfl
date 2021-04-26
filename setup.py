from setuptools import setup, find_packages

setup(
    name='cfl', 
    version='0.1.1', # version <1.0 indicates 'pre-release' 
    description='Causal Feature Learning', #TODO: one sentence summary 
    url='https://github.com/eberharf/cfl',
    author='Jenna Kahn and Iman Wahle',
    author_email = 'iwahle@caltech.edu',
    # license='BSD 2-clause',  # TODO: what is our license
    packages=find_packages(where='cfl'), #TODO: add visual bars? what else do we package? 
    package_dir={'': 'cfl'},
    python_requires = ">=3.7",  #TODO: good? it probably works with 3.6 too idk. guess that means do testing
    # install_requires=['tqdm','matplotlib' ,'tensorflow', 'keras', 'numpy~=1.19.2', 'scikit-learn'],
    install_requires=['matplotlib' ,'tensorflow', 'keras', 'numpy~=1.9.3', 'scikit-learn'], 
 

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: Free for non-commercial use', #TODO: update if we change license
        'Natural Language :: English',
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