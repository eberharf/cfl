.. cfl documentation master file, created by
   sphinx-quickstart on Thu Dec 10 13:42:33 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Welcome to CFL
=====================

Causal Feature Learning (CFL) is an unsupervised algorithm designed to construct macro-variables from low-level data, accounting for the causal relationships between these macro-variables. 


.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   getting_started/SETUP
   getting_started/cfl_intro.md
   getting_started/quick_start.ipynb

.. toctree:: 
   :maxdepth: 1
   :caption: Additional Examples 

   examples/cfl_code_intro.ipynb
   examples/el_nino_example.ipynb
   examples/mnist.ipynb
   examples/basic_visualizations.ipynb

API Reference 
*********************************
:ref:`api-index`

.. toctree:: 
   :maxdepth: 1
   :caption: More Info 

   more_info/Visual_Bars_data
   more_info/CDEs
   more_info/clustering
   more_info/dvc_intro



Contributors
*****************************************************

-  Jenna Kahn & Iman Wahle [first authors; name order chosen randomly]
-  Krzysztof Chalupka
-  Daniel Israel 
-  Patrick Burauel
-  Pietro Perona
-  Frederick Eberhardt

Jenna Kahn and Iman Wahle designed the software and wrote the code in
this repository. Daniel Israel wrote the MNIST example notebook and 
constributed feedback to the code. 

Krzysztof Chalupka, Pietro Perona and Frederick Eberhardt developed the
original theory for CFL. Krzysztof also wrote the original code upon
which this software is based.

Code development benefitted from regular discussions with Patrick
Burauel. 

License and Citations
********************************************

CFL is released under a BSD-like
license for non-commercial use
only. If you use CFL in
published research work, we
encourage you to cite this
repository:

::

   Causal Feature Learning (2021). https://github.com/eberharf/cfl

or use the BibTex reference:

::

   @misc{cfl2021,
   title = "Causal Feature Learning",
   year = "2021",
   publisher = "GitHub",
   url = "https://github.com/eberharf/cfl"}


