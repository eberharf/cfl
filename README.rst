cfl
===

Set-up Instructions
-------------------

Instructions for setting installing CFL and its dependencies can be
found `here`_ - A quick start guide with example code can be found
`here <https://github.com/eberharf/cfl/blob/master/examples/quick_start_guide.ipynb>`__
- Complete documentation can be found here

Running CFL
-----------

Go to the `examples`_ folder to find Jupyter Notebooks that demonstrate
how to use the CFL code. Check out the [Quick Start
Guide](https://github.com/eberharf/cfl/blob/master/examples/quick_start_guide.ipynb
first if you’re just getting started.


License and Citations
~~~~~~~~~~~~~~~~~~~~~~~~~~

CFL is released under a BSD-like
license for non-commercial use
only. If you use CFL in
published research work, we
encourage you to cite this
repository:
+----------------------------------+
| ``Causal Feature Learning (2021). |
| https://github.com/eberharf/cfl`` |
+----------------------------------+

or use the BibTex reference:
+----------------------------------+
| ``@misc{cfl2021, title = "Causal Feature Learning",|
| year      = "2021",                                |
| publisher = "GitHub",                              |
| url       = "https://github.com/eberharf/cfl"} }`` |
+----------------------------------+

Contributors
------------

-  Jenna Kahn & Iman Wahle [first authors; order chosen randomly]
-  Krzysztof Chalupka
-  Patrick Burauel
-  Pietro Perona
-  Frederick Eberhardt

Jenna Kahn and Iman Wahle designed the software and wrote the code in
this repository.

Krzysztof Chalupka, Pietro Perona and Frederick Eberhardt developed the
original theory for CFL. Krzysztof also wrote the original code upon
which this software is based.

Code development benefitted from regular discussions with Patrick
Burauel.

--------------

Repository Contents
-------------------

.. _cfl-1:

``cfl``
~~~~~~~

This folder contains all of the functional code for CFL. The most
current documentation for the ``cfl`` package can be viewed using
``PyDoc``. Use the following instructions to open the documentation:

1. First, make sure that you have a local copy of the ``cfl`` repository
   installed on your machine according to the above instructions.

2. Open a terminal window. Start a PyDoc server on the HTTP port 1234 by
   typing:

::

   python -m pydoc -p 1234

3. Press ``b`` to open the webpage in your browser.
4. Scroll past the Built-In Modules to the link to **``cfl``**
   ``(package)``. Click on this link to view the various sub-modules in
   ``cfl`` and see details about each module.

``examples``
~~~~~~~~~~~~

contains example applications of ``cfl`` for various data sets. Look
here if you’re just getting started.

``visual_bars``
~~~~~~~~~~~~~~~

Contains code for generating visual bars data set (see Chalupka 2015)
and code to efficiently test the performance of CFL with different
parameters on the visual bars data set. We use the visual bars data as
simple toy data to run through different parts of CFL. Since this data
is entirely synthetic, the ground truth at each step is known and can be
compared against the CFL results. Details about

.. _here: https://github.com/eberharf/cfl/blob/master/SETUP.md
.. _examples: https://github.com/eberharf/cfl/blob/master/examples