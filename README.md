![Build
Status](https://github.com/eberharf/cfl/workflows/AutomatedTests/badge.svg)
[![Python
3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/)

<p align="center">
<img src=docs/logo.jpg width="300" />
</p>
<h1 align="center">Causal Feature Learning</h1>

<h3 align="center"><a href="https://cfl.readthedocs.io/en/latest/index.html#">Tutorials and Documentation</a></h3>

-- -------------------------------------- --

Causal Feature Learning (CFL) is an unsupervised algorithm designed to construct
macro-variables from low-level data, preserving the causal relationships present
in the data.


Please visit our [ReadTheDocs page](https://cfl.readthedocs.io/en/latest/index.html#)
for the latest documentation and tutorials.

<!-- -------------------------------------------- -->
## Installation

```
pip install cfl
```
<!-- 

## Set-up 

Go [here](https://cfl.readthedocs.io/en/latest/getting_started/SETUP.html) for
instructions on installing and setting up CFL. 

## Running CFL

Go to the
[examples](https://cfl.readthedocs.io/en/latest/examples/cfl_code_intro.html)
section of our Read The Docs page to see several demonstrations of how to use
the CFL code. Check out the
[Background](https://cfl.readthedocs.io/en/latest/getting_started/cfl_intro.html)
on CFL for a quick theoretical introduction.


## Repository Contents
### `cfl`
This folder contains all of the functional code for CFL. The most current
function-level documentation for the `cfl` package is currently in the
docstrings (API level documentation will soon be added to ReadtheDocs)


### `docs/source/examples`
contains example applications of `cfl` for various data sets. Look here if
you're just getting started.

### `visual_bars`
Contains code for generating visual bars data set (see Chalupka 2015) and code
to efficiently test the performance of CFL with different parameters on the
visual bars data set. We use the visual bars data as simple toy data to run
through different parts of CFL. Since this data is entirely synthetic, the
ground truth at each step is known and can be compared against the CFL results.
Details about this data set can be found on ReadtheDocs.

- `generate_visual_bars_data.py`: module to generate VisualBarsData objects,
  which create and return images and the associated properties of the images (eg
  ground truth, target behavior)

### `data/el_nino`
Contains the pickle file for the El Nino data. See the `el_nino_example.ipynb`
notebook for an example of how to load this data.

### `tests`
This folder contains the automated test suite for checking the expected
functionality of the code and preventing regression (loss of functionality). -->

<!-- -------------------------------------------- -->
## Contributors

- Jenna Kahn & Iman Wahle [first authors; order chosen randomly]
- Krzysztof Chalupka
- Patrick Burauel
- Pietro Perona
- Frederick Eberhardt


Jenna Kahn and Iman Wahle designed the software and wrote the code in this
repository.

Krzysztof Chalupka, Pietro Perona and Frederick Eberhardt developed the original
theory for CFL. Krzysztof also wrote the original code upon which this software
is based.

Code development benefitted from regular discussions with Patrick Burauel.



<!-- -------------------------------------- -->
## License and Citations

CFL is released under a BSD-like license for non-commercial use only. If you use
CFL in published research work, we encourage you to cite this repository:

```
Causal Feature Learning (2022). https://github.com/eberharf/cfl
```

or use the BibTex reference:

```
@misc{cfl2022,
    title     = "Causal Feature Learning",
    year      = "2022",
    publisher = "GitHub",
    url       = "https://github.com/eberharf/cfl"}
  }
```

<!-- ----------------------------------------------------------

----------------------------------- -->

## Questions? Comments? Feedback? 

Contact Iman Wahle (imanwahle@gmail.com) or Jenna Kahn. 
