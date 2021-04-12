# Welcome to CFL 

![Build Status](https://github.com/eberharf/cfl/workflows/dev_regression_tests/badge.svg)

## Set-up 

Go [here](https://cfl.readthedocs.io/en/latest/SETUP.html) for instructions on installing and setting up CFL. 

## Running CFL

Go to the `examples` directory in GitHub for Jupyter Notebooks that demonstrate how to use the CFL code. Check out the [Background](https://cfl.readthedocs.io/en/latest/cfl_intro.html) on CFL for a quick theoretical introduction.

-----------------------------------------
-------------------------------------------
## Repository Contents
### `cfl`
This folder contains all of the functional code for CFL. The most current function-level documentation for the `cfl` package is currently in the docstrings (API level documentation will soon be added to ReadtheDocs)


### `examples`
contains example applications of `cfl` for various data sets. Look here if you're just getting started.

### `visual_bars`
Contains code for generating visual bars data set (see Chalupka 2015) and code to efficiently test the performance of CFL with different parameters on the visual bars data set. We use the visual bars data as simple toy data to run through different parts of CFL. Since this data is entirely synthetic, the ground truth at each step is known and can be compared against the CFL results. Details about this data set can be found on ReadtheDocs.

- `generate_visual_bars_data.py`: module to generate VisualBarsData objects, which create and return images and the associated properties of the images (eg ground truth, target behavior)

### `data/el_nino`
Contains the pickle file for the El Nino data. See the `el_nino_example.ipynb` notebook for an example of how to load this data.

### `data/galaxies`
Contains the .dvc (data version control) files associated with each file of the galaxy data. The actual data is stored on Google Drive.

Instructions on how to use .dvc are in `examples/dvc_intro.md`

### `testing`
This folder contains the automated test suite for checking the expected functionality of the code and preventing regression (loss of functionality).

--------------------------------------------
## Contributors

- Jenna Kahn & Iman Wahle [first authors; order chosen randomly]
- Krzysztof Chalupka
- Patrick Burauel
- Pietro Perona
- Frederick Eberhardt


Jenna Kahn and Iman Wahle designed the software and wrote the code in this repository.

Krzysztof Chalupka, Pietro Perona and Frederick Eberhardt developed the original theory for CFL. Krzysztof also wrote the original code upon which this software is based.

Code development benefitted from regular discussions with Patrick Burauel.



--------------------------------------
## License and Citations

CFL is released under a BSD-like license for non-commercial use only. If you use CFL in published research work, we encourage you to cite this repository:

```
Causal Feature Learning (2021). https://github.com/eberharf/cfl
```

or use the BibTex reference:

```
@misc{cfl2021,
    title     = "Causal Feature Learning",
    year      = "2021",
    publisher = "GitHub",
    url       = "https://github.com/eberharf/cfl"}
  }
```

----------------------------------------------------------

-----------------------------------

## Questions? Comments? Feedback? 

Contact Iman Wahle or Jenna Kahn. 
