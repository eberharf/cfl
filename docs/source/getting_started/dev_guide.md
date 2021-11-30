# Contribute to CFL

Contributions to `cfl` are welcomed and encouraged! Here are some ways to 
contribute.

### Submit a bug report or feature request. Please include:
    - a short code snippet that reproduces the issue
    - any error tracebacks
    - operating system type and version number, python version number, and 
      cfl version number

### Contribute code, docs, and tests (more detailed instructions to come)
    - fork the cfl repository
    - clone your fork to your local machine
    - install the development dependencies in requirements.yml
    - add the upstream remote
    - sync your main branch with the upstream main branch
    - create a feature branch and make your changes on it
    - run pytest to ensure all tests still pass
    - commit and push
    - open a pull request

### Contribute `Block`s
Did you develop your own conditional probability estimator or clusterer while
performing your analysis? Please consider sharing it with others! Your `Block`
should:
    - algorithmically align with the type of `Block` it is (i.e. a new 
      `CauseClusterer` `Block` should perform unsupervised clustering on the 
      conditional probabilities estimated by any `CondProbEstimator` and
      return cluster labels over all samples)
    - be placed in the corresponding directory
    - pass all associated tests (instructions to come for how to test your 
      `Block`)
    - inherit the `Block` class or a child of the `Block` class
Please follow the instructions under "Contribute code" to create a pull request.

### Contribute examples
Have a cool dataset you've run `cfl` on? We'd love to see it!
    - Put together a concise Jupyter Notebook with some background on your data
      and an annotated run of CFL, similar to the [El Niño example notebook]
      (https://cfl.readthedocs.io/en/latest/examples/el_nino_example.html).
    - Include this notebook in the `docs/source/user_examples/` directory.
    - Follow the instructions under "Contribute code" to create a pull request.