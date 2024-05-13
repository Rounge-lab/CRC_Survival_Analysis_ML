# Time-varying Kernel Smoothed Cox Regression package

## Content explanation:
- TVKCox.py: Contains an implementation of the kernel smoothed Cox regression 
  model and optimization algorithm for sparse data with time-varying coefficients.
  The primary interface is the TVKCox_regression class which fits the model and
  is compatable with sklearn pipeline interface using the 
  set_output(transform="pandas") function.

- helper_functions: Contains an implementation of B-spline basis function
  generation using the Cox De Boor frmula. Not interacted with directly.

- generate_synthetic_survdata.py: Contains an implementation of a synthetic
  data generator that handles time-varying covariates and coefficients, categorical
  covariates and random baseline hazards. The main interface is the function:
  get_longitudinal_testdata().
