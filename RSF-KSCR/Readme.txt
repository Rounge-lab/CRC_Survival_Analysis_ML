# RSF-KSCR procedure

## Content of folder
- All R and python scipts used for the synthetic study and RSF-KSCR analysis frameworks

- src: model package, to use: pip install src and import TVKCox_Regression (see 
  Readme in src folder)

- synthetic_study: self-contained folder for running the synthetic study (see
  Readme in synthetic_study folder)
  
## Script explanations
- Minimal_Depth_VariableSelection.R: Performs minimal depth variable selection 
  saving the covariates selected in a csv file.

- GridSearch.py: Runs the grid search using sklearns GridSearchCV framework with 
  TV-AUC as criteria, saves results and parameters in csv and pickles models.

- GridSearch_custom.py: Runs the grid search using a custom CV implementation
  and partial log likelihood change without test fold as creteria, saves results 
  and parameters in csv and pickles models.

- Evaluate_GridSearcg_Results.py: sequentially prints each model along the
  regularization paths number of non-zero coefficients and training data 
  evaluation metrics. Can generate plot of regularization path for constant
  coefficents and save indicies of the zero-coefficents for the selected model.

- Bootstrap_Cox_Model.py: Bootstraps the selected model to generate confidence
  intervals, saves resulting arrays in csv files. 

- Create_CoefficientPlots.py: Generates plots of the non-zero coefficients,
  constant or time varying, with pr without confidence intervals.

- Generate_Test:Predictions.py: Produces csv file with predictions from the 
  selected KSCR models on a given data group for the holdout test set. 

- Evaluate_Test_Predictions.R: calculates evaluation metrics and creates plots
 / stores csv files with results. 