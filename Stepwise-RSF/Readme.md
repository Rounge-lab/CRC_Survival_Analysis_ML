# Stepwise procedure for RSF

## Content of folder
- All R and python scripts used for the analysis framework utilizing automated stepwise covariate selection with random survival forests (RSF)

## Script explanations
- stepwise_rsf_procedure.R performs the stepwise algorithm, takes 4 input arguments: 
  1: An integer between 1 and 3 to select the data group
  2: A string, either "metastasis" or "location" or "none" to select cancer specific subsets or not
  3: A string, either "Distant", "Regional", "Localized", "distal", "proximal" to select which subset to use
  4: A string, either "true" or "false" to select if training and test data should be used together or not

- bootstrap_rsf_model.R performs bootstrapping for the selected covariates, takes 1 argument:
  1: An integer between 1 and 3 to select the data group

- process_evalMetrics_dataGroups.R calculates various performance metrics by datagroup and save them + creates some plots, takes 1 argument:
  1: A string, either "T" or "F" to select if evaluation should be done on the holdout test set or using out-of-bag samples
  
- process_evalMetrics_cancerSpecific.R calculates various performance metrics by cancer specific subsets and save them + creates some plots, takes 1 argument:
  1: A string, either "T" or "F" to select if evaluation should be done on the holdout test set or using out-of-bag samples
  
- process_evalMetrics_alldata.R calculates various performance metrics by data group using train and test data and save them + creates some plots, takes no arguments

## Remaining files
- All remaining files have descriptive names and are run in rstudio as ad-hoc analysis and plot creation
