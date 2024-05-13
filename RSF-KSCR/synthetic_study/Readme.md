# Simulation study
Runs the simulation study used in this master thesis

## Content

-   generate_syn_study_data.py to generate the simulated data and save
    as csv files.

-   train_test_id_split.py to split into train and test sets on unique ID
    and save them as csv files.

-   step_1_rsf_min_depth_selection.R to perform minimal depth variable step_1_rsf_min_depth_selection
    and save the selected covariates names to a csv file.

-   step_2_tvcox.py runs the kernel smoothed Cox regression model with limited
    gridsearch, evaluates the model and stores selected covariates and performance
    scores as csv files.

-   random_inspection.py runs the second step saving a plot of coefficient paths and
    printing results for inspecting a single randomly chosen study.

-   cleanup.py Cleans up all csv files except final performance scores and selected covariates.

-   results_summary.py summarizes the results over all runs and saves as a sumamry csv file
    deleting the performance score csv files underway. 

-   run.sh runs one instance of the study takes the run number (functions as random seed) and
    max number of cores to use as input (runs all scripts above except results_summary.py
    and random_inspection.py)

-   run_study.sh run a loop for the desiered number of studies to perform and summarizes

-   random_inspection.sh runs the random inspection and cleans up

-   random_inspection_script.sh generates the random run id runs the random_inspection.sh
    script. 

## How to use

-   Edit the run_study.sh script to select the number of studies to run and how many cores 
    to use then run ./run_study.sh in terminal

-   To perform random inspection run ./random_inspection_script.sh in your terminal

## Note
make sure yoy have all python and R requierements installed as well as install the 
src TVKCox_Regression package. 
