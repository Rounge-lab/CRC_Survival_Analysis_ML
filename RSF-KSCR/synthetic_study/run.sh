#! /bin/sh
run=$1
max_cores=$2

# Generate synthetic data for the given run of the study and save as csv files in data_sets/data_files subfolder
echo "Creating synthetic data for run ${run}"
python generate_syn_study_data.py $run 100 1000 True 1 False True

# Create file names for the data files for string concatination in script arguments
file_1="run_${run}_df_strong_coef_t0.csv" 
file_2="run_${run}_df_strong_coef_full_t0.csv"

echo "Starting analysis for run ${run}"
for f in $file_1 $file_2;
do
    # string concatination variables for file names
    name=${f::${#f}-4}
    test_id="_test_ids.csv"
    rsf_chosen="_rsf_chosen_var_supervised.csv"
    
    # Split data into train and test on subject ids and save as csv file in data_sets/data_files subfolder
    echo "Splitting data into train and test for ${f}"
    python train_test_id_split.py $f $run

    # Run the RSF model minimal depth variable selection and create rsf prediction file for evaluation
    # and save the chosen variables as a csv file in data_sets/covars subfolder
    echo "Running RSF model minimal depth variable selection for ${f}"
    Rscript --vanilla step_1_rsf_min_depth_var_selection.R $f ${name} ${name}${test_id} ${max_cores}
    
    # Fit Kernels moothed Cox model with constant coefficents and create predictions and evaluation files for the model
    # save results as csv files in results final subfolder
    echo "Running Kernels moothed Cox model with constant coefficents for ${f}"
    python step_2_tvcox.py $f ${name}${rsf_chosen} ${name}${test_id} 1 0 ${max_cores} $1
    
    # Fit Kernels moothed Cox model with time quadratic b-spline time varying coefficents and create predictions and 
    # evaluation files for the model. Save results as csv files in results final subfolder
    echo "Running Kernels moothed Cox model with time quadratic b-spline time varying coefficents for ${f}"
    python step_2_tvcox.py $f ${name}${rsf_chosen} ${name}${test_id} 3 2 ${max_cores} $1

done

sleep 10
# Clean up datafiles from the run
echo "Cleaning up data files for ${f}"
python cleanup.py ${run}

done

