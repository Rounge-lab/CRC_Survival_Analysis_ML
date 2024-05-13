#!/bin/bash

for i in {0..49};
do
        ./run.sh $i 8 
done
wait
echo summarizing
srun python results_summary_.py 0 49

#savefile df_strong_coef_summary.csv
