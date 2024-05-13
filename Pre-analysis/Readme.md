# Pre-processing steps

## Content of folder
- All R and python scripts used to clean and pre-process the RNA count tables and
  metadata, split the data into data groups and data groups into train and test
  sets.
  
## Script explanations
- CRC_Data_Prep_miRNA.R: Processes counttables and metadata. performes data
  wrangling and splits into the three data groups, Full, No bdgrp1 and 
  No bdgrp1 no NA (called Complete metadata in this work). Saves these as csv 
  files. Original script aquiered from within the JanusRNA group, author is 
  credited in the script.

- Generate_Holdout_IDs.py: splits each datagroups unique JanusIDs into train 
  and test split attempting to maintain the confounder and event/censoring 
  distributions. Saves ID's into csv files 

- Generate_Tr_Te_Split.R: Uses the split IDs from Generate_Holdout_IDs.py
  to generate the data group files for both training and test. saves these as
  csv files.

- Generate_Combind_data.R: Uses the csv files from CRC_Data_Prep_miRNA.R
  to create datasets gor each datagroup using all samples (train and test)
