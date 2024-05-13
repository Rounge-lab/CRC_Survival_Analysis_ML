import pandas as pd
import numpy as np
import os

np.random.seed(91)

def main():
  """
    Split each data group into a train and a test set on unique IDs maintaining 
    distributions of confounders and event / censoring.
    Return the IDs for each train and test split.
  """
  # import preprocessed metadata and miRNA count tables
  df = pd.read_csv(f"path")
  df_count = pd.read_csv(f"path")

  df_melted = df_count.melt(id_vars=df_count.columns[0], ignore_index=False)
  df_pivoted = df_melted.pivot(index=df_melted.columns[1], columns = df_melted.columns[0])
  
  # Reset the index to make the first column a regular column
  df_pivoted.reset_index(inplace=True)

  # get unique id's for splitting into train/test
  df_byID = df.groupby("JanusID", as_index=False).first()
  df_byID = df_byID[["JanusID", "sex", "alder_tdato", "bd_grp"]]
  
  # Split into train/test set based on desiered fraction frequency matching on sex, age and blood donor group
  df_test_ = df_byID.sample(frac=0.2, random_state = 12)
  df_train_ = df_byID[~df_byID['JanusID'].isin(df_test_['JanusID'])]
  df_test_.to_csv(f"name")
  df_train_.to_csv(f"name")
  
  os.system("Rscript gen_tr_t_split.R")

if __name__ == "__main__":
  main()
