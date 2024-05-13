import os
import sys

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import csv
import pickle
from TVKCox_Regression import TVKCox_regression, Kernels
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import joblib
import matplotlib.pyplot as plt
import subprocess

plt.style.use('ggplot')

def fix_names(name_list:list):
  """
  Fixes incorrect naming from R data.frames
  Parameters
        ----------
        name_list : list
            List of strings with miRNA names to be corrected.

        Returns
        -------
        list
            List of strings with corrected miRNA names.
  """
  for col in name_list:
    i = name_list.index(col)
    temp = col.replace(".", "-")
    name_list = name_list[:i] + [temp] + name_list[i+1:]
  
  return name_list

def main(data_name:str, time_scale:str, q:int, model_name:str, get_vars:str, a:float, B:int, time_varying:bool, n_jobs:int, dg:str, mod_idx:int):
  
  np.random.seed(12)
  q = int(q)
  a = float(a)
  B = int(B)
  time_varying = time_varying == "true"
  n_jobs = int(n_jobs)
  mod_idx = int(mod_idx)
  
  # get the data and process it to appropriate form for the KSCR model:
  df = pd.read_csv(f"{data_name}.csv")
  df = df.drop("Unnamed: 0", axis=1)
  df = df.rename(columns={"tdato_diag_time_years": "T", "condition": "delta", "TDATO": "t", "JanusID": "id"})
  df["delta"].replace(["CRC", "C"], [1,0], inplace=True)
  control_to_case = [10.02075, 10.02622, 10.07824, 10.90783, 10.00432]
  for i in control_to_case:
    df[df.T == i]["delta"] = 1
  df.loc[df.delta == 0, "T"] = 10.0
  df["t"] = pd.to_datetime(df["t"].astype("str"))
  
  temp = df.groupby(["id", "delta"]).max("t")["T"].reset_index()
  
  temp["new_col"] = list(zip(temp["id"], temp["delta"], temp["T"]))
  id_list = temp.new_col.to_list()
  for id, delta, T in id_list:
    df.loc[(df["id" ] == id) & (df["delta"] == delta), "T"] = T 
  
  temp = df.groupby(["id", "delta"]).t.min().reset_index()
  
  temp["new_col"] = list(zip(temp["id"], temp["delta"], temp["t"]))
  id_list = temp.new_col.to_list()
  for id, delta, t in id_list:
    df_temp = df.loc[(df["id"] == id) & (df["delta"] == delta)].index.values.tolist()
    for value in df_temp:
      df.at[value, "t"] = (df.loc[value]["t"] - t).days / 365
  
  df["t"] = pd.to_numeric(df.t)
  
  # get chosen covariates:
  
  # either use covariates from minimal depth threshold, from stepwise RSF procedure or all
  if get_vars == "md_sel":
    df_cv = pd.read_csv(f"md_var_sel_{data_name}_{time_scale}.csv")["Unnamed: 0"].tolist()
    df_cv = fix_names(df_cv)
  
  elif get_vars == "stepwise":
    
    if data_name == "janus_crc_training":
      df_cv = pd.read_csv(f"output_all_data_reg_time.csv")
    elif data_name == "janus_crc_no_bdgrp1_training":
      df_cv = pd.read_csv(f"output_no_bdg1_reg_time.csv")
    else:
      df_cv = pd.read_csv(f"output_no_bdg1_no_NA_reg_time.csv")
    
    idx = df_cv["rsf.err"].idxmin()
    df_cv = df_cv.iloc[idx:]["rm.var"].tolist()
    df_cv = fix_names(df_cv)
  else:
    pass
  
    
  miRNA_cols = df.columns[1:371].tolist()
  if get_vars in ["md_sel", "stepwise"]:
    miRNA_cols = [col for col in miRNA_cols if col in df_cv]

  
  meta_cols = df.columns[371:].tolist()
  meta_cols.append(df.columns[0])
  
  # process columns for the KSCR model
  if data_name == "janus_crc_no_bdgrp1_no_NA_training":
    df_cols = miRNA_cols + ["id", "T", "t", "delta","sex", "alder_tdato", "bmi", "smoking_status"]
    df = df[df_cols]
    df.loc[:, "sex"] = (df["sex"] == "M").astype("int")
    df.loc[:, "smoking_status"] = (df["smoking_status"] == "smoker").astype("int")
    cont_cols = [col for col in df_cols if col not in ["id", "T", "t", "delta", "sex", "smoking_status"]]
    num_cat_cols = 2
  else:
    df_cols = miRNA_cols + ["id", "T", "t", "delta","sex", "alder_tdato"]
    df = df[df_cols]
    df.loc[:, "sex"] = (df["sex"] == "M").astype("int")
    cont_cols = [col for col in df_cols if col not in ["id", "T", "t", "delta", "sex"]]
    num_cat_cols = 1
  
  # adjust scale if chronological age is to be used
  use_age = False
  if use_age:
    df["T"] = df["T"] + df["alder_tdato"] - df["t"] 
    df["t"] = df["alder_tdato"] 
    df = df.drop(columns=["alder_tdato"], axis=1)
    df_cols = [col for col in df_cols if col not in ["alder_tdato"]]
    cont_cols = [col for col in cont_cols if col not in ["alder_tdato"]]
  
  df = df.dropna()
  
  # split into data av event indicator
  delta = df["delta"].copy(deep=True)
  df = df.drop(columns=["delta"], axis=1)
  
  # Get study timeframe
  tau = df["T"].max()
  
  # get parameters of best model from qualitative assesment
  tvcox_pipeline_p = pickle.load(open(f"{model_name}.p", "rb"))
  _covariate_list = tvcox_pipeline_p.named_steps.tvcox_model.covariate_list
  
  def get_bootstrap_CI(df, delta, mod, alpha:float = 0.1, B:int = 100, time_varying:bool = True, n_jobs:int = 1, verbose:int = 1):
    if not time_varying and mod.named_steps.tvcox_model.q > 1:
        raise ValueError(f"Non constant coefficient estimates must have time varying confidence intervals")
    
    # Discritize B-spline basis functions for matmul with model coefficents to generate coefficient paths 
    eval_times = np.linspace(0, tau, 50)
    basis_func_values = np.zeros((eval_times.shape[0], mod.named_steps.tvcox_model.q))
    for func in range(mod.named_steps.tvcox_model.q):
        basis_func_values[:,func] = mod.named_steps.tvcox_model.spline_basis_functions[func](eval_times)
    
    # Hold bootstrapped coefficent paths
    bootsrap_results = np.zeros((mod.named_steps.tvcox_model.p, 50, B+1))
    bootsrap_results[:,:,0] = mod.named_steps.tvcox_model.gamma_matrix @ basis_func_values.T
    
    def bs(i):
        if verbose >= 1:
            print(f"Progress: starting interation {i+1} out of {B}")
        
        # make bootstrapped training dataset and oob evaluation set
        bs_id = df["id"].sample(n = mod.named_steps.tvcox_model.n, replace=True, random_state=i)
        df["delta"] = delta
        data = pd.concat([df[df["id"] == id] for id in bs_id])
        oob_data = df[~df["id"].isin(list(data["id"].values))]  
        temp_delta = data["delta"].copy(deep=True)
        oob_delta = oob_data["delta"].copy(deep=True)
        data = data.drop(columns=["delta"], axis=1)
        oob_data = oob_data.drop(columns=["delta"], axis=1)
        
        # Initiate model pipeline
        ct = ColumnTransformer(
                       [("sc",StandardScaler(), cont_cols)],
                        remainder='passthrough',
                        verbose_feature_names_out = False
                )
        
        tvcox = TVKCox_regression()
        # Set the best model parameters 
        tvcox.set_params(**mod.named_steps.tvcox_model.get_params())
        
        tvcox_pipeline = Pipeline(
                [ ("scaler", ct),
        ("tvcox_model", tvcox)]
        )
        tvcox_pipeline.set_output(transform="pandas")
        # Set new random state for each bootstrap
        tvcox_pipeline.named_steps.tvcox_model.set_params(**{"rand_state": i+1, "scoring_method": "c_index_indv", "verbose": 1})
        
        # fit model and extract coefficient paths
        tvcox_pipeline.fit(data, temp_delta)
        
        ci_res = tvcox_pipeline.named_steps.tvcox_model.gamma_matrix @ basis_func_values.T
        mod.named_steps.tvcox_model.set_params(**{"scoring_method": "c_index_indv"})
        
        if verbose >= 1:
          print(f"Progress: ending interation {i+1} out of {B}")
        return (i, ci_res)
       
    
    bs_res = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(bs)(i) for i in range(0, B))

    for res in bs_res:
      bootsrap_results[:,:,res[0]+1] = res[1]
    
    return bootsrap_results
 
  results_ci = get_bootstrap_CI(df, delta, tvcox_pipeline_p, a, B, time_varying, n_jobs, 1) 
  
  # get indicies of zero coefficents
  try:
    with open(f"{dg}_{mod_idx}_q_{q}_idx_list", "rb") as fp:
      idx_list = pickle.load(fp)
  except:
    idx_list = []
  
  # reshape 3D array for storage and store variables for reconstruction
  _origdim = np.array(results_ci.shape)
  if results_ci.ndim > 2:
    results_ci = results_ci.reshape((results_ci.shape[0], -1))
  
  # store list of zero covariates with zero coefficents
  covariate_list = [i for j, i in enumerate(_covariate_list) if j not in idx_list]
  with open(f"cox_bs_{dg}_q{q}_covlist", "wb") as fp:
    pickle.dump(covariate_list, fp)
  
  # store coefficent paths and recostruction parameters
  np.savetxt(f"cox_bs_{dg}_q{q}_origdim.csv", _origdim ,delimiter=",")
  np.savetxt(f"cox_bs_ci_{dg}_q{q}.csv", results_ci, delimiter=",")
  
  
if __name__ == "__main__":
  main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5].lower(), sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9], sys.argv[10], sys.argv[11])
  
    
  
