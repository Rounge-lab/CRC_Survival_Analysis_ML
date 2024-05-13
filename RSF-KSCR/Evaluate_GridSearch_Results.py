import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import ast
import pickle
import subprocess
from labellines import labelLines

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

def main(data_name:str, dg:str, time_scale:str, get_vars:str, q:int, ts:str, choose_model:str, mod_idx:int):
  """
  Loads models generated when calculating full regularization path using GridSearch_Custom.py or
  GridSearch.py for qualitative assesment and selection of final model.

        Parameters
        ----------
        data_name : str
            File path to the dataset used
        dg: str
            The data group used
        time_scale : str
            Which time scale to use (reg_time->time-to-diagnosis or age->chornological age)
        get_vars : str
            Indicator of which selected covariates to use (md_sel->from minimal depth thresholding, 
            stepwise-> selected using the stepwise procedure)
        q : int
            The B-spline coefficent number (1-> constant)
        ts : int
            Integer representing the timestamp for the models to load (YYYYMMDD)
        choose_model : str
            Indicator to save the selected model using the index provided in mod_idx ("true" -> save)
        mod_idx : int
            Index of the selected model, represents the order the models are evaluated 

        Returns
        -------
        None
  """
  np.random.seed(23)
  
  q = int(q)
  mod_idx = int(mod_idx)
  choose_model = choose_model == "true"
  
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
  

  df = df.dropna()
  
  # adjust scale if chronological age is to be used
  use_age = False
  if use_age:
    df["T"] = df["T"] + df["alder_tdato"] - df["t"] 
    df["t"] = df["alder_tdato"] 
    df = df.drop(columns=["alder_tdato"], axis=1)
    df_cols = [col for col in df_cols if col not in ["alder_tdato"]]
    cont_cols = [col for col in cont_cols if col not in ["alder_tdato"]]
  
  # split into data av event indicator
  delta = df["delta"].copy(deep=True)
  df = df.drop(["delta"], axis=1)
  
  # get parameters from best model chosen by grid search
  with open(f"best_mod_params_{data_name}_q_{q}_{get_vars}.csv") as csv_file:
    reader = csv.reader(csv_file)
    mydict = dict(reader)
  
  covariate_list = ast.literal_eval(mydict["covariate_list"])
  
  # get the coefficient paths for all regularization levels
  if q > 1:
    df_ = np.loadtxt(f"cox_mod_coefficients_path_{data_name}_q_{q}_{get_vars}.csv", delimiter=",")
    df_ = df_.reshape(df_.shape[0], 10, 50)
  else:
    df_ = pd.read_csv(f"cox_mod_coefficients_path_{data_name}_q_{q}_{get_vars}.csv")
  
  # get scores for all regularization levels
  scores = pd.read_csv(f"cox_mod_scores_path_{data_name}_q_{q}_{get_vars}.csv")
  scores = scores.drop("Unnamed: 0", axis = 1)
  
  # for constant coefficients plot full regularization curves
  if q == 1:
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16,12), dpi=500)
    df_ = df_.drop(columns="Unnamed: 0", axis=1)
    plot_df = df_.T.set_axis(covariate_list, axis=1)
    plot_df = plot_df.rename(columns={"alder_tdato": "age at donation"})
    colormap = plt.cm.nipy_spectral
    x = np.arange(0, 10)
    ax.plot(x, np.exp(plot_df.iloc[:, col]), label=covariate_list[col])

    labelLines(plt.gca().get_lines(), xvals=np.linspace(1, 9, 14))
    plt.xticks(np.arange(0,10),rotation = 45)
    ax.set_xticklabels(["$Least regularization$", "$1$", "$2$", "$3$", "$4$", "$5$", "$6$", "$7$", "$8$", "$Most regularization$"], size=18)
    plt.title("Coefficient Regularization paths for the KSCR\n on the Complete metadata group ", size=24)
    plt.ylabel("Coefficient size", size=20)
    plt.xlabel("Regularization parameter (Lambda) size ranking", size=20)
    
    plt.tight_layout()
    plt.savefig(f"{data_name}_q_{q}_path_{get_vars}.jpeg", dpi=300)
    plt.close()
  
  # adjust time to event for individual sample predictions
  df_new = df[["T", "t"]]
  df_new["tte"] = df_new["T"] - df_new["t"]
  df_new = df_new.drop(columns=["T", "t"], axis=1)
  
  # get models trained on each regularization level and calculate training data performance and number of non-zero coefficients
  for k in range(10):
    temp = list(scores)[k].replace(":", "")
    try:
      tvcox_pipeline = pickle.load(open(f"{data_name}_mod_q_{q}_{get_vars}_{temp}_{ts}.p", "rb"))
    except:
      tvcox_pipeline = pickle.load(open(f"{data_name}_path_mod_q_{q}_{get_vars}_{temp}.p", "rb"))

    current_gm = tvcox_pipeline.named_steps.tvcox_model.gamma_matrix
    num_cov = np.sum(np.linalg.norm(current_gm, axis = 1) != 0.0)
    print(f"{k}, {temp}: num non-zero cov: {num_cov}")

    pllh_score = tvcox_pipeline.score(df, delta)
    print(f"{k}, {temp}, pllh",pllh_score)
    
    tvcox_pipeline.named_steps.tvcox_model.set_params(**{"scoring_method": f"cum_dyn_auc_indv", "return_full_score": True})
    auc = tvcox_pipeline.score(df, delta)
    print(f"{k}:, {temp}:, auc:",auc)
    
    tvcox_pipeline.named_steps.tvcox_model.set_params(**{"scoring_method": f"c_index_indv"})
    c_index = tvcox_pipeline.score(df, delta)
    print(f"{k}:, {temp}:, c_index:",c_index)
  
  # stor list of indicies for zero coefficients for the chosen regularization level
  if choose_model:
    idx_list = []
    for idx in range(df_.shape[0]):
      if q == 1:
        if np.abs(df_.iloc[idx, mod_idx]) == 0.0:
          idx_list.append(idx)
      else:
        if np.mean(np.abs(df_[idx,mod_idx, :])) == 0.0:
          idx_list.append(idx)
  
    with open(f"{dg}_{mod_idx}_q_{q}_idx_list", "wb") as fp:
      pickle.dump(idx_list, fp)

if __name__ == "__main__":
  main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8])
