import os
import sys

import pandas as pd
import numpy as np
import csv
import joblib
import pickle
import time

from TVKCox_Regression import TVKCox_regression, Kernels
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV

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

def main(data_name:str, time_scale:str, n_jobs:int, q:int, m:int, get_vars:str, bw:float, k:int, seed:int=91):
  """
  Performs a grid search to find optimal parameters for a Kernel Smoothed Cox Regression model.
  Fits and saves model piepline for each regularization level using the optimal parameters 
  found during gridsearch.

        Parameters
        ----------
        data_name : str
            File path to the dataset used
        time_scale : str
            Which time scale to use (reg_time->time-to-diagnosis or age->chornological age)
        n_jobs : int
            Number of cores to use.
        q : int
            The B-spline coefficent number (1-> constant)
        m : int
            The B-spline order (1-> constant, 2->linear, 3->quadratic ...)
        get_vars : str
            Indicator of which selected covariates to use (md_sel->from minimal depth thresholding, 
            stepwise-> selected using the stepwise procedure)
        bw : float
            The desiered bandwidth, if 0 estimates optimal kernel bandwidth.
        k : int
            Number of folds to use during cross validation
        seed : int
            Random seed number.

        Returns
        -------
        None
  """

  np.random.seed(int(seed))
  
  q = int(q)
  m = int(m)
  k = int(k)  
  n_jobs = int(n_jobs)
  bw = float(bw)
  
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
    df_cv = pd.read_csv(f"path")["Unnamed: 0"].tolist()
    df_cv = fix_names(df_cv)
  
  elif get_vars == "stepwise":
    
    if data_name == "name1":
      df_cv = pd.read_csv(f"name1")
    elif data_name == "name2":
      df_cv = pd.read_csv(f"name2")
    else:
      df_cv = pd.read_csv(f"name3")
    
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
  if data_name == "name3":
    df_cols = miRNA_cols + ["sex", "alder_tdato", "bmi", "smoking_status"] + ["id", "T", "t", "delta"]
    df = df[df_cols]
    df.loc[:, "sex"] = (df["sex"] == "M").astype("int")
    df.loc[:, "smoking_status"] = (df["smoking_status"] == "smoker").astype("int")
    cont_cols = [col for col in df_cols if col not in ["id", "T", "t", "delta", "sex", "smoking_status"]]
    num_cat_cols = 2
  else:
    df_cols = miRNA_cols + ["sex", "alder_tdato"] + ["id", "T", "t", "delta"]
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
  
  # Get study timeframe
  tau = df["T"].max()
  
  # Create scaler
  ct = ColumnTransformer(
                       [("sc",StandardScaler(), cont_cols)],
                        remainder='passthrough',
                        verbose_feature_names_out = False
                )
  
  
  # make k-fold cv sets based on unique ids
  df["cvLabel"] = np.zeros(df.shape[0])
  delta = df.delta.copy(deep=True)
  df_ids = df[["id", "T", "delta", "sex", "alder_tdato"]].groupby(["id", "T", "delta", "sex", "alder_tdato"]).agg({'id':'first', 'T':'first', 'delta':'first', 'sex':'first', 'alder_tdato':'mean'})
  for i in range(k):
      samp = df_ids.sample(frac = 1/float(k), random_state=745)["id"].tolist()
      df_ids = df_ids.drop(df_ids[df_ids["id"].isin(samp)].index)
      df.loc[:,"cvLabel"][df["id"].isin(samp)] = i

  dist = df.groupby("cvLabel").count()["id"].to_numpy()
  dist = dist / len(dist)
  
  if np.any(dist < (.8/ k)):
      raise ValueError("The distribution of the cvLabels is too skewd, try a different k")
   
  myCViterator = []
  df = df.reset_index()
  for i in range(k):
      trainIndices = df[ df['cvLabel']!=i ].index.values.astype(int)
      testIndices =  df[ df['cvLabel']==i ].index.values.astype(int)
      myCViterator.append( (trainIndices, testIndices) )
  
  df = df.drop(["cvLabel", "index", "delta"], axis=1)
  
  # select base regularization level and hard threshold
  if q == 1:
    nu = 2e1
    phi, Lambda = 1e2, 1e2
  elif q == 2:
    nu = 2.82e2
    phi, Lambda = 1e3, 1e3 
  else:
    nu = 4e3
    phi, Lambda = 1e4, 1e4
  
  # Initiate model pipeline
  params = {"tau": tau,"q": q, "m": m,"iter_limit": 40, "opt_type": "GroupedDescent", "phi": 0.5**18*phi, "Lambda": 0.5**18*Lambda, "alpha": 1.0,
                      "nu": 0.5**18*nu,"network_penalty": False, "tol": 1.0e-4, "verbose": 0, "t": 0.009, "scoring_method": "pllh", 
                      "kernel": Kernels.Epanechnikov, "w_threshold": 0.3, "random_start_variance": nu/phi, "rand_state": int(seed),
                      "opt_algorithm": "BFGS", "random_descent_cycle": True, "surrogate_likelihood": True, "penalty_type": "L0", "track_pllh": False}

  tvcox = TVKCox_regression()
  tvcox.set_params(**params)
 
  tvcox_pipeline = Pipeline(
          [ ("scaler", ct),
  ("tvcox_model", tvcox)]
  )
  tvcox_pipeline.set_output(transform="pandas")
  
  # Select optimal bandwidth
  if bw == 0:
  
    def generate_bw(C, df: pd.DataFrame, tau):
        return list(np.round(tau*C*df.groupby("id").first().shape[0]**(-(1/8)), 3))
    
    bw_list = generate_bw(np.arange(2,24,3)*0.05, df, tau)
    
    bw_params = {}
    for i in range(0, len(bw_list)):
      bw_params[i+1] = bw_list[i]
    
    key_list = list(np.repeat([key for key in bw_params.keys()], k))
    k_list = list(np.tile(np.arange(1,k), len(bw_params.keys())))
    idx_list = zip(key_list, k_list) 

    def tvcox_pipe(key, it, vals, params_dict, cv_it):
      print(f"[CV {it}/{k}; {key}/{len(vals.keys())}] START; bw={vals[key]}")
      t0 = time.perf_counter()
      tvcox = TVKCox_regression()
      tvcox.set_params(**params_dict)
      tvcox.set_params(**{"bw": vals[key]})
      tvcox_pipeline = Pipeline(
              [ ("scaler", ct),
      ("tvcox_model", tvcox)]
      )
      tvcox_pipeline.set_output(transform="pandas")
      
      df_tr = df.iloc[cv_it[it][0],]
      df_ts = df.iloc[cv_it[it][1],]

      delta_tr = delta.iloc[cv_it[it][0]]
      delta_ts = delta.iloc[cv_it[it][1]]

      tvcox_pipeline.fit(df_tr, delta_tr)
      tvcox_pipeline.named_steps.tvcox_model.set_params(**{"scoring_method": "pllh"})
      full_pllh = tvcox_pipeline.score(df, delta)
      leave_out_pllh = tvcox_pipeline.score(df_tr, delta_tr)
      score = full_pllh - leave_out_pllh
      
      t1 = time.perf_counter()
      print(f"[CV {it}/{k}; {key}/{len(vals.keys())}] END; time:{(t1-t0)/60} minutes; score={score}; bw={vals[key]}")
      res = [key, it, score]
        
      return res
  
 
    result = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(tvcox_pipe)(key, it, bw_params, params, myCViterator) for key, it in idx_list)
    result = sorted(result, key=lambda d: d[0])

    bw_array = np.zeros((len(bw_params.keys()), k))

    for res in result:
      bw_array[res[0],res[1]] = res[2]
      
    
    bw_sum = np.diff(np.mean(bw_array, axis=1)) 
    bw_idx = np.argmax(np.abs(bw_sum) < 0.05) + 1
    bw_use = bw_list[bw_idx]

    tvcox_pipeline.named_steps.tvcox_model.set_params(**{"network_penalty": True, "bw": bw_use, "scoring_method": "pllh"})
  
  # or use specified bandwidth   
  else:
    tvcox_pipeline.named_steps.tvcox_model.set_params(**{"network_penalty": True, "bw": bw, "scoring_method": "pllh"})
  
  # multipliers to reduce size of regularization at each step
  multiplier_1 = np.repeat(0.5, 6)
  multiplier_2 = np.array([1, 3, 5, 7, 9, 14])
  multiplier = multiplier_1**multiplier_2
  
  # Create search grid
  vals = {1:list(multiplier[0]*np.array([nu, phi, Lambda, 1.0])) , 2:list(multiplier[1]*np.array([nu, phi, Lambda, 1.0])), 3:list(multiplier[2]*np.array([nu, phi, Lambda, 1.0])), 
  4:list(multiplier[3]*np.array([nu, phi, Lambda, 1.0])), 5:list(multiplier[4]*np.array([nu, phi, Lambda, 1.0])), 6:list(multiplier[5]*np.array([nu, phi, Lambda, 1.0])),
  7:list(multiplier[0]*np.array([nu, phi, Lambda, 0.7])) , 8:list(multiplier[1]*np.array([nu, phi, Lambda, 0.7])), 9:list(multiplier[2]*np.array([nu, phi, Lambda, 0.7])), 
  10:list(multiplier[3]*np.array([nu, phi, Lambda, 0.7])), 11:list(multiplier[4]*np.array([nu, phi, Lambda, 0.7])), 12:list(multiplier[5]*np.array([nu, phi, Lambda, 0.7])),
  13:list(multiplier[0]*np.array([nu, phi, Lambda, 0.5])) , 14:list(multiplier[1]*np.array([nu, phi, Lambda, 0.5])), 15:list(multiplier[2]*np.array([nu, phi, Lambda, 0.5])), 
  16:list(multiplier[3]*np.array([nu, phi, Lambda, 0.5])), 17:list(multiplier[4]*np.array([nu, phi, Lambda, 0.5])), 18:list(multiplier[5]*np.array([nu, phi, Lambda, 0.5])),
  19:list(multiplier[0]*np.array([nu, phi, Lambda, 0.2])) , 20:list(multiplier[1]*np.array([nu, phi, Lambda, 0.2])), 21:list(multiplier[2]*np.array([nu, phi, Lambda, 0.2])), 
  22:list(multiplier[3]*np.array([nu, phi, Lambda, 0.2])), 23:list(multiplier[4]*np.array([nu, phi, Lambda, 0.2])), 24:list(multiplier[5]*np.array([nu, phi, Lambda, 0.2]))}  
  
  # list to tell each fold and which data and parameters to use
  key_list = list(np.repeat([key for key in vals.keys()], k))
  k_list = list(np.tile(np.arange(1,k), len(vals.keys())))
  idx_list = zip(key_list, k_list) 
  
  # grid search
  def tvcox_pipe(key, it, vals, params_dict, cv_it):
    print(f"[CV {it}/{k}; {key}/{len(vals.keys())}] START; nu={vals[key][0]}, alpha={vals[key][3]}")
    t0 = time.perf_counter()
    # Initiate model with chosen and grid parameters
    tvcox = TVKCox_regression()
    tvcox.set_params(**params_dict)
    tvcox.set_params(**{"phi": vals[key][1], "Lambda": vals[key][2], "alpha": vals[key][3], "nu": vals[key][0]})
    tvcox_pipeline = Pipeline(
            [ ("scaler", ct),
    ("tvcox_model", tvcox)]
    )
    tvcox_pipeline.set_output(transform="pandas")
    
    # Split data into training and test fold
    df_tr = df.iloc[cv_it[it][0],]
    df_ts = df.iloc[cv_it[it][1],]

    delta_tr = delta.iloc[cv_it[it][0]]
    delta_ts = delta.iloc[cv_it[it][1]]
    
    # Fit model
    tvcox_pipeline.fit(df_tr, delta_tr)
    tvcox_pipeline.named_steps.tvcox_model.set_params(**{"scoring_method": "pllh"})
    
    # Calculate the change in PLLH when the test fold is removed
    full_pllh = tvcox_pipeline.score(df, delta)
    leave_out_pllh = tvcox_pipeline.score(df_tr, delta_tr)
    score = full_pllh - leave_out_pllh
    
    t1 = time.perf_counter()
    print(f"[CV {it}/{k}; {key}/{len(vals.keys())}] END; time:{(t1-t0)/60} minutes; score={score}; nu={vals[key][0]}, alpha={vals[key][3]}")
    res = [key, it, score]
      
    return res
  
 
  result = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(tvcox_pipe)(key, it, vals, params, myCViterator) for key, it in idx_list)
  result = sorted(result, key=lambda d: d[0])
  
  # Get and store the test scores
  score_array = np.zeros((len(vals.keys()), k))

  for res in result:
    score_array[res[0],res[1]] = res[2]

  np.savetxt(f"CV_score_{data_name}_q_{q}_{get_vars}_{time.strftime('%Y%m%d')}.csv", score_array, delimiter=",")
  
  # Get the best parameters
  best_idx = np.argmax(np.sum(score_array, axis=1))
  best_vals = vals[best_idx]

  bm_params = {"tau": tau,"q": q, "m": m,"iter_limit": 40, "opt_type": "GroupedDescent", "phi": best_vals[1], "Lambda": best_vals[2], "alpha": best_vals[3],
                      "nu": best_vals[0],"network_penalty": True, "tol": 1.0e-4, "verbose": 1, "t": 0.009, "scoring_method": "pllh", 
                      "kernel": Kernels.Epanechnikov, "w_threshold": 0.3, "random_start_variance": nu/phi, "rand_state": 2,
                      "opt_algorithm": "BFGS", "random_descent_cycle": False, "surrogate_likelihood": True, "penalty_type": "L0", "track_pllh": False}
  
  # Save best parameters
  with open(f"best_mod_params_{data_name}_q_{q}_{get_vars}.csv", "w") as f:
    writer = csv.writer(f)
    for key, value in bm_params.items():
      writer.writerow([key, value])
  
  multiplier_1 = np.repeat(0.5, 10)
  multiplier_2 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 10])
  multiplier = multiplier_1**multiplier_2

  # Regularization path
  vals = {1: list(multiplier[0]*np.array([nu, phi, Lambda])), 2: list(multiplier[1]*np.array([nu, phi, Lambda])), 3: list(multiplier[2]*np.array([nu, phi, Lambda])),
        4: list(multiplier[3]*np.array([nu, phi, Lambda])), 5: list(multiplier[4]*np.array([nu, phi, Lambda])), 6: list(multiplier[5]*np.array([nu, phi, Lambda])),
        7: list(multiplier[6]*np.array([nu, phi, Lambda])), 8: list(multiplier[7]*np.array([nu, phi, Lambda])), 9: list(multiplier[8]*np.array([nu, phi, Lambda])),
        10: list(multiplier[9]*np.array([nu, phi, Lambda]))}
  
  # Hold coefficient paths 
  if q == 1:
    gm_path = np.zeros((len(cont_cols)+num_cat_cols, len(vals)))
  else:
    gm_path = np.zeros((len(cont_cols)+num_cat_cols, len(vals), 50))
  
  scores = [x for x in range(0,len(vals))]
  col_names = [f"nu_{i}" for i in list(reversed((multiplier*nu).tolist()))]
  scores = pd.DataFrame(data=[scores], columns=col_names)
  scores.to_csv(f"cox_mod_scores_path_{data_name}_q_{q}_{get_vars}.csv")
  
  # Parallell calculation of coefficient paths for each regularization level
  def tvcox_pipe(vals, key, params_dict:dict, p):
      tvcox = TVKCox_regression()
      tvcox.set_params(**params_dict)
      tvcox.set_params(**{"phi": vals[key][1], "Lambda": vals[key][2], "nu": vals[key][0]})
      tvcox_pipeline = Pipeline(
              [ ("scaler", ct),
      ("tvcox_model", tvcox)]
      )
      tvcox_pipeline.set_output(transform="pandas")
      tvcox_pipeline.fit(df, delta)
      pickle.dump(tvcox_pipeline, open(f"{data_name}_mod_q_{q}_{get_vars}_nu_{vals[key][0]}_{time.strftime('%Y%m%d')}.p", "wb"))
      if q == 1:
        res = [key, tvcox_pipeline.named_steps.tvcox_model.gamma_matrix.reshape(p, )]
      
      else:
        basis_func_values = np.zeros((50, q))
        time_points = np.linspace(0, tau, 50)
        
        for func in range(q):
          basis_func_values[:,func] = tvcox_pipeline.named_steps.tvcox_model.spline_basis_functions[func](time_points)
        
        paths = np.zeros((p, 50))
        
        for i in range(50):
          paths[:,i] =  tvcox_pipeline.named_steps.tvcox_model.gamma_matrix @ basis_func_values[i,:].T
        
        res = [key, paths]
      
      return res
  
 
  result = joblib.Parallel(n_jobs=10)(joblib.delayed(tvcox_pipe)(vals, key, bm_params, len(cont_cols)+num_cat_cols) for key in vals)
  result = sorted(result, key=lambda d: d[0])
  for mod in result:
      if q == 1:
        gm_path[:,len(vals)-mod[0]] = mod[1]
      else:
        gm_path[:,len(vals)-mod[0],:] = mod[1]
  
  # Save coefficent paths for each regularization level
  if q == 1:
    gammas = pd.DataFrame(data=gm_path, columns=col_names)
    gammas.to_csv(f"cox_mod_coefficients_path_{data_name}_q_{q}_{get_vars}.csv")
  else:
    gm_path = gm_path.reshape(gm_path.shape[0], -1)
    np.savetxt(f"cox_mod_coefficients_path_{data_name}_q_{q}_{get_vars}.csv", gm_path, delimiter=",")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9])
