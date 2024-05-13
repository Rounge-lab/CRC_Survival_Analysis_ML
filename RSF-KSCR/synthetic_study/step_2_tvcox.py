import sys
import numpy as np
import pandas as pd
from TVKCox_Regression import TVKCox_regression, Kernels
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from pathlib import Path

def cox_reg(filename, filename2, filename3, q: int, m: int, n_jobs:int, run:int):
    
    run = int(run)
    np.random.seed(run)
    
    q = int(q)
    m = int(m)
    n_jobs = int(n_jobs)
    data_name = filename[:-4]
    

    root = Path('data_sets', 'data_files')
    root2 = Path('data_sets', 'covars')
    df = pd.read_csv(root / f"{filename}")
    if filename == filename2:
        covariate_list = df.columns.tolist()
        covariate_list = [col for col in covariate_list if col not in ["id", "T", "t", "event", "Cat_True", "X213"]]
        cat_cov = ["Cat_True", "X213"]
    else:
        df2 = pd.read_csv(root2 / f"{filename2}")
        covariate_list = df2["Unnamed: 0"].tolist()
        covariate_list = [col for col in covariate_list if col not in ["Cat_True", "X213"]]

        cat_cov = []
        if "Cat_True" in df2["Unnamed: 0"].tolist():
            cat_cov.append("Cat_True")
        if "X213" in df2["Unnamed: 0"].tolist():
            cat_cov.append("X213")
    
    df = df[["id", "T", "t", "event"] + covariate_list + cat_cov]
    
    sample = pd.read_csv(root / f"{filename3}")
    sample = sample["id"].values.tolist()
 
    df_test = df[df["id"].isin(sample)]
    df = df[~df.index.isin(df_test.index)]
    delta = df["event"].copy()
    delta_test = df_test["event"].copy()
    df = df.drop(["event"], axis=1)
    df_test = df_test.drop(["event"], axis=1)

    tau = 100
    ct = ColumnTransformer(
                     [("sc",StandardScaler(), covariate_list)],
                      remainder='passthrough',
                      verbose_feature_names_out = False
              )

    # make k-fold cv sets based on unique ids
    df["cvLabel"] = np.zeros(df.shape[0])
    df["delta"] = delta
    df_ids = df[["id", "T", "delta"]].groupby(["id","T", "delta"]).first().reset_index().copy()
    k = 6
    for i in range(k):
        samp = df_ids.sample(frac = 1/float(k))["id"].tolist()
        df_ids = df_ids.drop(df_ids[df_ids["id"].isin(samp)].index)
        df["cvLabel"][df["id"].isin(samp)] = i
    
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
    nu = 1e1
    phi, Lambda = 1e2, 1e2
    tvcox = TVKCox_regression()
    tvcox.set_params(**{"tau": tau, "q": q, "m": m, "opt_type": "GroupedDescent", "phi": 0.5**15*phi, "Lambda": 0.5**15*Lambda, "alpha": 1.0,
                        "nu": 0.5**15*nu,"network_penalty": False, "tol": 1.0e-4, "verbose": 1, "t": 0.05, "scoring_method": "pllh", 
                        "kernel": Kernels.Epanechnikov, "w_threshold": 0.2, "random_start_variance": nu/phi, "rand_state": run, 
                        "opt_algorithm": "BFGS", "random_descent_cycle": True, "surrogate_likelihood": True, "penalty_type": "L0", "track_pllh": False})
    
    tvcox_pipeline = Pipeline(
            [ ("scaler", ct),
    ("tvcox_model", tvcox)]
    )
    tvcox_pipeline.set_output(transform="pandas")
    def generate_bw(C, df: pd.DataFrame, tau):
        return list(tau*C*df.groupby("id").first().shape[0]**(-(1/8)))
    
    bw_list = generate_bw(np.arange(1,24,2)*0.05, df, tau)
    params1 = {"tvcox_model__bw": bw_list} 
    
    
    clf1 = GridSearchCV(tvcox_pipeline, params1, n_jobs=n_jobs, cv=myCViterator, verbose=10, refit=True)
    clf1.fit(df, delta)
    results = pd.DataFrame(clf1.cv_results_)
    res_1 =results[['param_tvcox_model__bw', "mean_test_score"]].sort_values(by="param_tvcox_model__bw", ascending=False)
    res_2 = results[['param_tvcox_model__bw', "mean_test_score"]].sort_values(by="param_tvcox_model__bw", ascending=False).shift(1)
    res_1["diff"] = (res_1.iloc[1:, 1]- res_2.iloc[:-1, 1])< 0.0002
    res_1["diff_val"] = (res_1.iloc[1:, 1]- res_2.iloc[:-1, 1])
    bw_use = res_1[res_1["diff"] == True]["param_tvcox_model__bw"].min()

    multiplier_1 = np.repeat(0.5, 6)
    multiplier_2 = np.array([1, 3, 5, 7, 9, 14])
    multiplier = multiplier_1**multiplier_2

    tvcox_pipeline.named_steps.tvcox_model.set_params(**{"network_penalty": True, "bw": bw_use, "scoring_method": "custom_cv_score_sparse"})
    params2 = [{"tvcox_model__Lambda": [multiplier[0]*Lambda], "tvcox_model__phi": [multiplier[0]*phi], "tvcox_model__nu": [multiplier[0]*nu], "tvcox_model__alpha": [1.0, 0.5], "tvcox_model__bw": bw_list},
                {"tvcox_model__Lambda": [multiplier[1]*Lambda], "tvcox_model__phi": [multiplier[1]*phi], "tvcox_model__nu": [multiplier[1]*nu], "tvcox_model__alpha": [1.0, 0.5], "tvcox_model__bw": bw_list}, 
                {"tvcox_model__Lambda": [multiplier[2]*Lambda], "tvcox_model__phi": [multiplier[2]*phi], "tvcox_model__nu": [multiplier[2]*nu], "tvcox_model__alpha": [1.0, 0.5], "tvcox_model__bw": bw_list},
                {"tvcox_model__Lambda": [multiplier[3]*Lambda], "tvcox_model__phi": [multiplier[3]*phi], "tvcox_model__nu": [multiplier[3]*nu], "tvcox_model__alpha": [1.0, 0.5], "tvcox_model__bw": bw_list},
                {"tvcox_model__Lambda": [multiplier[4]*Lambda], "tvcox_model__phi": [multiplier[4]*phi], "tvcox_model__nu": [multiplier[4]*nu], "tvcox_model__alpha": [1.0, 0.5], "tvcox_model__bw": bw_list},
                {"tvcox_model__Lambda": [multiplier[5]*Lambda], "tvcox_model__phi": [multiplier[5]*phi], "tvcox_model__nu": [multiplier[5]*nu], "tvcox_model__alpha": [1.0, 0.5], "tvcox_model__bw": bw_list}]
    
    
    clf2 = GridSearchCV(tvcox_pipeline, params2, n_jobs=n_jobs, cv=myCViterator, verbose=10, refit=True)
    clf2.fit(df, delta)
    final_model = clf2.best_estimator_

    final_model.named_steps.tvcox_model.set_params(**{"scoring_method": "c_index_indv", "return_full_score": True})
    fm_c_idx = final_model.score(df_test, delta_test)
    
    final_model.named_steps.tvcox_model.set_params(**{"scoring_method": "c_index_avg"})
    fm_c_avg_idx = final_model.score(df_test, delta_test)
    final_model.named_steps.tvcox_model.set_params(**{"scoring_method": "cum_dyn_auc_indv"})
    fm_auc = final_model.score(df_test, delta_test)
    final_model.named_steps.tvcox_model.set_params(**{"scoring_method": "cum_dyn_auc_avg"})
    fm_avg_auc = final_model.score(df_test, delta_test)   

    n = len(covariate_list + cat_cov)
    cov_list = covariate_list + cat_cov
    true_cov = ["Con_True_1", "Con_True_2", "Con_True_3", "Con_True_4", "Cat_True"]
    gamma_mat = final_model.named_steps.tvcox_model.gamma_matrix    
    tp_no_ci, tn_no_ci, fp_no_ci, fn_no_ci = 0,0,0,0
    for i in range(len(cov_list)):
        if q == 1:
            if np.abs(gamma_mat[i]) > 1e-3:
                    if cov_list[i] in true_cov:
                        tp_no_ci += 1
                    else:
                        fp_no_ci += 1
            else:
                if cov_list[i] in true_cov:
                    fn_no_ci += 1
                else:
                    tn_no_ci += 1
        else:
            if np.abs(np.mean(gamma_mat[i,:])) > 1e-2:
                if cov_list[i] in true_cov:
                    tp_no_ci += 1
                else:
                    fp_no_ci += 1
            else:
                if cov_list[i] in true_cov:
                    fn_no_ci += 1
                else:
                    tn_no_ci += 1
   

    df_c_index = pd.DataFrame({f"c_index_avg_q_{q}_m_{m}": [fm_c_avg_idx]})
    auc_cols = [f"auc_avg_q_{q}_m_{m}_{10*i}" for i in range(1,fm_avg_auc.shape[0]+1)]
    df_auc = pd.DataFrame({f"{auc_cols[i]}": [fm_avg_auc[i]] for i in range(fm_avg_auc.shape[0])})

    df_c_index_indv = pd.DataFrame({f"c_index_indv_q_{q}_m_{m}": [fm_c_idx]})
    auc_cols_indv = [f"auc_indv_q_{q}_m_{m}_{i}" for i in [5,10,20,30,40,50]]
    df_auc_indv = pd.DataFrame({f"{auc_cols_indv[i]}": [fm_auc[i]] for i in range(fm_auc.shape[0])})

    coefs = [tp_no_ci, tn_no_ci, fp_no_ci, fn_no_ci, n]
    coef_names = [f"tp_no_ci_q_{q}_m_{m}", 
                  f"tn_no_ci_q_{q}_m_{m}", f"fp_no_ci_q_{q}_m_{m}", f"fn_no_ci_q_{q}_m_{m}", f"num_cov_q_{q}_m_{m}"]

    df_coefs = pd.DataFrame({f"{coef_names[i]}": [coefs[i]] for i in range(len(coefs))})

    df_c_index.to_csv(f"results/final/{data_name}_cox_c_index_avg_q_{q}_m_{m}.csv", index=False)
    df_auc.to_csv(f"results/final/{data_name}_cox_auc_avg_q_{q}_m_{m}.csv", index=False)
    df_c_index_indv.to_csv(f"results/final/{data_name}_cox_c_index_indv_q_{q}_m_{m}.csv", index=False)
    df_auc_indv.to_csv(f"results/final/{data_name}_cox_auc_indv_q_{q}_m_{m}.csv", index=False)
    df_coefs.to_csv(f"results/final/{data_name}_cox_coefs_q_{q}_m_{m}.csv", index=False)

    
    return

if __name__ == "__main__":
    cox_reg(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])