import sys
import numpy as np
import pandas as pd
from TVKCox_Regression import TVKCox_regression, Kernels
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import csv
import ast
from labellines import labelLines
import joblib
import matplotlib.pyplot as plt

plt.style.use('ggplot')

def cox_reg(filename, filename2, filename3, q: int, m: int):
    np.random.seed(13)
    q = int(q)
    m = int(m)
    N = 1000
 
    df = pd.read_csv(f"{filename}")
    df2 = pd.read_csv(f"{filename2}")
    
    covariate_list = df2["Unnamed: 0"].tolist()
    covariate_list = [col for col in covariate_list if col not in ["Cat_True", "X213", "event", "id", "T", "t"]]
    
    cat_cov =  []
    if "Cat_True" in df2["Unnamed: 0"].tolist():
       cat_cov.append("Cat_True")
    if "X213" in df2["Unnamed: 0"].tolist():
        cat_cov.append("X213")
    df = df[["id", "T", "t", "event"] + covariate_list + cat_cov]
    sample = pd.read_csv(f"{filename3}")
    sample = sample["id"].values.tolist()
 
    df_test = df[df["id"].isin(sample)]
    df = df[~df.index.isin(df_test.index)]
    delta = df["event"].copy()
    delta_test = df_test["event"].copy()
    df = df.drop(["event"], axis=1)
    df_test = df_test.drop(["event"], axis=1)

    tau = 100
    nu, phi, Lambda = 1e1, 1e2, 1e2
    ct = ColumnTransformer(
                     [("sc",StandardScaler(), covariate_list)],
                      remainder='passthrough',
                      verbose_feature_names_out = False
              )
    
    with open(f"inspection_bm_params.csv", "r") as f:
        r = csv.reader(f)
        params = {rows[0]: rows[1] for rows in r}
    
    params.pop("data")
    params.pop("gamma_matrix")
    params.pop("kernel")
    params.pop("optimizer")
    params.pop("spline_basis_functions")
    params.pop("scipy_tol")
    
    for key in params.keys():
        try:
            params[key] = ast.literal_eval(params[key])
        except:
            continue

    vals = {1: [0.5**1*nu, 0.5**1*phi, 0.5**1*Lambda], 2: [0.5**3*nu, 0.5**3*phi, 0.5**3*Lambda], 3: [0.5**5*nu, 0.5**5*phi, 0.5**5*Lambda],
            4: [0.5**7*nu, 0.5**7*phi, 0.5**7*Lambda], 5: [0.5**9*nu, 0.5**9*phi, 0.5**9*Lambda], 6: [0.5**10*nu, 0.5**10*phi, 0.5**10*Lambda],
            7: [0.5**11*nu, 0.5**11*phi, 0.5**11*Lambda], 8: [0.5**12*nu, 0.5**12*phi, 0.5**12*Lambda]}

    gm_path = np.zeros((len(covariate_list+cat_cov), len(vals)))
    scores = {}

    def tvcox_pipe(vals, key, params_dict:dict, p):
        tvcox = TVKCox_regression()
        tvcox.set_params(**params_dict)
        tvcox.set_params(**{"phi": vals[key][1], "Lambda": vals[key][2], "nu": vals[key][0], "scoring_method": "custom_cv_score_sparse"})
        tvcox_pipeline = Pipeline(
                [ ("scaler", ct),
        ("tvcox_model", tvcox)]
        )
        tvcox_pipeline.set_output(transform="pandas")
        tvcox_pipeline.fit(df, delta)
    
        res = [key, tvcox_pipeline.score(df_test, delta_test), tvcox_pipeline.named_steps.tvcox_model.gamma_matrix.reshape(p, )]
        
        return res


    result = joblib.Parallel(n_jobs=8)(joblib.delayed(tvcox_pipe)(vals, key, params, len(covariate_list+cat_cov)) for key in vals)
    result = sorted(result, key=lambda d: d[0])
    for mod in result:
        gm_path[:,len(vals)-mod[0]] = mod[2]
        scores[len(vals)-mod[0]] = mod[1]

    with open(f"score_paths.csv", "w") as f:
        w = csv.writer(f)
        for key, val in scores.items():
            w.writerow([key, val])

    gm_path = np.exp(gm_path)
    for col in cat_cov:
        covariate_list.append(f"{col}")

    for i in range(len(covariate_list)):
        a = covariate_list[i]
        if a == "Con_True_1":
            covariate_list[i] = r'$\beta_{1}$'
        elif a == "Con_True_2":
            covariate_list[i] = r'$\beta_{2}$'
        elif a == "Con_True_3_corr":
            covariate_list[i] = r'$\beta_{3}$'
        elif a == "Con_True_4_corr":
            covariate_list[i] = r'$\beta_{4}$'
        elif a == "Cat_True":
            covariate_list[i] = r'$\beta_{5}$'
        elif a == "corr_with_Con_True_3_corr_1":
            covariate_list[i] = 'Correlated noise 1'
        elif a == "corr_with_Con_True_3_corr_2":
            covariate_list[i] = 'Correlated noise 2' 
        elif a == "corr_with_Con_True_4_corr_1":
            covariate_list[i] = 'Correlated noise 3'
        elif a == "corr_with_Con_True_4_corr_2":
            covariate_list[i] = 'Correlated noise 4' 
        else:
            continue
    
    label_covariates = [r'$\beta_{1}$', r'$\beta_{2}$', r'$\beta_{3}$', r'$\beta_{4}$', r'$\beta_{5}$',
                        'Correlated noise 1', 'Correlated noise 2', 'Correlated noise 3', 'Correlated noise 4']
    for i in range(int(np.floor(gm_path.shape[0]/3))):
        if covariate_list[i] in label_covariates:
            plt.plot(gm_path[i,:], label = covariate_list[i])
        else:
            plt.plot(gm_path[i,:])

    #labelLines(plt.gca().get_lines(), zorder=2.5)
    #plt.savefig("random_inspection_coef_paths_1.jpeg", dpi=200)
    
    for i in range(int(np.floor(gm_path.shape[0]/3)), int(2*np.floor(gm_path.shape[0]/3))):
        if covariate_list[i] in label_covariates:
            plt.plot(gm_path[i,:], label = covariate_list[i])
        else:
            plt.plot(gm_path[i,:])

    #labelLines(plt.gca().get_lines(), zorder=2.5)
    #plt.savefig("random_inspection_coef_paths_2.jpeg", dpi=200)
    
    for i in range(int(2*np.floor(gm_path.shape[0]/3)), gm_path.shape[0]):
        if i >len(covariate_list)-1:
            continue
        else:
            if covariate_list[i] in label_covariates:
                plt.plot(gm_path[i,:], label = covariate_list[i])
            else:
                plt.plot(gm_path[i,:])

    labelLines(plt.gca().get_lines(), xvals=np.linspace(0, 5, 9))
    plt.xlabel("Regularization level")
    plt.ylabel("Coefficient size")
    plt.title("Regularization paths for KSCR model\n with constant coefficients on synthetic data")
    plt.tight_layout()
    plt.savefig("random_inspection_coef_paths.jpeg", dpi=200)

    return

if __name__ == "__main__":
    cox_reg(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
