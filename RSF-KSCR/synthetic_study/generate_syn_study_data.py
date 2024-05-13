import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from TVKCox_Regression.generate_synthetic_survdata import get_longitudinal_testdata
from scipy.stats import norm

def zero_t(df_: pd.DataFrame):
        df = df_.copy()
        id = df.at[0, "id"]
        t_min = df.at[0, "t"]
        for row in range(1, df.shape[0]):
            if df.at[row, "id"] == id:
                df.at[row, "t"] = df.at[row, "t"] - t_min
                df.at[row, "T"] = df.at[row, "T"] - t_min
            else:
                id = df.at[row, "id"]
                t_min = df.at[row, "t"]
                df.at[row, "t"] = df.at[row, "t"] - t_min
                df.at[row, "T"] = df.at[row, "T"] - t_min
        return df

def beta_func1(x, type:str, T:int=120):
    if type == "weak":
        return 1.3*np.exp(-10*((x/T)-0.1)**2) + 1
    else:
        return 2.5*np.exp(-10*((x/T)-0.15)**2) + 1

def beta_func2(x, type:str, T:int=120):
    if type == "weak":
        return 1.0 + 0.5*(x/T)
    else:
        return 1.0 + 0.8*(x/T)
    
def beta_D(x, type:str, T:int=120):
    if type == "weak":
        return 1.5 - 0.5*(x/T)
    else:
        return 2.5 - 1.0*(x/T)

# corr func
def beta_corr_func1(x, type:str, T:int=120):
    if type == "weak":
        return 1.12 - 0.16*(x/T)
    else:
        return 1.2 - 0.35*(x/T)

def beta_corr_func2(x, type:str, T:int=120):
    if type == "weak":
        return 0.7 + 0.3*(x/T)
    else:
        return 0.40 + 0.6*(x/T)

def beta_corr_func3(x, type:str, T:int=120):
    if type == "weak":
        return 1 + (-20*(norm.pdf(x, loc=0.77*T, scale=0.29*T)))
    else:
        return 1 + (-110*(norm.pdf(x, loc=0.77*T, scale=0.29*T)))

def beta_corr_func4(x, type:str, T:int=120):
    if type == "weak":
        return 1 + (15*(norm.pdf(x, loc=0.77*T, scale=0.24*T))) # 65
    else:
        return 1 + (30*(norm.pdf(x, loc=0.77*T, scale=0.24*T))) # 90

def beta_corr_func5(x, type:str, T:int=120):
    if type == "weak":
        return 1 + (30*(norm.pdf(x, loc=0.425*T, scale=0.192*T)))
    else:
        return 1 + (45*(norm.pdf(x, loc=0.425*T, scale=0.192*T)))

def main(run:int=0, T:int=100, N:int=1000, ncc:bool = True, nnc_controls_per_case:int=1, show_plots:bool=False, save_file:bool=False):
    # set seed
    np.random.seed(32)
    
    run = int(run)
    T = int(T)
    N = int(N)
    nnc_controls_per_case = int(nnc_controls_per_case)
    ncc = True
    save_file = bool(save_file)

    x_arr = np.arange(1,T+1).reshape((T, 1))

    # strong coefficients 
    HR_true = beta_func1(x_arr, "strong", T)
    HR_true = np.append(HR_true, beta_corr_func2(x_arr, "strong", T), axis=1)

    for i in range(203): 
        HR_true = np.append(HR_true, np.ones((T,1)), axis=1)

    HR_true = np.append(HR_true, np.ones((T,1)), axis=1) 
    HR_true = np.append(HR_true, np.ones((T,1)) * 1.5, axis=1) 
    HR_true = np.append(HR_true, np.ones((T,1)), axis=1) 

    HR_true = np.append(HR_true, np.ones((T,1)), axis=1) 
    HR_true = np.append(HR_true, np.ones((T,1))*1.5, axis=1) 
    HR_true = np.append(HR_true, np.ones((T,1)), axis=1) 

    HR_true = np.append(HR_true, beta_D(x_arr, "strong", T), axis=1)
    HR_true = np.append(HR_true, np.ones((T,1)), axis=1) 

    # create sigma
    sigma = np.random.uniform(.55, .68, 9).reshape((3,3))
    sigma = (sigma * sigma.T ) / 2
    sigma = sigma * (np.ones((3,3)) - np.eye(3)) + np.eye(3)

    n_dim = 3
    n_batch = 37
    n_batch_ = 100
    mu = np.ones(n_dim)*0
    mu_ = 0.0
    tau = np.eye(n_dim)*0.5
    sigma_ = 1
    tau_ = 0.5
    dt = 1
    discrete_cov = {"var1":[[0,1], [0.5,0.5]], "var2": [[0,1], [0.5,0.5]]}
    param_dict = {"dt": dt, "mu": mu_, "sigma": sigma_, "tau": tau_, "n_batch": n_batch_}
    corr_params_dict = {"dt": dt, "mu": mu, "sigma": sigma, "tau": tau, "n_dim": n_dim, "n_batch": n_batch}
    
    survdata, X = get_longitudinal_testdata(N = N,
                                    T = T,
                                    p_censored = 0.4,
                                    sparse = True,
                                    time_varying = True,
                                    sparse_mean_num_measurements = 3,
                                    HR_true = HR_true,
                                    randstate=run,
                                    random_baseline_surv = True,
                                    ncc = ncc,
                                    data_generation_params=param_dict,
                                    corr_data_generation_params=corr_params_dict,
                                    discrete_cov=discrete_cov)
    survdata = survdata.rename(columns={"time": "T"})
    survdata["id"] = survdata.index

    data = X.set_index("id").join(survdata.set_index("id"), on="id").rename(columns={"time": "t"}).reset_index()
    data = data.rename(columns={"X1": "Con_True_1", "X2": "Con_True_2", "X13": "corr_with_Con_True_3_corr_1", "X15": "corr_with_Con_True_3_corr_2",
    "X14": "Con_True_3_corr", "X17": "Con_True_4_corr", "X16": "corr_with_Con_True_4_corr_1", "X18": "corr_with_Con_True_4_corr_2",
    "X19": "Cat_True"})


    if ncc:
        df_cases = data[data["event"] == 1]

        fraction = df_cases["id"].unique().shape[0] / data["id"].unique().shape[0]
        df_controls = data[data["event"] == 0].groupby("id").first()[["Cat_True", "X20"]].sample(frac = fraction*nnc_controls_per_case, random_state = run+29).reset_index()
        df_controls = data[data["id"].isin(df_controls["id"])]
        data_ncc = pd.concat([df_cases, df_controls])
        
        df = data_ncc.reset_index().drop("index", axis=1)
    else:
        df = data

    if save_file:
        df = zero_t(df)
        df.to_csv(f"data_sets/data_files/run_{run}_df_strong_coef_t0.csv", index=False)

        df_full_ = zero_t(data)
        df_full_.to_csv(f"data_sets/data_files/run_{run}_df_strong_coef_full_t0.csv", index=False)

    if show_plots == "True":
        df["t"].hist(bins=50)
        plt.show()

        df.groupby("id").first()["T"].hist(bins=50)
        plt.show()

        df.groupby("id").first()[["T", "Cat_True"]].groupby("Cat_True")["T"].hist(bins=50, legend=True, alpha=0.5)
        plt.show()

        df.groupby("id").first()[["T", "event"]].groupby("event")["T"].hist(bins=50, legend=True, alpha=0.5)
        plt.show()
        print(f"Proportion of event and non event subjects in total population: ", df_cases.groupby("id").first()["T"].shape[0] / data.groupby("id").first()["T"].shape[0])
        print(f"Number of event samples and non event samples: ", df["event"].value_counts())
        print(f"Proportion of event and non event subjects: ", df.groupby("id").first()["event"].value_counts() / df.groupby("id").first()["event"].value_counts().sum())
        print(f"Proportion of event and non event samples: ", df["event"].value_counts() / df["event"].value_counts().sum())
        print(f"Proportion of Cat_True for cases and controls: ", df.groupby("id").first()[["Cat_True", "event"]].groupby("event")["Cat_True"].sum() / df.groupby("id").first().groupby("event")["Cat_True"].count())
        print(f"Number of samples total: ", df.shape[0])
    
    
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])