import numpy as np
import pandas as pd
import sys
import os

def create_df(file_name:str, data_name:str, q:int, m:int,num_runs_lower:int, num_runs:int):
    df = pd.DataFrame()
    for run in range(num_runs_lower, num_runs+1):
        df_ = pd.read_csv(f"results/final/run_{run}_{data_name}_cox_{file_name}_q_{q}_m_{m}.csv")
        df = pd.concat([df, df_])
        if(os.path.exists(f"results/final/run_{run}_{data_name}_cox_{file_name}_q_{q}_m_{m}.csv") and os.path.isfile(f"results/final/run_{run}_{data_name}_cox_{file_name}_q_{q}_m_{m}.csv")):
            os.remove(f"results/final/run_{run}_{data_name}_cox_{file_name}_q_{q}_m_{m}.csv")
        if file_name in ["c_index_avg", "c_index_indv"]:
            if(os.path.exists(f"results/final/run_{run}_{data_name}_cox_{file_name}.csv") and os.path.isfile(f"results/final/run_{run}_{data_name}_cox_{file_name}.csv")):
                os.remove(f"results/final/run_{run}_{data_name}_cox_{file_name}.csv")
        
        if file_name == "auc":
            if(os.path.exists(f"results/final/run_{run}_{data_name}_cox_{file_name}.csv") and os.path.isfile(f"results/final/run_{run}_{data_name}_cox_{file_name}.csv")):
                os.remove(f"results/final/run_{run}_{data_name}_cox_{file_name}.csv")
    return df


def create_covars_df(data_name:str,num_runs_lower:int ,num_runs:int):
    true_covars = ["Con_True_1", "Con_True_2", "Con_True_3_corr", "Con_True_4_corr", "Cat_True"]
    false_covars = [f"X{i}" for i in range(3, 206)] + ["corr_with_Con_True_3_corr_1", "corr_with_Con_True_3_corr_2",  "corr_with_Con_True_4_corr_1",  "corr_with_Con_True_4_corr_2", "X213"]
    covars = true_covars + false_covars
    zeros = np.zeros(len(covars))

    df = pd.DataFrame()

    for run in range(num_runs_lower, num_runs+1):
        chosen_var_list = pd.read_csv(f"data_sets/covars/run_{run}_{data_name}_rsf_chosen_var_supervised.csv")["Unnamed: 0"].values.tolist()
        df_ = {f"{covars[i]}": [zeros[i]] for i in range(len(covars))}
        for cv in chosen_var_list:
            df_[cv][0] += 1

        df = pd.concat([df, pd.DataFrame.from_dict(df_)])

        if(os.path.exists(f"data_sets/covars/run_{run}_{data_name}_rsf_chosen_var_supervised.csv") and os.path.isfile(f"data_sets/covars/run_{run}_{data_name}_rsf_chosen_var_supervised.csv")):
            os.remove(f"data_sets/covars/run_{run}_{data_name}_rsf_chosen_var_supervised.csv")
    

    return df


def main(num_runs_lower:int, num_runs:int):
    num_runs_lower = int(num_runs_lower)
    num_runs = int(num_runs)

    filenames = ["c_index_avg", "c_index_indv", "auc_avg", "auc_indv", "coefs"]
    data_names = ["df_strong_coef_t0", "df_strong_coef_full_t0"]

    qm = [(1,0), (3,2)]
    
    for data_name in data_names:
        
        return_df = pd.DataFrame()
        
        try:
            df_temp = create_covars_df(data_name=data_name,num_runs_lower=num_runs_lower ,num_runs=num_runs)
            return_df = pd.concat([return_df, df_temp], axis = 1)
        
        except:
            pass

        
        for file_name in filenames:
            for q, m in qm:
                try:
                    df_temp = create_df(file_name=file_name, data_name=data_name, q=q, m=m,num_runs_lower=num_runs_lower ,num_runs=num_runs)
                    return_df = pd.concat([return_df, df_temp], axis = 1)
                
                except:
                    pass
                
                finally:
                    continue
        
        if return_df.empty:
            continue
        
        return_df.to_csv(f"results/summary/{data_name}_summary.csv", index=False)
    
   
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])