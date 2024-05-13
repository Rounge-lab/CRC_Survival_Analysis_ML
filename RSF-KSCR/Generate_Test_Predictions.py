import sys
import numpy as np
import pandas as pd
import pickle
from TVKCox_Regression import TVKCox_regression, Kernels
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def fix_names(name_list:list):
  for col in name_list:
    i = name_list.index(col)
    temp = col.replace(".", "-")
    name_list = name_list[:i] + [temp] + name_list[i+1:]
  
  return name_list

def generate_predictions(data_name:str, dg:str, get_vars:str):

    df = pd.read_csv(f"{data_name}_test.csv")
    df = df.drop("Unnamed: 0", axis=1)
    df = df.rename(columns={"tdato_diag_time_years": "T", "condition": "delta", "TDATO": "t", "JanusID": "id"})
    df["delta"].replace(["CRC", "C"], [1,0], inplace=True)
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

    # get chosen vars:
    if get_vars == "md_sel":
        df_cv = pd.read_csv(f"path")["Unnamed: 0"].tolist()
        df_cv = fix_names(df_cv)

    elif get_vars == "stepwise":

        if data_name + "_test" == "name1":
            df_cv = pd.read_csv(f"path1")
        elif data_name + "_test" == "name2":
            df_cv = pd.read_csv(f"path2")
        else:
            df_cv = pd.read_csv(f"path3")

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

    if data_name == "name3":
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

    df_preds = df[["id", "T", "t", "delta"]].copy(deep=True)
    delta = df.delta.copy(deep=True)
    df = df.drop("delta", axis=1)

    # Import best models:
    best_models = [f"{dg}_q1m0", f"{dg}_q4m2"]
    mods = ["q1m0", "q4m2"]
    for i in range(len(best_models)):
        model = pickle.load(open(f"{best_models[i]}.p", "rb"))
        df_preds["risk_" + mods[i]] = model.predict(df)
    
    df_preds.to_csv(f"{dg}_risk_scores.csv")

    return

if __name__ == "__main__":
    generate_predictions(sys.argv[1], sys.argv[2], sys.argv[3])
