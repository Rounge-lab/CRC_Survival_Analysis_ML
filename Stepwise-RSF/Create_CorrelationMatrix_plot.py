import sys
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 
import seaborn as sns

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

def main(data_name:str):
  """
  Creates correlation matrix plot for the intersection of miRNAs
  chosen by RSF models using the stepwise procedure.

        Parameters
        ----------
        data_name : str
            File path of the data the models were trained on
        

        Returns
        -------
        None
  """
    
    # Get data
    df1 = pd.read_csv(f"path")
    df2 = pd.read_csv(f"path")
    df3 = pd.read_csv(f"path")
    
    # Get selected variable for each data group
    idx = df1["rsf.err"].idxmin()
    df1 = df1.iloc[idx:]["rm.var"].tolist()
    df1 = fix_names(df1)

    idx = df2["rsf.err"].idxmin()
    df2 = df2.iloc[idx:]["rm.var"].tolist()
    df2 = fix_names(df2)

    idx = df3["rsf.err"].idxmin()
    df3 = df3.iloc[idx:]["rm.var"].tolist()
    df3 = fix_names(df3)
    
    # Get the intersection of selected miRNAs
    def intersection(lst1, lst2):
        
        return [item for item in lst1 if item in lst2]

    cov12 = intersection(df1, df2)
    cov13 = intersection(df1, df3)
    cov23 = intersection(df2, df3)
    cov = intersection(cov12, intersection(cov13, cov23))
    
    # create a dataframe for seaborn heatmap annotation
    data = pd.read_csv(f"{data_name}.csv")
    data = data[cov]

    # Create correlation matrix plot of common miRNAs selected on all data groups
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8), dpi=300)
    fig.suptitle(f"Correlation matrix for the core set of\n miRNA's on the Full data group", size=24)
    ax = sns.heatmap(data.corr().round(2), cmap="Blues", annot=True)
    ax.set_xticklabels(ax.get_xticklabels() ,rotation=45, ha="right")
    ax.tick_params(axis='both', which='major', labelsize=14)
    #ax.tick_params(axis="x", which='major', rotation=45, )
    fig.tight_layout(rect=[0, 0.0, 1, 0.99])
    plt.savefig("name.jpeg", dpi=300)

if __name__ == "__main__":
    main(sys.argv[1])
