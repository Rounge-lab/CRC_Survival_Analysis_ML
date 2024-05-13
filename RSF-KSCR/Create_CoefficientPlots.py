import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import ast
import pickle
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

def plotting(df,x:int, y:int, idx:int, covariate_list:list, data_name:str, q:int, a:int, ci:str):

  if q > 1:
    fig, ax = plt.subplots(nrows=x, ncols=y, figsize = (12,16), dpi=300)
    
    fig.suptitle("Coefficient paths for time varying coefficients KSCR with\n bootstrapped {(1-a)*100}% confidence intervals on\n the Complete metadata data group",size=24)
    fig.supylabel('Coefficient size', size=20)
    fig.supxlabel('Time to diagnosis (years)', size=20)

    i = 0
    c = 0
    j = 0
    while c < x:
      if j == df.shape[0]:
        break
      
      if x == 1 or y == 1:
        ax[i].plot(df[j,:,3], np.exp(df[j,:,1]), label = covariate_list[j], color="blue")
        if ci == "True"
          ax[i].plot(df[j,:,3], np.exp(df[j,:,0]), color="blue", alpha=0)
          ax[i].plot(df[j,:,3], np.exp(df[j,:,2]), color="blue", alpha=0)
          lines = ax[i].get_lines()
          ax[i].fill_between(df[j,:,3], lines[1].get_ydata(), lines[2].get_ydata(), alpha=0.3, color="blue")
        ax[i].set_title(covariate_list[j], size=18)
        ax[i].axhline(y = 1, color="black", linestyle = "dashed", alpha=0.7)
        ax[i].set_ylim(0,2)
        ax[i].tick_params(axis='both', which='major', labelsize=16)

      else:
        ax[c, i].plot(df[j,:,3], np.exp(df[j,:,1]), label = covariate_list[j], color="blue")
        if ci == "True"
          ax[c, i].plot(df[j,:,3], np.exp(df[j,:,0]), color="blue", alpha=0)
          ax[c, i].plot(df[j,:,3], np.exp(df[j,:,2]), color="blue", alpha=0)
          lines = ax[c, i].get_lines()
          ax[c, i].fill_between(df[j,:,3], lines[1].get_ydata(), lines[2].get_ydata(), alpha=0.3, color="blue")
        ax[c, i].set_title(covariate_list[j], size=18)
        ax[c, i].axhline(y = 1, color="black", linestyle = "dashed", alpha=0.7)
        ax[c, i].set_ylim(0,2)
        ax[c, i].tick_params(axis='both', which='major', labelsize=16)

      i+=1
      j+=1
      if i == y:
        i = 0
        c += 1
      
    fig.tight_layout(rect=[0, 0.0, 1, 0.99])
    plt.savefig(f"{data_name}_cox_coef_q_{q}.jpeg", dpi=300)
  
  else:
    df = pd.DataFrame(data=df, columns=[f"Lower bound {(1-a)*100}% bootstrap CI","estimated coefficient",f"Upper bound {(1-a)*100}% bootstrap CI"])
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (12,12), dpi=200)
    
    fig.suptitle(f"Coefficient values with bootstrapped {(1-a)*100}% confidence intervals", size = 24)
    fig.supxlabel('Coefficient size', size=16)

    y_ticks = np.arange(1, len(covariate_list)+1)
    
    ax.errorbar(x = np.exp(df["estimated coefficient"].values), y = y_ticks, 
      xerr = np.exp(df[[f"Lower bound {(1-a)*100}% bootstrap CI", f"Upper bound {(1-a)*100}% bootstrap CI"]].values.T),
      fmt="o", capsize=5, color="blue")
    ax.set_yticks(y_ticks)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_yticklabels(covariate_list, size=14)
    ax.axvline(x = 1, color="black")
    
    plt.tight_layout()
    plt.savefig(f"{data_name}_cox_coef_q_{q}_bootstrap.jpeg")


def main(dg:str, q:int, alpha:float, mod_idx:int, ci:str):
  """
  Creates coefficient plots with or without confidence intervals
  from bootstrap results generated using Bootstrap_Cox_Model.py

        Parameters
        ----------
        dg : str
            The data group used to generate the data
        q : int
            The B-spline coefficent number (1-> constant)
        alpha : float
            The significance level to use (0.05 -> 95%)
        mod_idx : int
            The number assigned to the chosen model when using Evaluate_GridSearch_Results.py
        ci : str
            Indicator to display confidence intervals ("True" -> display CI)

        Returns
        -------
        None
  """

  q = int(q)
  alpha = float(alpha)
  mod_idx = int(mod_idx)
  ci = ci
  
  _origdim = np.loadtxt(f"cox_bs_{dg}_q{q}_origdim.csv", delimiter=",")
  ci = np.loadtxt(f"cox_bs_ci_{dg}_q{q}.csv", delimiter=",")
  
  # reshape to 3D for time varying coef paths
  if int(_origdim.shape[0]) > 2:
    ci = ci.reshape(ci.shape[0], ci.shape[1] // int(_origdim[2]), int(_origdim[2]))
  
  eval_times = eval_times = np.linspace(0, 10, 50)
  ci_ = np.zeros((ci.shape[0], 50, 4))
  ci_[:, :, 3] = eval_times
  ci_[:, :, 1] = ci[:,:,0]
  ci_[:, :, 0] = np.quantile(ci[:,:,0:], alpha/2, axis=2)
  ci_[:, :, 2] = np.quantile(ci[:,:,0:], 1-(alpha/2), axis=2)

  if q == 1:
    ci_ = ci_[:,0,0:3].reshape((ci_.shape[0], ci_.shape[2]-1))

  try:
    with open(f"{dg}_{mod_idx}_q_{q}_idx_list", "rb") as fp:
      idx_list = pickle.load(fp)
  except:
    idx_list = []

  if len(idx_list) != 0:
    ci_ = np.delete(ci_, idx_list, axis=0)


  with open(f"cox_bs_{dg}_q{q}_covlist", "rb") as fp:
    covariate_list = pickle.load(fp)

  covariate_list = [i for j, i in enumerate(covariate_list) if j not in idx_list]
  plotting(ci_, int(np.ceil(ci_.shape[0]/2)),2,0, covariate_list, dg, q, alpha, ci)
  
if __name__ == "__main__":
  main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
