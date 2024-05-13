import os
import sys

def main(run):      
    
    data_names = ["df_strong_coef_t0", "df_strong_coef_full_t0"]
    file_names = ["_cox_pred"]
    qms = [(1,0), (3,2)]
    for name in data_names:
        if(os.path.exists(f"data_sets/data_files/run_{run}_" + name + ".csv") and os.path.isfile(f"data_sets/data_files/run_{run}_" + name + ".csv")):
            os.remove(f"data_sets/data_files/run_{run}_" + name + ".csv")
        if(os.path.exists(f"data_sets/data_files/run_{run}_" + name + "_test_ids.csv") and os.path.isfile(f"data_sets/data_files/run_{run}_" + name + "_test_ids.csv")):
            os.remove(f"data_sets/data_files/run_{run}_" + name + "_test_ids.csv")
        for file in file_names:
            if file == "_rsf_pred":
                if(os.path.exists(f"results/perliminary/run_{run}_" + name + file + ".csv") and os.path.isfile(f"results/perliminary/run_{run}_" + name + file + ".csv")):
                    os.remove(f"results/perliminary/run_{run}_" + name + file + ".csv")
            else:
                for q,m in qms:
                    if(os.path.exists(f"results/perliminary/run_{run}_" + name + f"_q_{q}_m_{m}" + file + ".csv") and os.path.isfile(f"results/perliminary/run_{run}_" + name + f"_q_{q}_m_{m}" + file + ".csv")):
                        os.remove(f"results/perliminary/run_{run}_" + name + f"_q_{q}_m_{m}" + file + ".csv")
                
    return

if __name__ == "__main__":
    main(sys.argv[1])