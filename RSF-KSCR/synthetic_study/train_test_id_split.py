import numpy as np
import pandas as pd
import sys
from pathlib import Path

def main(filename:str, run:int):
    run = int(run)
    
    np.random.seed(run)
    
    root = Path('data_sets', 'data_files')
    df = pd.read_csv(root / f"{filename}")
    df_id = df.groupby("id").first()["event"]
    sample = df_id.sample(frac = 0.2).reset_index()["id"]
    sample.to_csv(f"data_sets/data_files/{filename[:-4]}_test_ids.csv", index = False)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

