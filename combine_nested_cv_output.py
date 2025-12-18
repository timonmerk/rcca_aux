import pandas as pd
import os
from tqdm import tqdm
from joblib import Parallel, delayed


def load_and_combine(folder_name, rs, region):
    folder_ = f"{folder_name}_{rs}_{region}"
    files = os.listdir(f"/scratch/tm162/rcca_run/{folder_}")
    l_ = []
    for f in tqdm(files):
        if f.endswith(".csv"):
            df_temp = pd.read_csv(f"/scratch/tm162/rcca_run/{folder_}/{f}")
            l_.append(df_temp)
    df = pd.concat(l_, ignore_index=True)
    df.to_csv(f"/scratch/tm162/rcca_run/{folder_}.csv", index=False)

combinations = []
folder_name = "outccn"
for rs in ["suds", "rs"]: # 
    for region in ["SC", "C", "all"]:
        #load_and_combine(folder_name, rs, region)
        combinations.append((folder_name, rs, region))
Parallel(n_jobs=6)(delayed(load_and_combine)(folder_name, rs, region) for folder_name, rs, region in combinations)

