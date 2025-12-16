import pandas as pd
import os
from tqdm import tqdm


folder_name = "out"
for rs in ["rs", "suds"]:
    for region in ["SC", "C", "all"]:
        folder_ = f"{folder_name}_{rs}_{region}"
        files = os.listdir(f"/scratch/tm162/rcca_run/{folder_}")
        l_ = []
        for f in tqdm(files):
            if f.endswith(".csv"):
                df_temp = pd.read_csv(f"/scratch/tm162/rcca_run/{folder_}/{f}")
                l_.append(df_temp)
        df = pd.concat(l_, ignore_index=True)
        df.to_csv(f"/scratch/tm162/rcca_run/{folder_}.csv", index=False)



