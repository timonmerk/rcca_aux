import pandas as pd
import os
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

folder_read = "outercv"
folder_read = "outercv_lim"
folder_read = "outercv_limmm"
folder_read = "outercvs"
folder_read = "outerccn"
folder_read = "outerccn_without_ccdim_decoding"
#folder_read = "outerccn_without_ccdim_decoding_mse"
folder_read = "outerccn_only_ccdim"
folder_read = "outerccn_wo_reg01"

for DECODE_SUDS in [True, False]:
    df_comb = []
    for loc in ["all", "SC", "C"]:
        if DECODE_SUDS:
            folder_read_ = folder_read + "_suds"
            label_ = "score_feat"
        else:
            folder_read_ = folder_read + "_rs"
            label_ = "YBOCS II Total Score"
        folder_read_ += f"_{loc}"

        files_ = os.listdir(f"/scratch/tm162/rcca_run/{folder_read_}")
        df_ = pd.concat([pd.read_csv(f"/scratch/tm162/rcca_run/{folder_read_}/{f}") for f in files_ if f.endswith(".csv")], ignore_index=True)

        df_plt = df_.query("AU == @label_")
        df_plt["cond"] = "None"
        df_plt.loc[(df_plt["SUDS"]==True) & (df_plt["INCLUDE_AUDIO"]==False) & (df_plt["INCLUDE_AU"]==False), "cond"] = "SUDS"
        df_plt.loc[(df_plt["SUDS"]==True) & (df_plt["INCLUDE_AUDIO"]==True) & (df_plt["INCLUDE_AU"]==False), "cond"] = "SUDS + Audio"
        df_plt.loc[(df_plt["SUDS"]==True) & (df_plt["INCLUDE_AUDIO"]==False) & (df_plt["INCLUDE_AU"]==True), "cond"] = "SUDS + AU"
        df_plt.loc[(df_plt["SUDS"]==True) & (df_plt["INCLUDE_AUDIO"]==True) & (df_plt["INCLUDE_AU"]==True), "cond"] = "SUDS + AU + Audio"
        df_plt["region"] = loc
        df_comb.append(df_plt)

    df_plt = pd.concat(df_comb, ignore_index=True)
    region = "all"
    r_ = df_plt.query("cond == 'SUDS' and region == @region").sort_values("subject")["r"].values
    r_au_audio = df_plt.query("cond == 'SUDS + AU + Audio' and region == @region").sort_values("subject")["r"].values

    #r_test = (r_au_audio) - (r_)
    #r_zero = np.zeros_like(r_test)
    
    #stats.permutation_test((r_au_audio, r_), statistic=lambda x, y: np.mean(x) - np.mean(y), vectorized=False, n_resamples=5000, alternative="greater")

    plt.figure(figsize=(8,6))
    sns.boxplot(data=df_plt, x="region", y="r", hue="cond",
                dodge=True, showmeans=True, hue_order=["SUDS", "SUDS + Audio", "SUDS + AU", "SUDS + AU + Audio"],)
    sns.swarmplot(data=df_plt, x="region", y="r", hue="cond",
                  dodge=True, color=".25", hue_order=["SUDS", "SUDS + Audio", "SUDS + AU", "SUDS + AU + Audio"],)
    plt.title(f"Decode SUDS: {DECODE_SUDS}")
    plt.tight_layout()
    plt.savefig(f"/scratch/tm162/rcca_run/figures/{folder_read}_{DECODE_SUDS}.pdf")
