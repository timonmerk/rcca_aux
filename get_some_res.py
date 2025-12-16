import pandas as pd
import os
from scipy import stats
import numpy as np


folder_read = "16_12_1/out_rs_rcca_output_with_fau_au"
#folder_read = "16_12_1/out2_output_with_fau_au"
files_ = os.listdir(f"/scratch/tm162/rcca_run/{folder_read}")
df_ = pd.concat([pd.read_csv(f"/scratch/tm162/rcca_run/{folder_read}/{f}") for f in files_ if f.endswith(".csv")], ignore_index=True)
label_ = "YBOCS II Total Score"
#label_ = "score_feat"

df_plt = df_.query("AU == @label_")
df_plt["cond"] = "None"
# set to 'SUDS' where SUDS is True and AUDIO is False and INCLUDE_AU is False
# set to 'SUDS + Audio' where SUDS is True and AUDIO is True and INCLUDE_AU is False
# set to 'SUDS + AU + Audio' where SUDS is True and AUDIO is True and INCLUDE_AU is True
# set to 'SUDS + AU' where SUDS is True and AUDIO is False and INCLUDE_AU is True
df_plt.loc[(df_plt["SUDS"]==True) & (df_plt["INCLUDE_AUDIO"]==False) & (df_plt["INCLUDE_AU"]==False), "cond"] = "SUDS"
df_plt.loc[(df_plt["SUDS"]==True) & (df_plt["INCLUDE_AUDIO"]==True) & (df_plt["INCLUDE_AU"]==False), "cond"] = "SUDS + Audio"
df_plt.loc[(df_plt["SUDS"]==True) & (df_plt["INCLUDE_AUDIO"]==False) & (df_plt["INCLUDE_AU"]==True), "cond"] = "SUDS + AU"
df_plt.loc[(df_plt["SUDS"]==True) & (df_plt["INCLUDE_AUDIO"]==True) & (df_plt["INCLUDE_AU"]==True), "cond"] = "SUDS + AU + Audio"

r_ = df_plt.query("cond == 'SUDS'").sort_values("subject")["r"].values
r_au_audio = df_plt.query("cond == 'SUDS + AU + Audio'").sort_values("subject")["r"].values

r_test = r_au_audio - r_
r_zero = np.zeros_like(r_test)

stats.permutation_test((r_test, r_zero), statistic=lambda x, y: np.mean(x) - np.mean(y), vectorized=False, n_resamples=10000, alternative='greater')

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6,6))
sns.boxplot(data=df_plt, x="cond", y="r", showmeans=True)
sns.swarmplot(data=df_plt, x="cond", y="r", color=".25")
plt.tight_layout()
plt.savefig(f"/scratch/tm162/rcca_run/16_12_1/boxplot_r_suds_RS.png", dpi=300)
