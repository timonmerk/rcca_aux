import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os
from scipy import stats
import numpy as np
# import pdf
from matplotlib.backends.backend_pdf import PdfPages

folder_ = "out2"  # out_shuffle

files_ = os.listdir("out2")
kcca_ = [bool(f.split("_")[-1][:-4] == "True") for f in files_]
norm_ = [bool(f.split("_")[14] == "True") for f in files_]

df_ = pd.concat([pd.read_csv("out2/" + f) for f in os.listdir("out2")]).reset_index(drop=False)
idx_best_suds = df_.query("INCLUDE_AU == False and INCLUDE_AUDIO == False and INCLUDE_AU == False").groupby("subject")["r"].idxmax()
idx_best_all = df_.query("INCLUDE_AU == True or INCLUDE_AUDIO == True").groupby("subject")["r"].idxmax()

df_best_suds = df_.loc[idx_best_suds].reset_index(drop=True)
df_best_all = df_.loc[idx_best_all].reset_index(drop=True)

cca_dims = df_["num_cc"].unique()
regs = df_["reg"].unique()

df_["norm"] = norm_
df_["kcca"] = kcca_

df_suds_c = df_.query("SUDS == True and INCLUDE_AU == False and INCLUDE_AUDIO == False and norm == True and kcca == False and num_cc == 25 and reg == 1000")
df_all_c = df_.query("SUDS == True and INCLUDE_AU == True and INCLUDE_AUDIO == False and norm == True and kcca == False and num_cc == 25 and reg == 1000")


#df_suds_c = df_best_suds.copy()
#df_all_c = df_best_all.copy()

df_suds_c["condition"] = "SUDS Only"
df_all_c["condition"] = "SUDS + AU"
# sort both by subject
df_suds_c = df_suds_c.sort_values("subject")
df_all_c = df_all_c.sort_values("subject")
df_concat_c = pd.concat([df_suds_c, df_all_c])


# plot the r values for both with a boxplot, plot individual dots and connect by lines, but a permutation test to compare r 
stats_comp = stats.permutation_test((df_all_c["r"]- df_suds_c["r"], np.zeros(len(df_suds_c))), statistic=lambda x, y: np.mean(y) - np.mean(x), n_resamples=10000)
plt.figure(figsize=(8, 6))
sns.boxplot(x="condition", y="r", data=df_concat_c, palette="Set2", showmeans=True)
sns.stripplot(x="condition", y="r", data=df_concat_c, color="black", jitter=False, dodge=True)
for i in range(len(df_suds_c)):
    plt.plot(["SUDS Only", "SUDS + AU"], [df_suds_c["r"].values[i], df_all_c["r"].values[i]], color="gray", alpha=0.5)
plt.title(f"RCCA Results Comparison\nPermutation p-value: {stats_comp.pvalue:.4f}")
plt.ylabel("Correlation Coefficient (r)")
plt.savefig("rcca_suds_au_comparison_reg_1000_cc_25.pdf")
plt.close()

# kenel_ = False
# norm_ = False

# pdf_ = PdfPages("rcca_results_summary_shuffle.pdf")

# for norm_ in [False, True]:
#     for kenel_ in [False, True]:
#         df_suds_only = df_.query("SUDS == True and INCLUDE_AU == False and INCLUDE_AUDIO == False and NORMALIZE == @norm_ and KERNEL_CCA == @kenel_")
#         df_suds_only_mean = df_suds_only.groupby(["num_cc", "reg"])["r"].mean().reset_index()

#         df_suds_au = df_.query("SUDS == True and INCLUDE_AU == True and INCLUDE_AUDIO == False and NORMALIZE == @norm_ and KERNEL_CCA == @kenel_")
#         df_suds_au_mean = df_suds_au.groupby(["num_cc", "reg"])["r"].mean().reset_index()

#         df_suds_audio = df_.query("SUDS == True and INCLUDE_AU == False and INCLUDE_AUDIO == True and NORMALIZE == @norm_ and KERNEL_CCA == @kenel_")
#         df_suds_audio_mean = df_suds_audio.groupby(["num_cc", "reg"])["r"].mean().reset_index()

#         df_suds_video_audio = df_.query("SUDS == True and INCLUDE_AU == True and INCLUDE_AUDIO == True and NORMALIZE == @norm_ and KERNEL_CCA == @kenel_")
#         df_suds_video_audio_mean = df_suds_video_audio.groupby(["num_cc", "reg"])["r"].mean().reset_index()

#         df_suds_only_pivot = pd.pivot_table(df_suds_only_mean, values="r", index="num_cc", columns="reg")
#         df_suds_au_pivot = pd.pivot_table(df_suds_au_mean, values="r", index="num_cc", columns="reg")
#         df_suds_audio_pivot = pd.pivot_table(df_suds_audio_mean, values="r", index="num_cc", columns="reg")
#         df_suds_video_audio_pivot = pd.pivot_table(df_suds_video_audio_mean, values="r", index="num_cc", columns="reg")
#         # show 4 images in a 2x2 grid
#         fig, axs = plt.subplots(2, 2, figsize=(12, 10))
#         sns.heatmap(df_suds_only_pivot, annot=True, fmt=".2f", ax=axs[0, 0], cmap="coolwarm", vmin=-0.3, vmax=0.3)
#         axs[0, 0].set_title("SUDS Only")
#         sns.heatmap(df_suds_au_pivot, annot=True, fmt=".2f", ax=axs[0, 1], cmap="coolwarm", vmin=-0.3, vmax=0.3)
#         axs[0, 1].set_title("SUDS + AU")
#         sns.heatmap(df_suds_audio_pivot, annot=True, fmt=".2f", ax=axs[1, 0], cmap="coolwarm", vmin=-0.3, vmax=0.3)
#         axs[1, 0].set_title("SUDS + Audio")
#         sns.heatmap(df_suds_video_audio_pivot, annot=True, fmt=".2f", ax=axs[1, 1], cmap="coolwarm", vmin=-0.3, vmax=0.3)
#         axs[1, 1].set_title("SUDS + AU + Audio")
#         fig.suptitle(f"RCCA Results Kernel norm={norm_} kernel = {kenel_})", fontsize=16)
#         plt.tight_layout()
#         pdf_.savefig(fig)
#         plt.close(fig)
# pdf_.close()