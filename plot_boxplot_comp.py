import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os
from scipy import stats
import numpy as np
# import pdf
from matplotlib.backends.backend_pdf import PdfPages

folder_ = "out_mlp_aux_2"  # out_shuffle
folder_ = "out2"
folder_ = "out_mlp_2"
folder_ = "out_rs_rcca"

if folder_ != "out2":
    files_ = os.listdir(folder_)
    df_ = pd.concat([pd.read_csv(os.path.join(folder_, f)) for f in files_]).reset_index(drop=False)
else:
    df_ = pd.read_csv("out2_filtered.csv")

cca_dims = df_["num_cc"].unique()
regs = df_["reg"].unique()

# params with out_mlp_aux_2
#num_cc = 1
#reg = 100
GET_BEST = False

if GET_BEST is False:
    reg = 10000
    num_cc = 25
    str_save = f"{folder_}_boxplot_comparison_reg_{reg}_cc_{num_cc}.pdf"
else:
    str_save = f"{folder_}_boxplot_comparison_best.pdf"
INCLUDE_AU = True
INCLUDE_AUDIO = True
if INCLUDE_AU and INCLUDE_AUDIO:
    cond_name = "SUDS + AU + Audio"
else:
    cond_name = "SUDS + AU"


df_concat = []
for shuffle_ in [False, True]:  # need to be that order
    if GET_BEST is False:
        df_suds_c = df_.query("SUDS == True and INCLUDE_AU == False and INCLUDE_AUDIO == False and num_cc == @num_cc and reg == @reg and SHUFFLE == @shuffle_")
        df_all_c = df_.query("SUDS == True and INCLUDE_AU == @INCLUDE_AU and INCLUDE_AUDIO == @INCLUDE_AUDIO and num_cc == @num_cc and reg == @reg and SHUFFLE == @shuffle_")
    else:
        df_suds_c = df_.query("SUDS == True and INCLUDE_AU == False and INCLUDE_AUDIO == False and SHUFFLE == @shuffle_")
        idx_best_suds = df_suds_c.groupby("subject")["r"].idxmax()
        df_suds_c = df_suds_c.loc[idx_best_suds]
        df_all_c = df_.query("SUDS == True and INCLUDE_AU == @INCLUDE_AU and INCLUDE_AUDIO == @INCLUDE_AUDIO and SHUFFLE == @shuffle_")
        idx_best_all = df_all_c.groupby("subject")["r"].idxmax()
        df_all_c = df_all_c.loc[idx_best_all]
    df_suds_c["condition"] = "SUDS Only"
    df_all_c["condition"] = cond_name
    df_suds_c = df_suds_c.sort_values("subject")
    df_all_c = df_all_c.sort_values("subject")
    df_concat_c = pd.concat([df_suds_c, df_all_c])
    df_concat_c["SHUFFLE"] = shuffle_
    df_concat.append(df_concat_c)
df_concat_c = pd.concat(df_concat)

plt.figure()
for idx, shuffle_ in enumerate([False, True]):
    plt.subplot(1, 4, idx+1)
    df_check = df_concat_c.query("SHUFFLE == @shuffle_")
    df_suds_c = df_check.query("condition == 'SUDS Only'")
    df_all_c = df_check.query("condition == @cond_name")
    stats_comp = stats.permutation_test((df_all_c["r"].values- df_suds_c["r"].values, np.zeros(len(df_suds_c["r"].values))),
                                        statistic=lambda x, y: np.mean(y) - np.mean(x), n_resamples=5000)
    sns.boxplot(x="condition", y="r", data=df_check, palette="Set2", showmeans=True)
    sns.stripplot(x="condition", y="r", data=df_check, color="black", jitter=False, dodge=True)
    for i in range(len(df_suds_c)):
        plt.plot(["SUDS Only", cond_name], [df_suds_c["r"].values[i], df_all_c["r"].values[i]], color="gray", alpha=0.5)
    plt.title(f"SHUFFLE: {shuffle_}\np-value: {stats_comp.pvalue:.4f}")
    plt.ylabel("Correlation Coefficient (r)")
    plt.xticks(rotation=90)

# compare now for SUDS + AU + Audio shuffle vs no shuffle
plt.subplot(1, 4, 3)
df_suds_all_shuffle = df_concat_c.query("condition == @cond_name and SHUFFLE == True")
df_suds_all_no_shuffle = df_concat_c.query("condition == @cond_name and SHUFFLE == False")
stats_comp_all = stats.permutation_test((df_suds_all_no_shuffle["r"].values - df_suds_all_shuffle["r"].values, np.zeros(len(df_suds_all_no_shuffle["r"].values))),
                                        statistic=lambda x, y: np.mean(x) - np.mean(y), n_resamples=5000)

sns.boxplot(x="SHUFFLE", y="r", data=df_concat_c.query("condition == @cond_name"), palette="Set2", showmeans=True)
sns.stripplot(x="SHUFFLE", y="r", data=df_concat_c.query("condition == @cond_name"), color="black", jitter=False, dodge=True)
for i in range(len(df_suds_all_no_shuffle)):
    plt.plot([0, 1], [df_suds_all_no_shuffle["r"].values[i], df_suds_all_shuffle["r"].values[i]], color="gray", alpha=0.5)
plt.title(f"{cond_name}\np-value: {stats_comp_all.pvalue:.4f}")
plt.ylabel("Correlation Coefficient (r)")
plt.xticks(rotation=90)

plt.subplot(1, 4, 4)
df_suds_shuffle = df_concat_c.query("condition == 'SUDS Only' and SHUFFLE == True")
df_suds_no_shuffle = df_concat_c.query("condition == 'SUDS Only' and SHUFFLE == False")
stats_comp_all = stats.permutation_test((df_suds_no_shuffle["r"].values - df_suds_shuffle["r"].values, np.zeros(len(df_suds_no_shuffle["r"].values))),
                                        statistic=lambda x, y: np.mean(x) - np.mean(y), n_resamples=5000)

sns.boxplot(x="SHUFFLE", y="r", data=df_concat_c.query("condition == 'SUDS Only'"), palette="Set2", showmeans=True)
sns.stripplot(x="SHUFFLE", y="r", data=df_concat_c.query("condition == 'SUDS Only'"), color="black", jitter=False, dodge=True)
for i in range(len(df_suds_all_no_shuffle)):
    plt.plot([0, 1], [df_suds_no_shuffle["r"].values[i], df_suds_shuffle["r"].values[i]], color="gray", alpha=0.5)
plt.title(f"SUDS Only\np-value: {stats_comp_all.pvalue:.4f}")
plt.ylabel("Correlation Coefficient (r)")
plt.xticks(rotation=90)

plt.tight_layout()
plt.savefig(str_save)
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