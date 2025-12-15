import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os
from scipy import stats
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

folder_name = "out_mlp_aux_3"
folder_name = "out2"
folder_name = "out_mlp_2"
folder_name = "out_rs_rcca"
vmin_ = -1
vmax_ = 1

df_orig = pd.concat([pd.read_csv(os.path.join(folder_name, f))
                 for f in os.listdir(folder_name)]).reset_index(drop=False)

if folder_name == "out2":
    #df_orig = df_.copy()
    files_ = os.listdir(folder_name)
    shuffle_l = [True if "shuffle" in f else False for f in files_]
    norm_l = [True if "norm_True" in f else False for f in files_]
    kcca_l = [True if "kcca_True" in f else False for f in files_]
    df_orig["NORMALIZE"] = norm_l
    df_orig["KERNEL_CCA"] = kcca_l
    df_orig["SHUFFLE"] = shuffle_l
    df_ = df_orig.query("NORMALIZE == True and KERNEL_CCA == False").copy()
    df_.to_csv("out2_filtered.csv", index=False)
else:
    df_ = df_orig.copy()

pdf_ = PdfPages(f"{folder_name}_heatmaps.pdf")

for SHUFFLE_ in [True, False]:
    df__ = df_.query("SHUFFLE == @SHUFFLE_")
    df_suds_only = df__.query("SUDS == True and INCLUDE_AU == False and INCLUDE_AUDIO == False")
    df_suds_only_mean = df_suds_only.groupby(["num_cc", "reg"])["r"].mean().reset_index()

    df_suds_au = df__.query("SUDS == True and INCLUDE_AU == True and INCLUDE_AUDIO == False ")
    df_suds_au_mean = df_suds_au.groupby(["num_cc", "reg"])["r"].mean().reset_index()

    df_suds_audio = df__.query("SUDS == True and INCLUDE_AU == False and INCLUDE_AUDIO == True")
    df_suds_audio_mean = df_suds_audio.groupby(["num_cc", "reg"])["r"].mean().reset_index()

    df_suds_video_audio = df__.query("SUDS == True and INCLUDE_AU == True and INCLUDE_AUDIO == True")
    df_suds_video_audio_mean = df_suds_video_audio.groupby(["num_cc", "reg"])["r"].mean().reset_index()

    df_suds_only_pivot = pd.pivot_table(df_suds_only_mean, values="r", index="num_cc", columns="reg")
    df_suds_au_pivot = pd.pivot_table(df_suds_au_mean, values="r", index="num_cc", columns="reg")
    df_suds_audio_pivot = pd.pivot_table(df_suds_audio_mean, values="r", index="num_cc", columns="reg")
    df_suds_video_audio_pivot = pd.pivot_table(df_suds_video_audio_mean, values="r", index="num_cc", columns="reg")
    # show 4 images in a 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    sns.heatmap(df_suds_only_pivot, annot=True, fmt=".2f", ax=axs[0, 0], cmap="coolwarm", vmin=vmin_, vmax=vmax_)
    axs[0, 0].set_title("SUDS Only")
    sns.heatmap(df_suds_au_pivot, annot=True, fmt=".2f", ax=axs[0, 1], cmap="coolwarm", vmin=vmin_, vmax=vmax_)
    axs[0, 1].set_title("SUDS + AU")
    sns.heatmap(df_suds_audio_pivot, annot=True, fmt=".2f", ax=axs[1, 0], cmap="coolwarm", vmin=vmin_, vmax=vmax_)
    axs[1, 0].set_title("SUDS + Audio")
    sns.heatmap(df_suds_video_audio_pivot, annot=True, fmt=".2f", ax=axs[1, 1], cmap="coolwarm", vmin=vmin_, vmax=vmax_)
    axs[1, 1].set_title("SUDS + AU + Audio")
    fig.suptitle(f"{folder_name} SHUFFLE = {SHUFFLE_}", fontsize=16)
    pdf_.savefig(fig)
pdf_.close()

