import pandas as pd
import os
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import numpy as np

folder_ = "out_rs_rcca_output_with_fau_au"
files_ = os.listdir(folder_)
READ = False
if READ:
    df_ = pd.concat([pd.read_csv(os.path.join(folder_, f)) for f in files_]).reset_index(drop=False)
    df_.to_csv("out_rs_rcca_output_with_fau_au_combined.csv", index=False)
else:
    df_ = pd.read_csv("out_rs_rcca_output_with_fau_au_combined.csv")
# rename AU column to label
df_ = df_.rename(columns={"AU": "label"})

AU_COLUMNS = [f for f in df_["label"] if f.startswith("AU_")]
audio_columns = ["Loudness_sma3","alphaRatio_sma3","hammarbergIndex_sma3","slope0-500_sma3","slope500-1500_sma3","spectralFlux_sma3","mfcc1_sma3","mfcc2_sma3","mfcc3_sma3","mfcc4_sma3","F0semitoneFrom27.5Hz_sma3nz","jitterLocal_sma3nz","shimmerLocaldB_sma3nz","HNRdBACF_sma3nz","logRelF0-H1-H2_sma3nz","logRelF0-H1-A3_sma3nz","F1frequency_sma3nz","F1bandwidth_sma3nz","F1amplitudeLogRelF0_sma3nz","F2frequency_sma3nz","F2bandwidth_sma3nz","F2amplitudeLogRelF0_sma3nz","F3frequency_sma3nz","F3bandwidth_sma3nz","F3amplitudeLogRelF0_sma3nz","F0semitoneFrom27.5Hz_sma3nz_amean","F0semitoneFrom27.5Hz_sma3nz_stddevNorm","F0semitoneFrom27.5Hz_sma3nz_percentile20.0","F0semitoneFrom27.5Hz_sma3nz_percentile50.0","F0semitoneFrom27.5Hz_sma3nz_percentile80.0","F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2","F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope","F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope","F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope","F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope","loudness_sma3_amean","loudness_sma3_stddevNorm","loudness_sma3_percentile20.0","loudness_sma3_percentile50.0","loudness_sma3_percentile80.0","loudness_sma3_pctlrange0-2","loudness_sma3_meanRisingSlope","loudness_sma3_stddevRisingSlope","loudness_sma3_meanFallingSlope","loudness_sma3_stddevFallingSlope","spectralFlux_sma3_amean","spectralFlux_sma3_stddevNorm","mfcc1_sma3_amean","mfcc1_sma3_stddevNorm","mfcc2_sma3_amean","mfcc2_sma3_stddevNorm","mfcc3_sma3_amean","mfcc3_sma3_stddevNorm","mfcc4_sma3_amean","mfcc4_sma3_stddevNorm","jitterLocal_sma3nz_amean","jitterLocal_sma3nz_stddevNorm","shimmerLocaldB_sma3nz_amean","shimmerLocaldB_sma3nz_stddevNorm","HNRdBACF_sma3nz_amean","HNRdBACF_sma3nz_stddevNorm","logRelF0-H1-H2_sma3nz_amean","logRelF0-H1-H2_sma3nz_stddevNorm","logRelF0-H1-A3_sma3nz_amean","logRelF0-H1-A3_sma3nz_stddevNorm","F1frequency_sma3nz_amean","F1frequency_sma3nz_stddevNorm","F1bandwidth_sma3nz_amean","F1bandwidth_sma3nz_stddevNorm","F1amplitudeLogRelF0_sma3nz_amean","F1amplitudeLogRelF0_sma3nz_stddevNorm","F2frequency_sma3nz_amean","F2frequency_sma3nz_stddevNorm","F2bandwidth_sma3nz_amean","F2bandwidth_sma3nz_stddevNorm","F2amplitudeLogRelF0_sma3nz_amean","F2amplitudeLogRelF0_sma3nz_stddevNorm","F3frequency_sma3nz_amean","F3frequency_sma3nz_stddevNorm","F3bandwidth_sma3nz_amean","F3bandwidth_sma3nz_stddevNorm","F3amplitudeLogRelF0_sma3nz_amean","F3amplitudeLogRelF0_sma3nz_stddevNorm","alphaRatioV_sma3nz_amean","alphaRatioV_sma3nz_stddevNorm","hammarbergIndexV_sma3nz_amean","hammarbergIndexV_sma3nz_stddevNorm","slopeV0-500_sma3nz_amean","slopeV0-500_sma3nz_stddevNorm","slopeV500-1500_sma3nz_amean","slopeV500-1500_sma3nz_stddevNorm","spectralFluxV_sma3nz_amean","spectralFluxV_sma3nz_stddevNorm","mfcc1V_sma3nz_amean","mfcc1V_sma3nz_stddevNorm","mfcc2V_sma3nz_amean","mfcc2V_sma3nz_stddevNorm","mfcc3V_sma3nz_amean","mfcc3V_sma3nz_stddevNorm","mfcc4V_sma3nz_amean","mfcc4V_sma3nz_stddevNorm","alphaRatioUV_sma3nz_amean","hammarbergIndexUV_sma3nz_amean","slopeUV0-500_sma3nz_amean","slopeUV500-1500_sma3nz_amean","spectralFluxUV_sma3nz_amean","loudnessPeaksPerSec","VoicedSegmentsPerSec","MeanVoicedSegmentLengthSec","StddevVoicedSegmentLengthSec","MeanUnvoicedSegmentLength","StddevUnvoicedSegmentLength","equivalentSoundLevel_dBp","arousal","dominance","valence"] + [f"Dim {i}" for i in range(1024)] + ["duration"]
behav_columns = np.unique(AU_COLUMNS + audio_columns)

# take for each subject the max r value across num_cc and reg for each behavior feature
max_r_values = df_.groupby(["label", "SHUFFLE", "subject"])["r"].max().reset_index()
# make a histogram of the mean r values for each behavior feature, separate for shuffled and non-shuffled
mean_r = max_r_values.groupby(["label", "SHUFFLE"])["r"].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.histplot(data=mean_r, x="r", hue="SHUFFLE", bins=30, kde=True, stat="density", common_norm=False)
plt.title("Histogram of Mean Max r Values per Behavior Feature")
plt.xlabel("Mean Max r Value")
plt.ylabel("Density")
plt.savefig("behav/mean_max_r_values_histogram.pdf")

# ok, for the max_r_values, make a heatmap, and count for each best subject and behavior feature, how often each reg and num_cc was selected
df_best = df_.loc[
    df_["r"].eq(
        df_.groupby(["label", "SHUFFLE", "subject"])["r"].transform("max")
    )
]
heatmap_data_nonshuffled = np.zeros((len(df_["num_cc"].unique()), len(df_["reg"].unique())))
heatmap_data_shuffled = np.zeros((len(df_["num_cc"].unique()), len(df_["reg"].unique())))
num_ccs = sorted(df_["num_cc"].unique())
regs = sorted(df_["reg"].unique())
for _, row in df_best.iterrows():
    num_cc_idx = num_ccs.index(row["num_cc"])
    reg_idx = regs.index(row["reg"])
    if row["SHUFFLE"] == False:
        heatmap_data_nonshuffled[num_cc_idx, reg_idx] += 1
    else:
        heatmap_data_shuffled[num_cc_idx, reg_idx] += 1

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(heatmap_data_nonshuffled, annot=True, fmt=".0f", cmap="Blues", xticklabels=regs, yticklabels=num_ccs)
plt.title("Count of Best reg and num_cc (Unshuffled)")
plt.xlabel("reg")
plt.ylabel("num_cc")
plt.subplot(1, 2, 2)
sns.heatmap(heatmap_data_shuffled, annot=True, fmt=".0f", cmap="Blues", xticklabels=regs, yticklabels=num_ccs)
plt.title("Count of Best reg and num_cc (Shuffled)")
plt.xlabel("reg")
plt.ylabel("num_cc")
plt.tight_layout()
plt.savefig("behav/count_best_reg_numcc_heatmaps.pdf") 

# repeat the above but with average r values
avg_r_values = df_.groupby(["label", "SHUFFLE", "subject"])["r"].mean().reset_index()
mean_r_avg = avg_r_values.groupby(["label", "SHUFFLE"])["r"].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.histplot(data=mean_r_avg, x="r", hue="SHUFFLE", bins=30, kde=True, stat="density", common_norm=False)
plt.title("Histogram of Mean Average r Values per Behavior Feature")
plt.xlabel("Mean Average r Value")
plt.ylabel("Density")
plt.savefig("behav/mean_average_r_values_histogram.pdf")

# now plot the average across all behavior features heatmaps for shuffled vs non-shuffled in two heatmaps
avg_map = df_.groupby(["SHUFFLE", "num_cc", "reg"])["r"].mean().reset_index()
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
df_pivot = pd.pivot_table(avg_map.query("SHUFFLE == False"), values="r", index="num_cc", columns="reg")
sns.heatmap(df_pivot, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Average r Values across all Behavior Features (Unshuffled)")
plt.subplot(1, 2, 2)
df_pivot = pd.pivot_table(avg_map.query("SHUFFLE == True"), values="r", index="num_cc", columns="reg")
sns.heatmap(df_pivot, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Average r Values across all Behavior Features (Shuffled)")
plt.tight_layout()
plt.savefig("behav/average_r_values_heatmaps_shuffled_vs_unshuffled.pdf")


# for reg = 10000 and num_cc = 25
# plot the histogram of r values for all behavior features for shuffled vs non-shuffled
reg_val = 10000
num_cc_val = 25
df_filtered = df_.query("reg == @reg_val and num_cc == @num_cc_val")
plt.figure(figsize=(10, 6))
sns.histplot(data=df_filtered, x="r", hue="SHUFFLE", bins=30, kde=True, stat="density", common_norm=False)
plt.title(f"Histogram of r Values for reg={reg_val} and num_cc={num_cc_val}")
plt.xlabel("r Value")
plt.ylabel("Density")
plt.savefig(f"behav/r_values_histogram_reg_{reg_val}_numcc_{num_cc_val}.pdf")


mean_shuffled = []
mean_nonshuffled = []
i = 0
pdf_p = PdfPages(f"behav/heatmaps_behav_features_shuffled_vs_unshuffled_{i}.pdf")

for i, col in tqdm(enumerate(behav_columns)):
    if i % 50 == 0 and i > 0:
        pdf_p.close()
        pdf_p = PdfPages(f"behav/heatmaps_behav_features_shuffled_vs_unshuffled_{i}.pdf")
    df_col = df_.query("label == @col")
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    df_pivot = pd.pivot_table(df_col.query("SHUFFLE == False"), values="r", index="num_cc", columns="reg")
    sns.heatmap(df_pivot, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title(f"Heatmap for {col} (Unshuffled)")
    plt.subplot(1, 2, 2)
    df_pivot = pd.pivot_table(df_col.query("SHUFFLE == True"), values="r", index="num_cc", columns="reg")
    sns.heatmap(df_pivot, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title(f"{col} (Shuffled)")
    plt.tight_layout()
    pdf_p.savefig()
    mean_shuffled.append(df_col.query("SHUFFLE == True")["r"].mean())
    mean_nonshuffled.append(df_col.query("SHUFFLE == False")["r"].mean())
pdf_p.close()