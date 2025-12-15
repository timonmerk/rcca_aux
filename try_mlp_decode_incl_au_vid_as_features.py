import utils
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import rcca
from scipy import stats
import sys
import os

l_audio_features = ["Loudness_sma3","alphaRatio_sma3","hammarbergIndex_sma3","slope0-500_sma3","slope500-1500_sma3","spectralFlux_sma3","mfcc1_sma3","mfcc2_sma3","mfcc3_sma3","mfcc4_sma3","F0semitoneFrom27.5Hz_sma3nz","jitterLocal_sma3nz","shimmerLocaldB_sma3nz","HNRdBACF_sma3nz","logRelF0-H1-H2_sma3nz","logRelF0-H1-A3_sma3nz","F1frequency_sma3nz","F1bandwidth_sma3nz","F1amplitudeLogRelF0_sma3nz","F2frequency_sma3nz","F2bandwidth_sma3nz","F2amplitudeLogRelF0_sma3nz","F3frequency_sma3nz","F3bandwidth_sma3nz","F3amplitudeLogRelF0_sma3nz","F0semitoneFrom27.5Hz_sma3nz_amean","F0semitoneFrom27.5Hz_sma3nz_stddevNorm","F0semitoneFrom27.5Hz_sma3nz_percentile20.0","F0semitoneFrom27.5Hz_sma3nz_percentile50.0","F0semitoneFrom27.5Hz_sma3nz_percentile80.0","F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2","F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope","F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope","F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope","F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope","loudness_sma3_amean","loudness_sma3_stddevNorm","loudness_sma3_percentile20.0","loudness_sma3_percentile50.0","loudness_sma3_percentile80.0","loudness_sma3_pctlrange0-2","loudness_sma3_meanRisingSlope","loudness_sma3_stddevRisingSlope","loudness_sma3_meanFallingSlope","loudness_sma3_stddevFallingSlope","spectralFlux_sma3_amean","spectralFlux_sma3_stddevNorm","mfcc1_sma3_amean","mfcc1_sma3_stddevNorm","mfcc2_sma3_amean","mfcc2_sma3_stddevNorm","mfcc3_sma3_amean","mfcc3_sma3_stddevNorm","mfcc4_sma3_amean","mfcc4_sma3_stddevNorm","jitterLocal_sma3nz_amean","jitterLocal_sma3nz_stddevNorm","shimmerLocaldB_sma3nz_amean","shimmerLocaldB_sma3nz_stddevNorm","HNRdBACF_sma3nz_amean","HNRdBACF_sma3nz_stddevNorm","logRelF0-H1-H2_sma3nz_amean","logRelF0-H1-H2_sma3nz_stddevNorm","logRelF0-H1-A3_sma3nz_amean","logRelF0-H1-A3_sma3nz_stddevNorm","F1frequency_sma3nz_amean","F1frequency_sma3nz_stddevNorm","F1bandwidth_sma3nz_amean","F1bandwidth_sma3nz_stddevNorm","F1amplitudeLogRelF0_sma3nz_amean","F1amplitudeLogRelF0_sma3nz_stddevNorm","F2frequency_sma3nz_amean","F2frequency_sma3nz_stddevNorm","F2bandwidth_sma3nz_amean","F2bandwidth_sma3nz_stddevNorm","F2amplitudeLogRelF0_sma3nz_amean","F2amplitudeLogRelF0_sma3nz_stddevNorm","F3frequency_sma3nz_amean","F3frequency_sma3nz_stddevNorm","F3bandwidth_sma3nz_amean","F3bandwidth_sma3nz_stddevNorm","F3amplitudeLogRelF0_sma3nz_amean","F3amplitudeLogRelF0_sma3nz_stddevNorm","alphaRatioV_sma3nz_amean","alphaRatioV_sma3nz_stddevNorm","hammarbergIndexV_sma3nz_amean","hammarbergIndexV_sma3nz_stddevNorm","slopeV0-500_sma3nz_amean","slopeV0-500_sma3nz_stddevNorm","slopeV500-1500_sma3nz_amean","slopeV500-1500_sma3nz_stddevNorm","spectralFluxV_sma3nz_amean","spectralFluxV_sma3nz_stddevNorm","mfcc1V_sma3nz_amean","mfcc1V_sma3nz_stddevNorm","mfcc2V_sma3nz_amean","mfcc2V_sma3nz_stddevNorm","mfcc3V_sma3nz_amean","mfcc3V_sma3nz_stddevNorm","mfcc4V_sma3nz_amean","mfcc4V_sma3nz_stddevNorm","alphaRatioUV_sma3nz_amean","hammarbergIndexUV_sma3nz_amean","slopeUV0-500_sma3nz_amean","slopeUV500-1500_sma3nz_amean","spectralFluxUV_sma3nz_amean","loudnessPeaksPerSec","VoicedSegmentsPerSec","MeanVoicedSegmentLengthSec","StddevVoicedSegmentLengthSec","MeanUnvoicedSegmentLength","StddevUnvoicedSegmentLength","equivalentSoundLevel_dBp","arousal","dominance","valence"] + [f"Dim {i}" for i in range(1024)] + ["duration"]

df_merged, subs = utils.get_df_features("all", "all")
out_folder_name = "out_mlp_aux_3"
if not os.path.exists(f"/scratch/tm162/rcca_run/{out_folder_name}"):
    os.makedirs(f"/scratch/tm162/rcca_run/{out_folder_name}")


num_ccs = [1, 2, 3, 4, 5]
regs = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 100, 1000, 5000, 10000]
subs = [4, 5, 7, 9, 10, 11, 12]
INCLUDE_AU_l = [False, True] # True
INCLUDE_AUDIO_l = [False, True] #  True
SUDS = True
SHUFFLE_l = [True, False]
KERNEL_CCA_l = [False,]  # False
NORMALIZE_l = [True,]


def run_sub(df_merged, sub, num_cc, reg, SUDS=True, INCLUDE_AU=True, INCLUDE_AUDIO=True, NORMALIZE=True, KERNEL_CCA=False, SHUFFLE=True, ktype="gaussian", GET_RES=False):
    #INCLUDE_AUDIO = True; INCLUDE_AU = True
    save_name = f"/scratch/tm162/rcca_run/{out_folder_name}/rcca_sub_{sub}_numcc_{num_cc}_reg_{reg}_suds_{SUDS}_au_{INCLUDE_AU}_audio_{INCLUDE_AUDIO}_norm_{NORMALIZE}_kcca_{KERNEL_CCA}_shuffle_{SHUFFLE}.csv"

    if os.path.exists(save_name) and GET_RES is False:
        print(f"File {save_name} already exists. Skipping.")
        sys.exit(0)


    X_sub = df_merged.query("subject == @sub")
    X_sub["date"] = X_sub["time"].dt.date
    X_sub["session_id"] = X_sub["date"].astype("category").cat.codes
    X_neural = X_sub[[c for c in X_sub.columns if c.startswith("SC_") or c.startswith("C_")]]
    X_neural["session_id"] = X_sub["session_id"]

    cols_include = []
    if INCLUDE_AU:
        cols_include += [c for c in X_sub.columns if c.startswith("AU")]
    if INCLUDE_AUDIO:
        cols_include += l_audio_features
    if SUDS:
        cols_include += ["score_feat"]

    Y_fau = X_sub[cols_include]
    Y_fau["session_id"] = X_sub["session_id"]
    #count_sess_ids = Y_fau.groupby("session_id").count()["score_feat"]

    if sub in [4, 5, 7]:
        # remove columns that contain C_
        X_neural = X_neural[[c for c in X_neural.columns if not c.startswith("C_") and not "_C_" in c]]
        
    y_true_sub = []
    y_pred_sub = []
    for test_sess_id in X_sub["session_id"].unique():
        X_train = X_neural[X_neural["session_id"] != test_sess_id]
        X_test = X_neural[X_neural["session_id"] == test_sess_id]
        Y_fau_train = Y_fau[Y_fau["session_id"] != test_sess_id]
        Y_fau_test = Y_fau[Y_fau["session_id"] == test_sess_id]

        X_train_cca = X_train.drop(columns=["session_id"])  # 28 rows 331 cols
        Y_train_cca = Y_fau_train.drop(columns=["session_id"]) # 28 rows 42 cols
        X_test_cca = X_test.drop(columns=["session_id"]) # 10 rows 331 cols
        Y_test_cca = Y_fau_test.drop(columns=["session_id"]) # 10 rows 42 cols

        if INCLUDE_AU or INCLUDE_AUDIO:
            cols_to_move = [c for c in Y_train_cca.columns if c != "score_feat"]


            # concatenate all features except score_feat to X_train_cca and X_test_cca
            # and remove them from Y_test/Y_train
            X_train_cca = pd.concat([X_train_cca, Y_train_cca.drop(columns=["score_feat"])], axis=1)
            X_test_cca  = pd.concat([X_test_cca,  Y_test_cca.drop(columns=["score_feat"])],  axis=1)

            # Remove those columns from Y (train and test)
            Y_train_cca = Y_train_cca.drop(columns=cols_to_move)
            Y_test_cca  = Y_test_cca.drop(columns=cols_to_move)
            Y_train_cca = Y_train_cca[["score_feat"]].copy()
            Y_test_cca  = Y_test_cca[["score_feat"]].copy()

        idx_nan = X_train_cca.isna().any(axis=1) | Y_train_cca.isna().any(axis=1)
        X_train_cca = X_train_cca[~idx_nan]
        Y_train_cca = Y_train_cca[~idx_nan]
        idx_nan_test = X_test_cca.isna().any(axis=1) | Y_test_cca.isna().any(axis=1)
        if sum(idx_nan_test) == len(idx_nan_test):
            continue
        X_test_cca = X_test_cca[~idx_nan_test]
        Y_test_cca = Y_test_cca[~idx_nan_test]
    
        if NORMALIZE:
            x_scaler = StandardScaler().fit(X_train_cca)
            y_scaler = StandardScaler().fit(Y_train_cca)

            Xtr = x_scaler.transform(X_train_cca)
            Ytr = y_scaler.transform(Y_train_cca)
            Xte = x_scaler.transform(X_test_cca)
            Yte = y_scaler.transform(Y_test_cca)
        else:
            Xtr = X_train_cca.values
            Ytr = Y_train_cca.values
            Xte = X_test_cca.values
            Yte = Y_test_cca.values

        ## CCA 
        # cca = rcca.CCA(kernelcca=KERNEL_CCA, numCC=num_cc, reg=reg, verbose=False, ktype=ktype)
        # cca.train([Xtr, Ytr])
        # x_weights = cca.ws[0]
        # y_weights = cca.ws[1]
        # U = Xte @ x_weights
        # Y_te_pred = U @ np.linalg.pinv(y_weights)

        model = MLPRegressor(hidden_layer_sizes=(num_cc,), max_iter=500, random_state=42, alpha=reg)
        model.fit(Xtr, Ytr)
        Y_te_pred = model.predict(Xte)

        y_true_sub.append(Yte)
        y_pred_sub.append(Y_te_pred)
    
    y_true_sub = np.concatenate(y_true_sub).reshape(-1, 1)
    #if INCLUDE_AU is False and INCLUDE_AUDIO is False:
    #    y_pred_sub = np.concatenate(y_pred_sub).reshape(-1, 1)
    #else:
    y_pred_sub = np.concatenate(y_pred_sub).reshape(-1, 1)

    if SHUFFLE:
        np.random.shuffle(y_true_sub)
    df_res = []
    for i, col in enumerate(Y_fau_train.columns):
        if col != "score_feat":
            continue
        if col == "session_id":
            continue
        #r = np.corrcoef(y_true_sub[:, i], y_pred_sub[:, i])[0, 1]
        #print(f"sub {sub}, sess {test_sess_id}, AU {col}, r = {r:.3f}")
        # r, p = stats.spearmanr(y_true_sub[:, i], y_pred_sub[:, i])
        r, p = stats.pearsonr(y_true_sub[:, 0], y_pred_sub[:, 0])
        df_res.append({"subject": sub, "AU": col, "r": r, "p": p, "num_cc": num_cc, "reg": reg})

    df_res = pd.DataFrame(df_res)
    df_res["SUDS"] = SUDS
    df_res["INCLUDE_AU"] = INCLUDE_AU
    df_res["INCLUDE_AUDIO"] = INCLUDE_AUDIO
    df_res["NORMALIZE"] = NORMALIZE
    df_res["KERNEL_CCA"] = KERNEL_CCA
    df_res["SHUFFLE"] = SHUFFLE
    df_res["ktype"] = ktype
    if GET_RES is False:
        df_res.to_csv(save_name, index=False)
    return y_true_sub, y_pred_sub

if __name__ == "__main__":
    #idx_run = 1
    idx_run = int(sys.argv[1])

    pers_ = []
    SUDS = True

    combination_list = []

    for num_cc in num_ccs:
        for reg in regs:
            for sub in subs:
                for INCLUDE_AU in INCLUDE_AU_l:
                    for INCLUDE_AUDIO in INCLUDE_AUDIO_l:
                        for NORMALIZE in NORMALIZE_l:
                            for KERNEL_CCA in KERNEL_CCA_l:
                                for SHUFFLE_ in SHUFFLE_l:
                                    combination_list.append((num_cc, reg, sub, INCLUDE_AU, INCLUDE_AUDIO, NORMALIZE, KERNEL_CCA, SHUFFLE_))
    num_cc, reg, sub, INCLUDE_AU, INCLUDE_AUDIO, NORMALIZE, KERNEL_CCA, SHUFFLE = combination_list[idx_run]
    num_combinations = len(combination_list)
    if idx_run > num_combinations:
        print(f"Index {idx_run} out of range for {num_combinations} combinations")
        sys.exit(0)

    try:
        _ = run_sub(df_merged, sub, num_cc, reg, SUDS=SUDS, INCLUDE_AU=INCLUDE_AU, INCLUDE_AUDIO=INCLUDE_AUDIO, NORMALIZE=NORMALIZE, KERNEL_CCA=KERNEL_CCA, SHUFFLE=SHUFFLE)
    except Exception as e:
        print(f"Error for index {idx_run}, sub {sub}, num_cc {num_cc}, reg {reg}, INCLUDE_AU {INCLUDE_AU}, INCLUDE_AUDIO {INCLUDE_AUDIO}, NORMALIZE {NORMALIZE}, KERNEL_CCA {KERNEL_CCA}: {e}")
# Parallel processing using joblib
# results = Parallel(n_jobs=-1)(
#     delayed(run_sub)(df_merged, sub, num_cc, reg, SUDS=SUDS, INCLUDE_AU=INCLUDE_AU, INCLUDE_AUDIO=INCLUDE_AUDIO)
#     for num_cc in num_ccs
#     for reg in regs
#     for sub in subs
#     for INCLUDE_AU in INCLUDE_AU_l
#     for INCLUDE_AUDIO in INCLUDE_AUDIO_l
# )

# df_res = pd.concat(results)
# df_res["INCLUDE_AU"] = INCLUDE_AU
# df_res["INCLUDE_AUDIO"] = INCLUDE_AUDIO
# df_res["SUDS"] = SUDS

# df_res.to_csv("cca_eval/rccs_grid_suds_video_audio.csv", index=False)


# for num_cc in tqdm([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):  #, 2, 3, 4, 5]:
#     for reg in [1, 10, 100, 1000, 1000000]:  #, 1e4, 1e
#         reg = 1000
#         num_cc = 10
#         for sub in subs:
#             df_res = run_sub(df_merged, sub, num_cc, reg, SUDS=SUDS, INCLUDE_AU=INCLUDE_AU, INCLUDE_AUDIO=INCLUDE_AUDIO)
#             pers_.append(df_res)

#df_res = pd.concat(pers_)

# matrix = df_res.query("AU == 'score_feat'").groupby(["num_cc", "reg"])["r"].mean().unstack()

# plt.figure(figsize=(8, 6))
# sns.heatmap(matrix, annot=True, fmt=".3f", cmap="coolwarm", center=0)
# plt.title("Mean Pearson r for score_feat across subjects\nVarying number of CCA components and regularization")
# plt.xlabel("Regularization")
# plt.ylabel("Number of CCA Components")
# plt.tight_layout()
# plt.show()