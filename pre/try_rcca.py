import utils
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import rcca
from scipy import stats
import sys
import os

l_audio_features = ["Loudness_sma3","alphaRatio_sma3","hammarbergIndex_sma3","slope0-500_sma3","slope500-1500_sma3","spectralFlux_sma3","mfcc1_sma3","mfcc2_sma3","mfcc3_sma3","mfcc4_sma3","F0semitoneFrom27.5Hz_sma3nz","jitterLocal_sma3nz","shimmerLocaldB_sma3nz","HNRdBACF_sma3nz","logRelF0-H1-H2_sma3nz","logRelF0-H1-A3_sma3nz","F1frequency_sma3nz","F1bandwidth_sma3nz","F1amplitudeLogRelF0_sma3nz","F2frequency_sma3nz","F2bandwidth_sma3nz","F2amplitudeLogRelF0_sma3nz","F3frequency_sma3nz","F3bandwidth_sma3nz","F3amplitudeLogRelF0_sma3nz","F0semitoneFrom27.5Hz_sma3nz_amean","F0semitoneFrom27.5Hz_sma3nz_stddevNorm","F0semitoneFrom27.5Hz_sma3nz_percentile20.0","F0semitoneFrom27.5Hz_sma3nz_percentile50.0","F0semitoneFrom27.5Hz_sma3nz_percentile80.0","F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2","F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope","F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope","F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope","F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope","loudness_sma3_amean","loudness_sma3_stddevNorm","loudness_sma3_percentile20.0","loudness_sma3_percentile50.0","loudness_sma3_percentile80.0","loudness_sma3_pctlrange0-2","loudness_sma3_meanRisingSlope","loudness_sma3_stddevRisingSlope","loudness_sma3_meanFallingSlope","loudness_sma3_stddevFallingSlope","spectralFlux_sma3_amean","spectralFlux_sma3_stddevNorm","mfcc1_sma3_amean","mfcc1_sma3_stddevNorm","mfcc2_sma3_amean","mfcc2_sma3_stddevNorm","mfcc3_sma3_amean","mfcc3_sma3_stddevNorm","mfcc4_sma3_amean","mfcc4_sma3_stddevNorm","jitterLocal_sma3nz_amean","jitterLocal_sma3nz_stddevNorm","shimmerLocaldB_sma3nz_amean","shimmerLocaldB_sma3nz_stddevNorm","HNRdBACF_sma3nz_amean","HNRdBACF_sma3nz_stddevNorm","logRelF0-H1-H2_sma3nz_amean","logRelF0-H1-H2_sma3nz_stddevNorm","logRelF0-H1-A3_sma3nz_amean","logRelF0-H1-A3_sma3nz_stddevNorm","F1frequency_sma3nz_amean","F1frequency_sma3nz_stddevNorm","F1bandwidth_sma3nz_amean","F1bandwidth_sma3nz_stddevNorm","F1amplitudeLogRelF0_sma3nz_amean","F1amplitudeLogRelF0_sma3nz_stddevNorm","F2frequency_sma3nz_amean","F2frequency_sma3nz_stddevNorm","F2bandwidth_sma3nz_amean","F2bandwidth_sma3nz_stddevNorm","F2amplitudeLogRelF0_sma3nz_amean","F2amplitudeLogRelF0_sma3nz_stddevNorm","F3frequency_sma3nz_amean","F3frequency_sma3nz_stddevNorm","F3bandwidth_sma3nz_amean","F3bandwidth_sma3nz_stddevNorm","F3amplitudeLogRelF0_sma3nz_amean","F3amplitudeLogRelF0_sma3nz_stddevNorm","alphaRatioV_sma3nz_amean","alphaRatioV_sma3nz_stddevNorm","hammarbergIndexV_sma3nz_amean","hammarbergIndexV_sma3nz_stddevNorm","slopeV0-500_sma3nz_amean","slopeV0-500_sma3nz_stddevNorm","slopeV500-1500_sma3nz_amean","slopeV500-1500_sma3nz_stddevNorm","spectralFluxV_sma3nz_amean","spectralFluxV_sma3nz_stddevNorm","mfcc1V_sma3nz_amean","mfcc1V_sma3nz_stddevNorm","mfcc2V_sma3nz_amean","mfcc2V_sma3nz_stddevNorm","mfcc3V_sma3nz_amean","mfcc3V_sma3nz_stddevNorm","mfcc4V_sma3nz_amean","mfcc4V_sma3nz_stddevNorm","alphaRatioUV_sma3nz_amean","hammarbergIndexUV_sma3nz_amean","slopeUV0-500_sma3nz_amean","slopeUV500-1500_sma3nz_amean","spectralFluxUV_sma3nz_amean","loudnessPeaksPerSec","VoicedSegmentsPerSec","MeanVoicedSegmentLengthSec","StddevVoicedSegmentLengthSec","MeanUnvoicedSegmentLength","StddevUnvoicedSegmentLength","equivalentSoundLevel_dBp","arousal","dominance","valence"] + [f"Dim {i}" for i in range(1024)]# + ["duration"]

#out_folder_name = "out2_output_with_fau_au"
#RUN_SUDS = False
#RUN_SUDS = True

idx_run = 0
run_suds = 1
#idx_run = int(sys.argv[1])
#run_suds = int(sys.argv[2])

if run_suds == 0:
    RUN_SUDS = False
    out_folder_name = "16_12/out_rs_rcca_output_with_fau_au"
else:
    RUN_SUDS = True
    out_folder_name = "16_12/out2_output_with_fau_au"

if not os.path.exists(f"/scratch/tm162/rcca_run/{out_folder_name}"):
    os.makedirs(f"/scratch/tm162/rcca_run/{out_folder_name}")

num_ccs = [1, 2, 5, 10, 15, 25]
regs = [0.01, 0.1, 1.0, 10, 100, 1000, 10000, 10000]
subs = [4, 5, 7, 9, 10, 11, 12]
INCLUDE_AU_l = [True, False]
INCLUDE_AUDIO_l = [True, False]
SUDS = True
SHUFFLE_l = [True, False]
KERNEL_CCA_l = [False,]  # False
NORMALIZE_l = [True,]

def run_cv(X, Y, num_ccs, regs, col_idx_check = None, NORMALIZE=False):
    sess_ids = X["session_id"].unique()
    out_res = np.zeros([len(sess_ids), len(num_ccs), len(regs)])
    for sess_idx, sess in enumerate(sess_ids):
        X_train = X[X["session_id"] != sess].drop(columns=["session_id"])
        Y_train = Y[Y["session_id"] != sess].drop(columns=["session_id"])
        X_test = X[X["session_id"] == sess].drop(columns=["session_id"])
        Y_test = Y[Y["session_id"] == sess].drop(columns=["session_id"])

        if NORMALIZE:
            x_scaler = StandardScaler().fit(X_train)
            y_scaler = StandardScaler().fit(Y_train)
            X_train = x_scaler.transform(X_train)
            Y_train = y_scaler.transform(Y_train)
            X_test = x_scaler.transform(X_test)
            Y_test = y_scaler.transform(Y_test)
        else:
            X_train = X_train.values
            Y_train = Y_train.values
            X_test = X_test.values
            Y_test = Y_test.values
        for cc_idx, num_cc in enumerate(num_ccs):
            for reg_idx, reg in enumerate(regs):
                cca = rcca.CCA(kernelcca=False, numCC=num_cc, reg=reg, verbose=False,)
                cca.train([X_train, Y_train])
                x_weights = cca.ws[0]
                y_weights = cca.ws[1]
                U = X_test @ x_weights
                Y_pred = U @ np.linalg.pinv(y_weights)
                r, _ = stats.pearsonr(Y_test[:, col_idx_check], Y_pred[:, col_idx_check])
                out_res[sess_idx, cc_idx, reg_idx] = r

    out_res_ = np.nanmean(out_res, axis=0)
    best_idx = np.unravel_index(np.argmax(out_res_, axis=None), out_res_.shape)
    best_num_cc = num_ccs[best_idx[0]]
    best_reg = regs[best_idx[1]]

    return best_num_cc, best_reg


def run_sub(df_merged, sub, SUDS=True, INCLUDE_AU=True, INCLUDE_AUDIO=True, NORMALIZE=True,SHUFFLE=True, out_folder_name="out2", RUN_SUDS: bool = False,):
    save_name = f"/scratch/tm162/rcca_run/{out_folder_name}/rcca_sub_{sub}_suds_{SUDS}_au_{INCLUDE_AU}_audio_{INCLUDE_AUDIO}_norm_{NORMALIZE}_shuffle_{SHUFFLE}.csv"
    #if os.path.exists(save_name):
    #    print(f"File {save_name} already exists. Skipping.")
    #    sys.exit(0)

    X_sub = df_merged.query("subject == @sub")

    if RUN_SUDS:
        X_sub["date"] = X_sub["time"].dt.date
        X_sub["session_id"] = X_sub["date"].astype("category").cat.codes
        X_neural = X_sub[[c for c in X_sub.columns if c.startswith("SC_") or c.startswith("C_")]]
        X_neural["session_id"] = X_sub["session_id"]
    else:
        # set the session_id to be unique for each subject-date combination
        X_sub["session_id"] = (X_sub["subject"].astype(str) + "_" + X_sub["date"].astype(str)).astype("category").cat.codes
        X_neural = X_sub[[c for c in X_sub.columns if c.startswith("SC_") or c.startswith("C_")]]
        X_neural["session_id"] = X_sub["session_id"]

    cols_include = []
    if INCLUDE_AU:
        cols_include += [c for c in X_sub.columns if c.startswith("AU")]
    if INCLUDE_AUDIO:
        cols_include += l_audio_features
    if SUDS:
        if RUN_SUDS:
            cols_include += ["score_feat"]
        else:
            cols_include += ['YBOCS II Total Score']

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
        Y_train = Y_fau[Y_fau["session_id"] != test_sess_id]
        Y_test = Y_fau[Y_fau["session_id"] == test_sess_id]

        #X_train_cca = X_train.drop(columns=["session_id"])  # 28 rows 331 cols
        #Y_train_cca = Y_fau_train.drop(columns=["session_id"]) # 28 rows 42 cols
        #X_test_cca = X_test.drop(columns=["session_id"]) # 10 rows 331 cols
        #Y_test_cca = Y_fau_test.drop(columns=["session_id"]) # 10 rows 42 cols

        # get the nan columns in the first row of X_train_cca
        #nan_cols_X = X_train.columns[X_train_cca.iloc[0].isna()]
        

        # drop rows with NaNs
        idx_nan = X_train.isna().any(axis=1) | Y_train.isna().any(axis=1)
        X_train = X_train[~idx_nan]
        Y_train = Y_train[~idx_nan]
        idx_nan_test = X_test.isna().any(axis=1) | Y_test.isna().any(axis=1)
        if sum(idx_nan_test) == len(idx_nan_test):
            continue
        X_test = X_test[~idx_nan_test]
        Y_test = Y_test[~idx_nan_test]
        
        if RUN_SUDS:
            col_idx_check = Y_train.columns.get_loc("score_feat")
        else:
            col_idx_check = Y_train.columns.get_loc('YBOCS II Total Score')

        # Run cross-validation to find the best hyperparameters
        num_cc_best, reg_best = run_cv(X_train, Y_train, num_ccs, regs, col_idx_check, NORMALIZE=NORMALIZE)

        if NORMALIZE:
            # noramlize all columns except session_id but keep it in the dataframe
            x_scaler = StandardScaler().fit(X_train.drop(columns=["session_id"]))
            y_scaler = StandardScaler().fit(Y_train.drop(columns=["session_id"]))
            Xtr = x_scaler.transform(X_train.drop(columns=["session_id"]))
            Ytr = y_scaler.transform(Y_train.drop(columns=["session_id"]))
            Xte = x_scaler.transform(X_test.drop(columns=["session_id"]))
            Yte = y_scaler.transform(Y_test.drop(columns=["session_id"]))
            # add session_id back to the dataframe
            Xtr = pd.DataFrame(Xtr, columns=X_train.drop(columns=["session_id"]).columns)
            Xtr["session_id"] = X_train["session_id"].values
            Ytr = pd.DataFrame(Ytr, columns=Y_train.drop(columns=["session_id"]).columns)
            Ytr["session_id"] = Y_train["session_id"].values
            Xte = pd.DataFrame(Xte, columns=X_test.drop(columns=["session_id"]).columns)
            Xte["session_id"] = X_test["session_id"].values
            Yte = pd.DataFrame(Yte, columns=Y_test.drop(columns=["session_id"]).columns)
            Yte["session_id"] = Y_test["session_id"].values
        else:
            Xtr = X_train.copy()
            Ytr = Y_train.copy()
            Xte = X_test.copy()
            Yte = Y_test.copy()

        cca = rcca.CCA(kernelcca=False, numCC=num_cc_best, reg=reg_best, verbose=False,)
        cca.train([Xtr, Ytr])

        # Compute the correlation between the predicted and true values
        x_weights = cca.ws[0]
        y_weights = cca.ws[1]

        U = Xte @ x_weights
        Y_te_pred = U @ np.linalg.pinv(y_weights)

        y_true_sub.append(Yte)
        y_pred_sub.append(Y_te_pred)
    
    y_true_sub = np.vstack(y_true_sub)
    y_pred_sub = np.vstack(y_pred_sub)

    #if SHUFFLE:
    #    np.random.shuffle(y_true_sub)
    df_res = []
    for i, col in tqdm(enumerate(Y_train.columns)):
        #if col != "score_feat" and RUN_SUDS:
        #    continue
        #if col != 'YBOCS II Total Score' and not RUN_SUDS:
        #    continue
        if col == "session_id":
            continue
        #r = np.corrcoef(y_true_sub[:, i], y_pred_sub[:, i])[0, 1]
        #print(f"sub {sub}, sess {test_sess_id}, AU {col}, r = {r:.3f}")
        # r, p = stats.spearmanr(y_true_sub[:, i], y_pred_sub[:, i])
        if SHUFFLE is False:
            r, p = stats.pearsonr(y_true_sub[:, i], y_pred_sub[:, i])
        else:
            l_r = []
            l_p = []
            for _ in range(1000):  # to avoid occasional nan due to perfect correlation
                r, p = stats.pearsonr(np.random.permutation(y_true_sub[:, i]), y_pred_sub[:, i])
                l_r.append(r)
                l_p.append(p)
            r = np.mean(l_r)
            p = np.mean(l_p)
        df_res.append({"subject": sub, "AU": col, "r": r, "p": p})

    df_res = pd.DataFrame(df_res)
    df_res["SUDS"] = SUDS
    df_res["INCLUDE_AU"] = INCLUDE_AU
    df_res["INCLUDE_AUDIO"] = INCLUDE_AUDIO
    df_res["NORMALIZE"] = NORMALIZE
    df_res["SHUFFLE"] = SHUFFLE
    df_res["RUN_SUDS"] = RUN_SUDS
    df_res.to_csv(save_name, index=False)
    return df_res

if __name__ == "__main__":

    df_merged, subs = utils.get_df_features("all", "all", READ_RS=not RUN_SUDS)

    pers_ = []
    SUDS = True

    combination_list = []

    #for num_cc in num_ccs:
    #    for reg in regs:
    for sub in subs:
        for INCLUDE_AU in INCLUDE_AU_l:
            for INCLUDE_AUDIO in INCLUDE_AUDIO_l:
                for NORMALIZE in NORMALIZE_l:
                    for SHUFFLE_ in SHUFFLE_l:
                        combination_list.append((sub, INCLUDE_AU, INCLUDE_AUDIO, NORMALIZE, SHUFFLE_))
    sub, INCLUDE_AU, INCLUDE_AUDIO, NORMALIZE, SHUFFLE = combination_list[idx_run]
    num_combinations = len(combination_list)
    if idx_run >= num_combinations:
        print(f"Index {idx_run} out of range for {num_combinations} combinations")
        sys.exit(0)

    try:
        #_ = run_sub(df_merged, sub=10, num_cc=10, reg=10000, SUDS=True, INCLUDE_AU=False, INCLUDE_AUDIO=False, NORMALIZE=True, KERNEL_CCA=False, SHUFFLE=True, out_folder_name=out_folder_name, RUN_SUDS=RUN_SUDS)
        #_ = run_sub(df_merged, sub=10, num_cc=10, reg=10000, SUDS=True, INCLUDE_AU=True, INCLUDE_AUDIO=True, NORMALIZE=True, KERNEL_CCA=False, SHUFFLE=True, out_folder_name=out_folder_name, RUN_SUDS=RUN_SUDS)
        _ = run_sub(df_merged, sub, SUDS=SUDS, INCLUDE_AU=INCLUDE_AU, INCLUDE_AUDIO=INCLUDE_AUDIO, NORMALIZE=NORMALIZE, SHUFFLE=SHUFFLE, out_folder_name=out_folder_name, RUN_SUDS=RUN_SUDS)
    except Exception as e:
        print(f"Error for index {idx_run}, sub {sub}, INCLUDE_AU {INCLUDE_AU}, INCLUDE_AUDIO {INCLUDE_AUDIO}, NORMALIZE {NORMALIZE}: {e}")

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