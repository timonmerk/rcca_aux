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
import argparse
from sklearn import metrics

def compute_ccc(x,y):
    ''' Concordance Correlation Coefficient'''
    sxy = np.sum((x - x.mean())*(y - y.mean()))/x.shape[0]
    rhoc = 2*sxy / (np.var(x) + np.var(y) + (x.mean() - y.mean())**2)
    return rhoc

if len(sys.argv) > 1:
    DEBUG = False
else:
    DEBUG = True

l_audio_features = ["Loudness_sma3","alphaRatio_sma3","hammarbergIndex_sma3","slope0-500_sma3","slope500-1500_sma3","spectralFlux_sma3","mfcc1_sma3","mfcc2_sma3","mfcc3_sma3","mfcc4_sma3","F0semitoneFrom27.5Hz_sma3nz","jitterLocal_sma3nz","shimmerLocaldB_sma3nz","HNRdBACF_sma3nz","logRelF0-H1-H2_sma3nz","logRelF0-H1-A3_sma3nz","F1frequency_sma3nz","F1bandwidth_sma3nz","F1amplitudeLogRelF0_sma3nz","F2frequency_sma3nz","F2bandwidth_sma3nz","F2amplitudeLogRelF0_sma3nz","F3frequency_sma3nz","F3bandwidth_sma3nz","F3amplitudeLogRelF0_sma3nz","F0semitoneFrom27.5Hz_sma3nz_amean","F0semitoneFrom27.5Hz_sma3nz_stddevNorm","F0semitoneFrom27.5Hz_sma3nz_percentile20.0","F0semitoneFrom27.5Hz_sma3nz_percentile50.0","F0semitoneFrom27.5Hz_sma3nz_percentile80.0","F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2","F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope","F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope","F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope","F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope","loudness_sma3_amean","loudness_sma3_stddevNorm","loudness_sma3_percentile20.0","loudness_sma3_percentile50.0","loudness_sma3_percentile80.0","loudness_sma3_pctlrange0-2","loudness_sma3_meanRisingSlope","loudness_sma3_stddevRisingSlope","loudness_sma3_meanFallingSlope","loudness_sma3_stddevFallingSlope","spectralFlux_sma3_amean","spectralFlux_sma3_stddevNorm","mfcc1_sma3_amean","mfcc1_sma3_stddevNorm","mfcc2_sma3_amean","mfcc2_sma3_stddevNorm","mfcc3_sma3_amean","mfcc3_sma3_stddevNorm","mfcc4_sma3_amean","mfcc4_sma3_stddevNorm","jitterLocal_sma3nz_amean","jitterLocal_sma3nz_stddevNorm","shimmerLocaldB_sma3nz_amean","shimmerLocaldB_sma3nz_stddevNorm","HNRdBACF_sma3nz_amean","HNRdBACF_sma3nz_stddevNorm","logRelF0-H1-H2_sma3nz_amean","logRelF0-H1-H2_sma3nz_stddevNorm","logRelF0-H1-A3_sma3nz_amean","logRelF0-H1-A3_sma3nz_stddevNorm","F1frequency_sma3nz_amean","F1frequency_sma3nz_stddevNorm","F1bandwidth_sma3nz_amean","F1bandwidth_sma3nz_stddevNorm","F1amplitudeLogRelF0_sma3nz_amean","F1amplitudeLogRelF0_sma3nz_stddevNorm","F2frequency_sma3nz_amean","F2frequency_sma3nz_stddevNorm","F2bandwidth_sma3nz_amean","F2bandwidth_sma3nz_stddevNorm","F2amplitudeLogRelF0_sma3nz_amean","F2amplitudeLogRelF0_sma3nz_stddevNorm","F3frequency_sma3nz_amean","F3frequency_sma3nz_stddevNorm","F3bandwidth_sma3nz_amean","F3bandwidth_sma3nz_stddevNorm","F3amplitudeLogRelF0_sma3nz_amean","F3amplitudeLogRelF0_sma3nz_stddevNorm","alphaRatioV_sma3nz_amean","alphaRatioV_sma3nz_stddevNorm","hammarbergIndexV_sma3nz_amean","hammarbergIndexV_sma3nz_stddevNorm","slopeV0-500_sma3nz_amean","slopeV0-500_sma3nz_stddevNorm","slopeV500-1500_sma3nz_amean","slopeV500-1500_sma3nz_stddevNorm","spectralFluxV_sma3nz_amean","spectralFluxV_sma3nz_stddevNorm","mfcc1V_sma3nz_amean","mfcc1V_sma3nz_stddevNorm","mfcc2V_sma3nz_amean","mfcc2V_sma3nz_stddevNorm","mfcc3V_sma3nz_amean","mfcc3V_sma3nz_stddevNorm","mfcc4V_sma3nz_amean","mfcc4V_sma3nz_stddevNorm","alphaRatioUV_sma3nz_amean","hammarbergIndexUV_sma3nz_amean","slopeUV0-500_sma3nz_amean","slopeUV500-1500_sma3nz_amean","spectralFluxUV_sma3nz_amean","loudnessPeaksPerSec","VoicedSegmentsPerSec","MeanVoicedSegmentLengthSec","StddevVoicedSegmentLengthSec","MeanUnvoicedSegmentLength","StddevUnvoicedSegmentLength","equivalentSoundLevel_dBp","arousal","dominance","valence"] + [f"Dim {i}" for i in range(1024)]# + ["duration"]

if DEBUG is False:
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx_run", type=int, help="Index of the run")
    parser.add_argument("--run-suds", action="store_true")
    parser.add_argument("--region", type=str, default="all", choices=["SC", "C", "all"], help="Brain region to use")
    parser.add_argument("--output_folder", type=str, default="out")

    args = parser.parse_args()
    idx_run = args.idx_run
    RUN_SUDS = args.run_suds
    region = args.region
    out_folder_name = args.output_folder + ("_suds" if RUN_SUDS else "_rs")
    out_folder_name += f"_{region}"
else:
    idx_run = 0
    region = "C"
    RUN_SUDS = True
    out_folder_name = "debug_out_suds_rcca"

if not os.path.exists(f"/scratch/tm162/rcca_run/{out_folder_name}"):
    os.makedirs(f"/scratch/tm162/rcca_run/{out_folder_name}")

num_ccs = [1, 2, 5, 10, 15, 25]
regs = [0.01, 0.1, 1.0, 10, 100, 1000, 10000, 10000]

num_ccs = [1, 2, 5, 7, 10, 12, 15, 17, 20, 25, 30, 40, 50, 70, 80, 90, 100, ]
regs = [0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 70, 100, 200, 500, 1000, 2000, 5000, 10000]
subs = [4, 5, 7, 9, 10, 11, 12]
INCLUDE_AU_l = [True, False]
INCLUDE_AUDIO_l = [True, False]
SUDS = True
SHUFFLE_l = [True, False]
KERNEL_CCA_l = [False,]  # False
NORMALIZE_l = [True,]

def run_cv(X, Y, num_cc, reg, col_idx_check = None, NORMALIZE=False):

    sess_ids = X["session_id"].unique()
    y_pred = []
    y_true = []
    dim_correlations = np.empty((sess_ids.shape[0], num_cc), dtype=object)

    for i, sess_id in enumerate(sess_ids):
        X_train = X[X["session_id"] != sess_id].drop(columns=["session_id"])
        Y_train = Y[Y["session_id"] != sess_id].drop(columns=["session_id"])
        X_test = X[X["session_id"] == sess_id].drop(columns=["session_id"])
        Y_test = Y[Y["session_id"] == sess_id].drop(columns=["session_id"])

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

        cca = rcca.CCA(kernelcca=False, numCC=num_cc, reg=reg, verbose=False,)
        cca.train([X_train, Y_train])
        x_weights = cca.ws[0]
        y_weights = cca.ws[1]
        U = X_test @ x_weights
        Y_pred = U @ np.linalg.pinv(y_weights)
        y_pred.append(Y_pred[:, col_idx_check])
        y_true.append(Y_test[:, col_idx_check])

        # correlation of each canonical dimension with the chosen Y column
        for k in range(num_cc):
            dim_correlations[i, k] = U[:, k]

    y_pred = np.concatenate(y_pred)
    y_true =  np.concatenate(y_true)
    r, p = stats.pearsonr(y_true, y_pred)

    r_each_cc_dim = []
    for k in range(num_cc):
        r_dim, _ = stats.pearsonr(np.concatenate(dim_correlations[:, k]), y_true)
        r_each_cc_dim.append(r_dim)

    best_idv_cc_dim_corr = np.argmax(r_each_cc_dim)
    ccc = compute_ccc(y_true, y_pred)

    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)

    return r, p, ccc, mae, mse, best_idv_cc_dim_corr, 

def run_sub_get_hyperparams(df_merged, sub, num_cc, reg, SUDS=True, INCLUDE_AU=True, INCLUDE_AUDIO=True, NORMALIZE=True, out_folder_name="out2", RUN_SUDS: bool = False,
                            sess_test_id=None, ):
    save_name = f"/scratch/tm162/rcca_run/{out_folder_name}/rcca_sub_{sub}_sesstestidx_{sess_test_id}_suds_{SUDS}_au_{INCLUDE_AU}_audio_{INCLUDE_AUDIO}_reg_{reg}_numcc_{num_cc}.csv"
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

    if sub in [4, 5, 7]:
        # remove columns that contain C_
        X_neural = X_neural[[c for c in X_neural.columns if not c.startswith("C_") and not "_C_" in c]]

    #y_true_sub = []
    #y_pred_sub = []
    #for test_sess_id in X_sub["session_id"].unique():
    
    test_sess_id = sess_test_id
    X_train = X_neural[X_neural["session_id"] != test_sess_id]
    X_test = X_neural[X_neural["session_id"] == test_sess_id]
    Y_train = Y_fau[Y_fau["session_id"] != test_sess_id]
    Y_test = Y_fau[Y_fau["session_id"] == test_sess_id]

    idx_nan = X_train.isna().any(axis=1) | Y_train.isna().any(axis=1)
    X_train = X_train[~idx_nan]
    Y_train = Y_train[~idx_nan]
    idx_nan_test = X_test.isna().any(axis=1) | Y_test.isna().any(axis=1)
    if sum(idx_nan_test) == len(idx_nan_test):
        return None
    X_test = X_test[~idx_nan_test]
    Y_test = Y_test[~idx_nan_test]
    
    if RUN_SUDS:
        col_idx_check = Y_train.columns.get_loc("score_feat")
    else:
        col_idx_check = Y_train.columns.get_loc('YBOCS II Total Score')

    # Run cross-validation to find the best hyperparameters
    r,p, ccc, mae, mse, best_idv_cc_dim_corr = run_cv(X_train, Y_train, num_cc, reg, col_idx_check, NORMALIZE=NORMALIZE)
    df_res = pd.DataFrame([{"subject": sub, "r": r, "p": p}])
    df_res["SUDS"] = SUDS
    df_res["INCLUDE_AU"] = INCLUDE_AU
    df_res["INCLUDE_AUDIO"] = INCLUDE_AUDIO
    df_res["NORMALIZE"] = NORMALIZE
    df_res["sess_test_id"] = sess_test_id
    df_res["num_cc"] = num_cc
    df_res["reg"] = reg
    df_res["RUN_SUDS"] = RUN_SUDS
    df_res["region"] = region
    df_res["ccc"] = ccc
    df_res["mae"] = mae
    df_res["mse"] = mse
    df_res["best_idv_cc_dim_corr"] = best_idv_cc_dim_corr
    df_res.to_csv(save_name, index=False)

if __name__ == "__main__":

    df_merged, subs = utils.get_df_features(region, "all", READ_RS=not RUN_SUDS)
    pers_ = []
    SUDS = True
    combination_list = []

    for INCLUDE_AU in INCLUDE_AU_l:
        for INCLUDE_AUDIO in INCLUDE_AUDIO_l:
            for sub in subs:
                df_sub = df_merged.query("subject == @sub")
                if RUN_SUDS:
                    df_sub["date"] = df_sub["time"].dt.date
                    df_sub["session_id"] = df_sub["date"].astype("category").cat.codes
                else:
                    df_sub["session_id"] = (df_sub["subject"].astype(str) + "_" + df_sub["date"].astype(str)).astype("category").cat.codes
                for sess_test_id in df_sub["session_id"].unique():

                    for reg in regs:
                        #for num_cc in num_ccs:
                        combination_list.append({
                            "sub": sub,
                            "sess_test_id": sess_test_id,
                            #"num_cc": num_cc,
                            "reg": reg,
                            "SUDS": SUDS,
                            "INCLUDE_AU": INCLUDE_AU,
                            "INCLUDE_AUDIO": INCLUDE_AUDIO,
                        })
    df_comb = pd.DataFrame(combination_list)
    num_combinations = len(combination_list)
    if idx_run >= num_combinations:
        print(f"Index {idx_run} out of range for {num_combinations} combinations")
        sys.exit(0)

    sub, sess_test_id, SUDS, INCLUDE_AU, INCLUDE_AUDIO, reg = combination_list[idx_run]["sub"], combination_list[idx_run]["sess_test_id"], combination_list[idx_run]["SUDS"], combination_list[idx_run]["INCLUDE_AU"], combination_list[idx_run]["INCLUDE_AUDIO"], combination_list[idx_run]["reg"]

    try:
        for num_cc in num_ccs:
            _ = run_sub_get_hyperparams(df_merged, sub, num_cc, reg, SUDS=SUDS,
                    INCLUDE_AU=INCLUDE_AU, INCLUDE_AUDIO=INCLUDE_AUDIO,
                    out_folder_name=out_folder_name, RUN_SUDS=RUN_SUDS,
                    sess_test_id=sess_test_id)
        print(f"Completed index {idx_run}, sub {sub}, INCLUDE_AU {INCLUDE_AU}, INCLUDE_AUDIO {INCLUDE_AUDIO}")
    except Exception as e:
        print(f"Error for index {idx_run}, sub {sub}, INCLUDE_AU {INCLUDE_AU}, INCLUDE_AUDIO {INCLUDE_AUDIO}: {e}")

