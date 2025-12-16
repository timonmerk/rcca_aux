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

l_audio_features = ["Loudness_sma3","alphaRatio_sma3","hammarbergIndex_sma3","slope0-500_sma3","slope500-1500_sma3","spectralFlux_sma3","mfcc1_sma3","mfcc2_sma3","mfcc3_sma3","mfcc4_sma3","F0semitoneFrom27.5Hz_sma3nz","jitterLocal_sma3nz","shimmerLocaldB_sma3nz","HNRdBACF_sma3nz","logRelF0-H1-H2_sma3nz","logRelF0-H1-A3_sma3nz","F1frequency_sma3nz","F1bandwidth_sma3nz","F1amplitudeLogRelF0_sma3nz","F2frequency_sma3nz","F2bandwidth_sma3nz","F2amplitudeLogRelF0_sma3nz","F3frequency_sma3nz","F3bandwidth_sma3nz","F3amplitudeLogRelF0_sma3nz","F0semitoneFrom27.5Hz_sma3nz_amean","F0semitoneFrom27.5Hz_sma3nz_stddevNorm","F0semitoneFrom27.5Hz_sma3nz_percentile20.0","F0semitoneFrom27.5Hz_sma3nz_percentile50.0","F0semitoneFrom27.5Hz_sma3nz_percentile80.0","F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2","F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope","F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope","F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope","F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope","loudness_sma3_amean","loudness_sma3_stddevNorm","loudness_sma3_percentile20.0","loudness_sma3_percentile50.0","loudness_sma3_percentile80.0","loudness_sma3_pctlrange0-2","loudness_sma3_meanRisingSlope","loudness_sma3_stddevRisingSlope","loudness_sma3_meanFallingSlope","loudness_sma3_stddevFallingSlope","spectralFlux_sma3_amean","spectralFlux_sma3_stddevNorm","mfcc1_sma3_amean","mfcc1_sma3_stddevNorm","mfcc2_sma3_amean","mfcc2_sma3_stddevNorm","mfcc3_sma3_amean","mfcc3_sma3_stddevNorm","mfcc4_sma3_amean","mfcc4_sma3_stddevNorm","jitterLocal_sma3nz_amean","jitterLocal_sma3nz_stddevNorm","shimmerLocaldB_sma3nz_amean","shimmerLocaldB_sma3nz_stddevNorm","HNRdBACF_sma3nz_amean","HNRdBACF_sma3nz_stddevNorm","logRelF0-H1-H2_sma3nz_amean","logRelF0-H1-H2_sma3nz_stddevNorm","logRelF0-H1-A3_sma3nz_amean","logRelF0-H1-A3_sma3nz_stddevNorm","F1frequency_sma3nz_amean","F1frequency_sma3nz_stddevNorm","F1bandwidth_sma3nz_amean","F1bandwidth_sma3nz_stddevNorm","F1amplitudeLogRelF0_sma3nz_amean","F1amplitudeLogRelF0_sma3nz_stddevNorm","F2frequency_sma3nz_amean","F2frequency_sma3nz_stddevNorm","F2bandwidth_sma3nz_amean","F2bandwidth_sma3nz_stddevNorm","F2amplitudeLogRelF0_sma3nz_amean","F2amplitudeLogRelF0_sma3nz_stddevNorm","F3frequency_sma3nz_amean","F3frequency_sma3nz_stddevNorm","F3bandwidth_sma3nz_amean","F3bandwidth_sma3nz_stddevNorm","F3amplitudeLogRelF0_sma3nz_amean","F3amplitudeLogRelF0_sma3nz_stddevNorm","alphaRatioV_sma3nz_amean","alphaRatioV_sma3nz_stddevNorm","hammarbergIndexV_sma3nz_amean","hammarbergIndexV_sma3nz_stddevNorm","slopeV0-500_sma3nz_amean","slopeV0-500_sma3nz_stddevNorm","slopeV500-1500_sma3nz_amean","slopeV500-1500_sma3nz_stddevNorm","spectralFluxV_sma3nz_amean","spectralFluxV_sma3nz_stddevNorm","mfcc1V_sma3nz_amean","mfcc1V_sma3nz_stddevNorm","mfcc2V_sma3nz_amean","mfcc2V_sma3nz_stddevNorm","mfcc3V_sma3nz_amean","mfcc3V_sma3nz_stddevNorm","mfcc4V_sma3nz_amean","mfcc4V_sma3nz_stddevNorm","alphaRatioUV_sma3nz_amean","hammarbergIndexUV_sma3nz_amean","slopeUV0-500_sma3nz_amean","slopeUV500-1500_sma3nz_amean","spectralFluxUV_sma3nz_amean","loudnessPeaksPerSec","VoicedSegmentsPerSec","MeanVoicedSegmentLengthSec","StddevVoicedSegmentLengthSec","MeanUnvoicedSegmentLength","StddevUnvoicedSegmentLength","equivalentSoundLevel_dBp","arousal","dominance","valence"] + [f"Dim {i}" for i in range(1024)]# + ["duration"]

if len(sys.argv) > 0:
    DEBUG = True
else:
    DEBUG = False

if DEBUG is False:
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx_run', type=int, required=True, help='Index of the run')
    parser.add_argument('--run_suds', action='store_true', help='Whether to run SUDS or RS')
    parser.add_argument('--region', type=str, default='all', help='Region to use: SC, C, or all')
    parser.add_argument('output_folder', type=str, help='Output folder name', default="outercv")

    args = parser.parse_args()
    idx_run = args.idx_run
    RUN_SUDS = args.run_suds
    region = args.region
    out_folder_name = args.output_folder + ("_suds" if RUN_SUDS else "_rs") + f"_{region}"
    out_folder_name += f"_{region}"
else:
    idx_run = 0
    region = "C"
    RUN_SUDS = True
    out_folder_name = "debug_outercv"

gs_name = "out_" + ("suds" if RUN_SUDS else "rs") + f"_{region}.csv"
df_gs_res = pd.read_csv("/scratch/tm162/rcca_run/" + gs_name)

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

def run_sub(df_merged, sub, SUDS=True, INCLUDE_AU=True, INCLUDE_AUDIO=True, NORMALIZE=True,
            out_folder_name="out2", RUN_SUDS: bool = False,):
    save_name = f"/scratch/tm162/rcca_run/{out_folder_name}/rcca_sub_{sub}_suds_{SUDS}_au_{INCLUDE_AU}_audio_{INCLUDE_AUDIO}_norm_{NORMALIZE}.csv"
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

        X_train  = X_train.drop(columns=["session_id"])  # 28 rows 331 cols
        Y_train  = Y_train.drop(columns=["session_id"]) # 28 rows 42 cols
        X_test = X_test.drop(columns=["session_id"]) # 10 rows 331 cols
        Y_test = Y_test.drop(columns=["session_id"]) # 10 rows 42 cols

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
        
        if NORMALIZE:
            scaler_X = StandardScaler()
            scaler_X.fit(X_train)
            X_train = pd.DataFrame(scaler_X.transform(X_train), columns=X_train.columns, index=X_train.index)
            X_test = pd.DataFrame(scaler_X.transform(X_test), columns=X_test.columns, index=X_test.index)

            scaler_Y = StandardScaler()
            Y_train = pd.DataFrame(scaler_Y.fit_transform(Y_train), columns=Y_train.columns, index=Y_train.index)
            Y_test = pd.DataFrame(scaler_Y.transform(Y_test), columns=Y_test.columns, index=Y_test.index)



        df_q = df_gs_res.query("subject == @sub and sess_test_id == @test_sess_id and INCLUDE_AU == @INCLUDE_AU and INCLUDE_AUDIO == @INCLUDE_AUDIO and SUDS == @SUDS")
        try:
            idx_best = df_q["r"].idxmax()
            idx_best = df_q["p"].idxmin()
        except:
            continue
        num_cc_best = df_q.loc[idx_best, "num_cc"]
        reg_best = df_q.loc[idx_best, "reg"]

        cca = rcca.CCA(kernelcca=False, numCC=num_cc_best, reg=reg_best, verbose=False,)
        cca.train([X_train, Y_train])

        # Compute the correlation between the predicted and true values
        x_weights = cca.ws[0]
        y_weights = cca.ws[1]

        U = X_test @ x_weights
        Y_te_pred = U @ np.linalg.pinv(y_weights)

        y_true_sub.append(Y_test)
        y_pred_sub.append(Y_te_pred)
    
    if len(y_true_sub) == 0:
        print(f"No data for sub {sub}, SUDS {SUDS}, INCLUDE_AU {INCLUDE_AU}, INCLUDE_AUDIO {INCLUDE_AUDIO}")
        return pd.DataFrame()
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
        r, p = stats.pearsonr(y_true_sub[:, i], y_pred_sub[:, i])
        df_res.append({"subject": sub, "AU": col, "r": r, "p": p})

    df_res = pd.DataFrame(df_res)
    df_res["SUDS"] = SUDS
    df_res["INCLUDE_AU"] = INCLUDE_AU
    df_res["INCLUDE_AUDIO"] = INCLUDE_AUDIO
    df_res["RUN_SUDS"] = RUN_SUDS
    df_res.to_csv(save_name, index=False)
    return df_res

if __name__ == "__main__":

    df_merged, subs = utils.get_df_features(region, "all", READ_RS=not RUN_SUDS)
    pers_ = []
    SUDS = True
    combination_list = []

    for INCLUDE_AU in INCLUDE_AU_l:
        for INCLUDE_AUDIO in INCLUDE_AUDIO_l:
            for sub in subs:
    
                combination_list.append({
                    "sub": sub,
                    "SUDS": SUDS,
                    "INCLUDE_AU": INCLUDE_AU,
                    "INCLUDE_AUDIO": INCLUDE_AUDIO,
                })
    df_comb = pd.DataFrame(combination_list)

    num_combinations = len(combination_list)
    if idx_run >= num_combinations:
        print(f"Index {idx_run} out of range for {num_combinations} combinations")
        sys.exit(0)

    try:
        #_ = run_sub(df_merged, sub=10, num_cc=10, reg=10000, SUDS=True, INCLUDE_AU=False, INCLUDE_AUDIO=False, NORMALIZE=True, KERNEL_CCA=False, SHUFFLE=True, out_folder_name=out_folder_name, RUN_SUDS=RUN_SUDS)
        #_ = run_sub(df_merged, sub=10, num_cc=10, reg=10000, SUDS=True, INCLUDE_AU=True, INCLUDE_AUDIO=True, NORMALIZE=True, KERNEL_CCA=False, SHUFFLE=True, out_folder_name=out_folder_name, RUN_SUDS=RUN_SUDS)
        
#        for idx_run in tqdm(range(num_combinations)):
        sub, SUDS, INCLUDE_AU, INCLUDE_AUDIO = combination_list[idx_run]["sub"],combination_list[idx_run]["SUDS"], combination_list[idx_run]["INCLUDE_AU"], combination_list[idx_run]["INCLUDE_AUDIO"]

        _ = run_sub(df_merged, sub, SUDS=SUDS,
                                        INCLUDE_AU=INCLUDE_AU, INCLUDE_AUDIO=INCLUDE_AUDIO,
                                        out_folder_name=out_folder_name, RUN_SUDS=RUN_SUDS,
                                        )
        print("done")
    except Exception as e:
        print(f"Error for index {idx_run}, sub {sub}, INCLUDE_AU {INCLUDE_AU}, INCLUDE_AUDIO {INCLUDE_AUDIO}: {e}")
