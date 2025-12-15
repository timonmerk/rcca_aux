from try_mlp_decode_incl_au_vid_as_features import run_sub
import pandas as pd
import os
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import utils
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages

folder_ = "out_mlp_aux_2"  # out_shuffle
num_cc = 1
reg = 100
NORMAMLIZE = True
KERNEL_CCA = False
INCLUDE_AU = True
INCLUDE_AUDIO = True
SUDS = True
SHUFFLE = True
GET_RES = True

df_merged, subs = utils.get_df_features("all", "all")

y_preds = []
y_trues = []
rs = []
ps = []

for sub in subs:
    y_true, y_pred = run_sub(df_merged, sub=sub, num_cc=num_cc, reg=reg, SUDS=SUDS, INCLUDE_AU=INCLUDE_AU, INCLUDE_AUDIO=INCLUDE_AUDIO,
                         NORMALIZE=NORMAMLIZE, KERNEL_CCA=KERNEL_CCA, SHUFFLE=SHUFFLE, GET_RES=GET_RES)
    r, _p = stats.pearsonr(y_true, y_pred)
    y_trues.append(y_true)
    y_preds.append(y_pred)
    rs.append(r)
    ps.append(_p)

Z_SCORE = True
plt.figure(figsize=(10, 15))
for i in range(len(subs)):
    plt.subplot(len(subs), 1, i+1)
    pr_ = stats.zscore(np.squeeze(y_preds[i]))
    tr_ = stats.zscore(np.squeeze(y_trues[i]))
    if not Z_SCORE:
        pr_ = np.squeeze(y_preds[i])
        tr_ = np.squeeze(y_trues[i])
    plt.plot(pr_, label="Predicted SUDS")
    plt.plot(tr_, label="True SUDS")
    plt.legend()
    plt.title(f"Sub {subs[i]}: r = {rs[i][0]:.2f}, p = {ps[i][0]:.4f}")
    plt.xlabel("True SUDS")
    plt.ylabel("Predicted SUDS")
plt.tight_layout()
plt.savefig(f"{folder_}_sub_preds_numcc_{num_cc}_reg_{reg}_au_{INCLUDE_AU}_audio_{INCLUDE_AUDIO}_shuffle_{SHUFFLE}.pdf")
