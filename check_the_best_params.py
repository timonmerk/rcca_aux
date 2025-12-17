import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# import pdf from matplotlib
from matplotlib.backends.backend_pdf import PdfPages

pdf_pages = PdfPages("figures/best_params_heatmap.pdf")
for suds_ in ["rs", "suds"]:
    for loc in ["all", "C", "SC"]:
        df = pd.read_csv(f"out_{suds_}_{loc}.csv")

        df["cond"] = "None"
        df.loc[(df["SUDS"]==True) & (df["INCLUDE_AUDIO"]==False) & (df["INCLUDE_AU"]==False), "cond"] = "SUDS"
        df.loc[(df["SUDS"]==True) & (df["INCLUDE_AUDIO"]==True) & (df["INCLUDE_AU"]==False), "cond"] = "SUDS + Audio"
        df.loc[(df["SUDS"]==True) & (df["INCLUDE_AUDIO"]==False) & (df["INCLUDE_AU"]==True), "cond"] = "SUDS + AU"
        df.loc[(df["SUDS"]==True) & (df["INCLUDE_AUDIO"]==True) & (df["INCLUDE_AU"]==True), "cond"] = "SUDS + AU + Audio"

        max_idex = df.groupby(["cond", "sess_test_id", "subject"])["r"].idxmax()
        df_best = df.loc[max_idex].reset_index(drop=True)

        # make a heatmap of the best params for "SUDS" and "SUDS + AU + Audio"
        # count the occurrences of each param combination
        params = ["num_cc", "reg",]
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        for i, cond in enumerate(["SUDS", "SUDS + AU + Audio"]):
            df_cond = df_best.query("cond == @cond")
            param_counts = df_cond.groupby(params).size().unstack(fill_value=0)

            sns.heatmap(param_counts, annot=True, fmt="d", cmap="Blues", ax=axes[i])
            axes[i].set_title(f"Best Params for {cond}")
            axes[i].set_xlabel("reg")
            axes[i].set_ylabel("num_cc")
            plt.title(cond)
        plt.tight_layout()
        pdf_pages.savefig(fig)
pdf_pages.close()
