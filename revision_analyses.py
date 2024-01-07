import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import process_raw_data

"""
Manage extra processing experimental data as part of the revision process of the paper. 
"""

__author__ = "Rony Hirschhorn"



"""

FRAME RATE ANALYSIS

"""
def count_non_empty(row):
    return sum(row[col] == row[col] for col in process_raw_data.WORDS)


def frame_rate_analysis_prep(path_to_experiment_df, save_path):
    data = pd.read_csv(path_to_experiment_df)
    data['num_of_words'] = data.apply(lambda row: count_non_empty(row), axis=1)
    data.to_csv(os.path.join(save_path, "experiment_df_words_corrected_count.csv"), index=False)

    # stats
    data['participant'] = pd.to_numeric(data['participant'])
    stats = data.groupby('participant').mean(numeric_only=True)
    stats.describe().to_csv(os.path.join(save_path, "experiment_df_words_corrected_stats.csv"))
    return



"""

CONFIDENCE ANALYSIS

"""


def confidence_IA_corr(path_to_iaconf_df, save_path, save_name,title):
    """
    mean confidence for word-IA words: for each word here (in each image), we have both the mean confidence in that
    word (in the context of the image, across all participants who provided it), and the IA of that word.
    """
    data = pd.read_csv(path_to_iaconf_df)
    # plot scatterplot with regression line
    ax = sns.regplot(x="mean confidence rating", y="word IA", data=data,
                     scatter_kws={"color": "black", "alpha": 0.5},
                     line_kws={"color": "blue"},
                     ci=99)  # confidence interval level is 99%
    plt.xticks(ticks=[1, 2, 3, 4, 5], labels=["1", "2", "3", "4", "5"], fontsize=12)
    plt.xlabel(xlabel="Mean Confidence Rating", fontsize=15)
    plt.yticks(fontsize=12)
    plt.ylabel(ylabel="Word IA", fontsize=15)
    plt.title(title, fontsize=15)
    ax.spines[['right', 'top']].set_visible(False)  # get rid of these parts of the frame
    plt.savefig(os.path.join(save_path, f"conf_ia_corr_{save_name}.svg"),
                format="svg", dpi=1000, bbox_inches='tight', pad_inches=0.01)

    return


if __name__ == "__main__":
    #frame_rate_analysis_prep(path_to_experiment_df=r"..\experiment_df_words_corrected.csv",
    #                    save_path=r"..")
    confidence_IA_corr(path_to_iaconf_df=r"..word_IA_confidence.csv",
                       save_path=r"..",
                       save_name="orig_67",
                       title="Original Study 67ms")
