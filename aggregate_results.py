import os
import pandas as pd
import numpy as np
import plotter

"""
Manage the aggregation across the six experiments in the two studies:
- plot word IAs across experiments (this is shared to all experiments in the same study)
- plot word frequencies, separately per experiment
- plot word confidence across experiments, separately per experiment

The following code is run separately then the pre-processing performed separately for each experiment. Here, 
we rely on all experiments already being run, pre-processed and analyzed. This module just aggregates across experiments
to present them together. Therefore, it is important to first run the "manage_analysis" module prior to this. 
"""

__author__ = "Rony Hirschhorn"


REPLICATION = "Replication"
BLUR = "Blurred"
BW = "Black & White"
NAMING_DICT = {"replication": REPLICATION, "orig": REPLICATION,
               "blurred": BLUR, "blurry": BLUR, "Blurry": BLUR,
               "blackwhite": BW, "Black White": BW}
EXP_ORDER = [REPLICATION, BLUR, BW]
EXP_VERSION = "version"
EXP_VERSION_NAME = "Experiment"
IA_COL = "word IA"
IA_NAME = "Word IA"
FREQ_COL = "word freq in English language"
FREQ_NAME = "Word Frequency"
CONF_COL = "mean confidence rating"
CONF_NAME = "Confidence"
VERIDICALITY = "existInImage"
WORDS_PER_IMAGE = "total response words per stim"
IMAGE_NAME_COL = "image"

COLORS = {REPLICATION: "#006BA6", BLUR: "#05B3B3", BW: "#41464d"}

WORD_FREQ_FILE = "word_frequency_count"
WORD_CONF_FILE = "word_IA_confidence"


def plot_word_ia(word_ia_all_path):
    save_path = word_ia_all_path.rsplit("/", 1)[0]  # save path is the dir where the loaded file is
    save_name = word_ia_all_path.rsplit("/", 1)[-1].split('.')[0]  # name of plot is identical to the file it came from
    word_ia_all = pd.read_csv(word_ia_all_path)
    word_ia_all[EXP_VERSION].replace(NAMING_DICT, inplace=True)  # uniform naming
    plotter.plot_raincloud(df=word_ia_all, data_col_name=IA_COL, group_col_name=EXP_VERSION,
                           group_order=EXP_ORDER, group_spacing=0.5,
                           group_color_dict=COLORS,
                           y_title=IA_NAME, x_title=EXP_VERSION_NAME,
                           save_path=save_path, save_name=save_name,
                           marker_size=50, marker_alpha=0.25, marker_spread=0.2,
                           violin_width=0.35, violin_alpha=0.65, ymin=0.7, ymax=1.05, yskip=0.1)
    return


def plot_word_freq(word_freq_path):
    exps = [f for f in os.listdir(word_freq_path) if f.endswith(".csv") and WORD_FREQ_FILE in f]
    for exp in ["replication", "blurred", "blackwhite"]:
        exp_file = [item for item in exps if exp in item][0]
        exp_df = pd.read_csv(os.path.join(word_freq_path, exp_file))
        save_name = exp_file.split('.')[0]
        plotter.plot_corr(df=exp_df, x_col=IA_COL, x_name=IA_NAME, y_col=FREQ_COL, y_name=FREQ_NAME,
                          color=COLORS[NAMING_DICT[exp]], save_name=save_name, save_path=word_freq_path,
                          title=NAMING_DICT[exp], xmin=0.7, xmax=1.05, xskip=0.1,
                          ymin=1.0, ymax=7.01, yskip=1, marker_size=80, marker_alpha=0.15)
    return


def plot_word_conf(word_conf_path):
    exps = [f for f in os.listdir(word_conf_path) if f.endswith(".csv") and WORD_CONF_FILE in f]
    for exp in ["replication", "blurred", "blackwhite"]:
        exp_file = [item for item in exps if exp in item][0]
        exp_df = pd.read_csv(os.path.join(word_conf_path, exp_file))
        save_name = exp_file.split('.')[0]
        plotter.plot_corr(df=exp_df, x_col=IA_COL, x_name=IA_NAME, y_col=CONF_COL, y_name=CONF_NAME,
                          color=COLORS[NAMING_DICT[exp]], save_name=save_name, save_path=word_conf_path,
                          title=NAMING_DICT[exp], xmin=0.7, xmax=1.05, xskip=0.1,
                          ymin=1.0, ymax=5.01, yskip=1, marker_size=80, marker_alpha=0.15)
    return


def manage_plots(word_ia_all_path, word_freq_path, word_conf_path):
    """
    This method manages all the plots of this module.

    word_ia_all_path: For the plot of word IA across experiments, "plot_word_ia" expects a path to a csv file with the
    same structure as the one the "manage_analysis" module outputs ("experiment_IA_word.csv"), with a slight change -
    as this is collapsed across all experiments of interest, the file is expected to have an EXP_VERSION column,
    with the name of the experiment the IA data is taken from. So "word_ia_all_path" is a path to the collapsed
    "experiment_IA_word" file.

    word_freq_path: as word frequencies are of interest to our work separately for each experiment, no need to collapse
    anything here. Therefore, this is a path to a DIRECTORY, which is expected to contain "word_frequency_count.csv"
    files - the outputs of the "manage_analysis" module. For each such file, "plot_word_freq" method will plot word
    IAs against their previously-calculated frequency in the English language.

    plot_word_conf: same for confidence - no collapsing across experiments here either, so this is a path to a folder
    containing all previously-generated "word_IA_confidence.csv" files.

    """
    plot_word_ia(word_ia_all_path)
    plot_word_freq(word_freq_path)
    plot_word_conf(word_conf_path)
    return


def non_existent_words(freq_file_path, word_count_file_path, save_path):
    """
    This generates statistics about the non-veridical words provided in each experiment.
    The paths here are AGGREGATED across all experiments of interest, and therefore, the csvs are expected to have a
    "EXP_VERSION" column where the experiment is specified.
    In addition, as here we are interested in descriptives of non-veridical words, another key column is VERIDICALITY,
    where the following mapping is expected:
    1=the word appears in the image,
    0=the word doesn't appear in the image, OR
    empty=the word doesn't describe something tangible that can appear or not appear in the image (e.g. the word
    "beautiful" as a response to an image).

    freq_file_path: this is the "word_frequency_count.csv" file which is an output of the processing module,
    only (1) collapsed across all experiments (i.e., containing EXP_VERSION column) and (2) including a column denoting
    for each word if it exists in the image (VERIDICALITY).

    word_count_file_path: this is "analyze_data" module output file "words_per_image_raw.csv". Again, the expectation
    is that it will be collapsed and contain the EXP_VERSION column. This file contains for each image the total number
    of words provided for this image (WORDS_PER_IMAGE column).

    save_path: the path to which the descriptives of non-existent words per experiment should be saved (a csv for means,
    and another for sds).
    """
    words_ia_df = pd.read_csv(freq_file_path)
    words_ia_df[IMAGE_NAME_COL] = words_ia_df[IMAGE_NAME_COL].str.replace(".png", "").str.replace(".jpg", "")

    ia_counts = words_ia_df.groupby([EXP_VERSION, IMAGE_NAME_COL]).count()["word"].reset_index(drop=False)
    ia_counts.rename(columns={"word": "words with IA"}, inplace=True)

    existence_counts = words_ia_df.groupby([EXP_VERSION, IMAGE_NAME_COL])[VERIDICALITY].value_counts(dropna=False).unstack().reset_index(drop=False)
    existence_counts.rename(columns={np.nan: "conceptual", 0.0: "not in image", 1.0: "appears in image"}, inplace=True)
    # in case it didn't work
    existence_counts.columns = existence_counts.columns.fillna("conceptual")

    word_counts_df = pd.read_csv(word_count_file_path)
    word_counts_df.rename(columns={"stim_name": IMAGE_NAME_COL}, inplace=True)

    word_stats_df = pd.merge(word_counts_df, ia_counts, on=[EXP_VERSION, IMAGE_NAME_COL])
    word_stats_df = pd.merge(word_stats_df, existence_counts, on=[EXP_VERSION, IMAGE_NAME_COL])
    word_stats_df["pcnt IA out of provided"] = 100 * word_stats_df["words with IA"] / word_stats_df[WORDS_PER_IMAGE]
    word_stats_df["pcnt nonexistent out of IA"] = 100 * word_stats_df["not in image"] / word_stats_df["words with IA"]
    word_stats_df.to_csv(os.path.join(save_path, "word_counts_types.csv"), index=False)
    word_stats_df_noimage = word_stats_df.drop(columns=[IMAGE_NAME_COL], inplace=False)
    mean_df = word_stats_df_noimage.groupby([EXP_VERSION]).mean()
    mean_df.to_csv(os.path.join(save_path, "word_counts_types_mean.csv"), index=True)
    std_df = word_stats_df_noimage.groupby([EXP_VERSION]).std()
    std_df.to_csv(os.path.join(save_path, "word_counts_types_sd.csv"), index=True)

    return


if __name__ == "__main__":
    # plot aggregated results
    manage_plots(
    word_ia_all_path=r"..\aggregation\word_ia\experiment_IA_word_all.csv",
    word_freq_path=r"..\aggregation\word_freq",
    word_conf_path=r"..\aggregation\word_conf")
    # analyze non-existent words across all experiments.
    non_existent_words(freq_file_path=r"..\word_frequency_count_edited.csv",
                       word_count_file_path=r"..\words_per_image_raw_all.csv",
                       save_path=r"..\save_folder")