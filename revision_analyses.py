import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import functools as ft
import process_raw_data

"""
Manage extra processing experimental data as part of the revision process of the paper. 
"""

__author__ = "Rony Hirschhorn"



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


"""
Existence analyses
"""


def ia_in_image(word, ia_word_list):
    if word != word:  # word is empty - no word was actually provided
        return 0
    if word in ia_word_list:  # if a word is rare, it doesn't have an IA score
        return 1
    return 0


def count_ia(row, non_rare_df):
    curr_image = row['stim_name'].split('/')[-1].split('.')[0]
    non_rare_df_image = non_rare_df[non_rare_df['image'] == curr_image]
    ia_word_list = non_rare_df_image['word'].tolist()
    have_ia = sum(ia_in_image(row[col], ia_word_list) for col in process_raw_data.WORDS)
    return have_ia


def not_exists_in_image(word, curr_image, exist_df, rare_word_list):
    if word != word:  # the word is nan (i.e., not provided a word to begin with)
        return 0
    if word in rare_word_list:  # the word is rare, has no IA score, therefore not tagged as existing or non-existing
        return 0
    relevant_df = exist_df[exist_df['image'] == curr_image]
    relevant_entry = relevant_df[relevant_df['word'] == word]  # this should never be empty at this stage
    if relevant_entry.empty:  # some other error, shouldn't happen
        return 0
    exist = relevant_entry['existInImage'].item()
    if exist != exist:
        return 0
    """
    else, the word either got 1 (exists in image) or 0 (non). As we want to sum the NON existing, we need to return 
    '1' if the word DOESN'T exist
    """
    return int(exist) ^ 1


def count_nonexist(row, exist_df, rare_df):
    curr_image = row['stim_name'].split('/')[-1].split('.')[0]
    rare_df_image = rare_df[rare_df['image'] == curr_image]
    rare_word_list = rare_df_image['word'].tolist()
    existing = sum(not_exists_in_image(row[col], curr_image, exist_df, rare_word_list) for col in process_raw_data.WORDS)
    return existing


def nonexistent_posthoc_analysis(version, exist_file, lemmatized_file, rare_file, save_path):
    # A dataframe where every word with a word-IA in the image is marked as existing or not
    exist_df = pd.read_csv(exist_file)
    exist_df_relevant = exist_df[exist_df['version'] == version]

    # A dataframe where every word provided to a given image is marked as rare (=doesn't have a word IA) or not (has word IA)
    rare_df = pd.read_csv(rare_file)
    rare_df['image'] = rare_df['image'].apply(lambda x: x.split('/')[-1].split('.')[0])
    # we are only interested in this df to check rare words, therefore we'll only leave those which are tagged as such
    rare_df_onlynorare = rare_df[rare_df['word is rare'] == 0]  # these words have IA scores
    rare_df_onlyrare = rare_df[rare_df['word is rare'] == 1]  # these words do not have IA scores

    # A dataframe where we have all words provided by each participant to each image
    lemmatized_df = pd.read_csv(lemmatized_file)
    lemmatized_df = lemmatized_df.loc[:, ~lemmatized_df.columns.str.contains('^Unnamed')]  # delete Unnamed columns
    lemmatized_df['num_of_IA'] = lemmatized_df.apply(lambda row: count_ia(row, rare_df_onlynorare), axis=1)
    lemmatized_df['num_of_nonexist'] = lemmatized_df.apply(lambda row: count_nonexist(row, exist_df_relevant, rare_df_onlyrare), axis=1)
    lemmatized_df['nonexist_ratio'] = lemmatized_df['num_of_nonexist'] / lemmatized_df['num_of_IA']
    lemmatized_df.to_csv(os.path.join(save_path, "experiment_df_words_lemmatized_nonExistRatio.csv"), index=False)
    return




"""
Reviewer 1: words with IA but w/o any connection to the image
"""
COL_EXIST = "existInImage"
COL_NOEXIST_GISTONLY = "nonexist: right gist non-existent item"
COL_NOEXIST_CONFUSED = "nonexist:  confusion"
COL_NOEXIST_UNRELATED = "nonexist: no connection at all"
COL_NOEXIST_CONCEPTUAL = "conceptual word"

EXP_VERSION = "version"
IMAGE_NAME_COL = "image"


def existence_analysis(exist_file, save_path):
    # A dataframe where every word with a word-IA in the image is marked as existing or not
    exist_df = pd.read_csv(exist_file)

    # how many words with IA are there in for each image (in each version)
    ia_counts = exist_df.groupby([EXP_VERSION, IMAGE_NAME_COL]).count()["word"].reset_index(drop=False)
    ia_counts.rename(columns={"word": "words with IA"}, inplace=True)

    # how many words with IA are there in each existence type (1's in one of the crucial columns
    """
    "count" counts both "1"s and "0"s (and ignores NAs). 
    We want to sum ONLY the "1"s, those that were tagged as such (so we use "sum")
    """
    # exist
    exist_counts = exist_df.groupby([EXP_VERSION, IMAGE_NAME_COL]).sum()[COL_EXIST].reset_index(drop=False)
    # right gist, but not in image
    gist_counts = exist_df.groupby([EXP_VERSION, IMAGE_NAME_COL]).sum()[COL_NOEXIST_GISTONLY].reset_index(drop=False)
    # wrong gist (confusion)
    confusion_counts = exist_df.groupby([EXP_VERSION, IMAGE_NAME_COL]).sum()[COL_NOEXIST_CONFUSED].reset_index(drop=False)
    # completely unrelated
    unrelated_counts = exist_df.groupby([EXP_VERSION, IMAGE_NAME_COL]).sum()[COL_NOEXIST_UNRELATED].reset_index(drop=False)
    # conceptual
    conceptual_counts = exist_df.groupby([EXP_VERSION, IMAGE_NAME_COL]).sum()[COL_NOEXIST_CONCEPTUAL].reset_index(drop=False)

    # merge them
    df_list = [ia_counts, exist_counts, gist_counts, confusion_counts, unrelated_counts, conceptual_counts]
    word_stats_df = ft.reduce(lambda left, right: pd.merge(left, right, on=[EXP_VERSION, IMAGE_NAME_COL]), df_list)
    # convert floats to ints
    cols = [COL_EXIST, COL_NOEXIST_GISTONLY, COL_NOEXIST_CONFUSED, COL_NOEXIST_UNRELATED, COL_NOEXIST_CONCEPTUAL]
    word_stats_df[cols] = word_stats_df[cols].astype(int)

    """
    sanity check: the 'zeros' column should be all zeros as both 'sum' and 'ia' columns should contain all the IA words
    of this image (no more no less). ---> "df" should be empty
    """
    word_stats_df['sum'] = word_stats_df[cols].sum(axis=1)
    word_stats_df['zeros'] = word_stats_df['sum'] - word_stats_df['words with IA']
    df = word_stats_df[word_stats_df['zeros'] != 0]

    # after we are confident that we have all words tagged (and no double tagging), we can calculate %s
    word_stats_df.to_csv(os.path.join(save_path, "word_counts_types_raw.csv"), index=False)
    # save converted
    word_stats_df[cols] = word_stats_df[cols].div(word_stats_df['words with IA'], axis=0) * 100
    word_stats_df.to_csv(os.path.join(save_path, "word_counts_pctgs_types.csv"), index=False)

    # now do stats for %s
    word_stats_df_noimage = word_stats_df.drop(columns=[IMAGE_NAME_COL], inplace=False)
    mean_df = word_stats_df_noimage.groupby([EXP_VERSION]).mean()
    mean_df.to_csv(os.path.join(save_path, "word_counts_pctgs_mean.csv"), index=True)
    std_df = word_stats_df_noimage.groupby([EXP_VERSION]).std()
    std_df.to_csv(os.path.join(save_path, "word_counts_pctgs_sd.csv"), index=True)

    """
    Now, break it down: for each type, what's the average word IA score. As our results were about word IAs, 
    no need to collapse per image like we did above to get these stats. 
    """
    results = []
    raw_dfs = []
    for version in exist_df.version.unique().tolist():
        version_df = exist_df[exist_df[EXP_VERSION] == version]
        result = {}
        for col in cols:
            col_IA = version_df[version_df[col] == 1]  # only for this existence category, all words
            col_IA_mean = col_IA['word IA'].mean()
            col_IA_sd = col_IA['word IA'].std()
            result[col] = {f'Mean IA {version}': col_IA_mean, f'Std IA {version}': col_IA_sd}
            # for aggregating later
            col_IA.reset_index(inplace=True, drop=True)
            col_IA['status'] = col
            relevant_cols = ['word', 'word IA', 'version', 'image', 'status']
            col_IA = col_IA.loc[:, relevant_cols]
            raw_dfs.append(col_IA)
        result_df = pd.DataFrame(result)
        results.append(result_df)
    # actual data
    raw_df = pd.concat(raw_dfs)
    raw_df.to_csv(os.path.join(save_path, f"word_IA_per_existence_data.csv"), index=False)
    # stats
    results_df = pd.concat(results)
    results_df.to_csv(os.path.join(save_path, f"word_IA_per_existence_stats.csv"), index=True)
    return





"""
Slider (control exp. B)
"""

PATH_INTACT = "resources/stimuli/exp/orig/"
PATH_BLURRED = "resources/stimuli/exp/blur10/"
COL_STIM = "exp_stim"

def get_image(row):
    image = row[COL_STIM].replace(PATH_INTACT, "").replace(PATH_BLURRED, "").replace(".jpg", "")
    return image

def get_version(row):
    if PATH_INTACT in row[COL_STIM]:
        return "intact"
    return "blurred"

def slider_analysis(sub_df_file, save_path):
    data = pd.read_csv(sub_df_file)
    data['image'] = data.apply(lambda row: get_image(row), axis=1)
    data['version'] = data.apply(lambda row: get_version(row), axis=1)
    relevant_data = data[['participant', 'image', 'version', 'trial_slider.response', 'trial_slider.rt']]
    relevant_data.to_csv(os.path.join(save_path, "subject_df_perImage.csv"), index=False)
    relevant_data.drop(columns=['participant'], inplace=True)
    relevant_data[['trial_slider.response', 'trial_slider.rt']] = relevant_data[['trial_slider.response', 'trial_slider.rt']].astype(float)
    grouped = relevant_data.groupby(['image', 'version']).mean().reset_index(drop=False)
    grouped.to_csv(os.path.join(save_path, "subject_df_perImage_grouped.csv"), index=False)
    pivot = grouped.pivot(index='image', columns='version', values='trial_slider.response').reset_index(drop=False)
    pivot.to_csv(os.path.join(save_path, "slider_per_image_version.csv"), index=False)
    return


if __name__ == "__main__":
    slider_analysis(
        sub_df_file=r"..\subject_df.csv",
        save_path=r"..\processed")
