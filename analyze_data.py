import os
import collections
import re
import pandas as pd
import numpy as np
import statistics
import pickle
from collections import Counter
from spellchecker import SpellChecker
from nltk.corpus import stopwords  # NOTE: this requires a one-time download of stopwords using NLTK data installer
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import num2words
import process_raw_data
from wordfreq import word_frequency, zipf_frequency

"""
Manage all the analyses of the gist experimental data. 
"""

__author__ = "Rony Hirschhorn"


STIM_DUR_FROM_FRAMES = "stim_dur_from_frames"
SAMPLE_IMAGE = "im0000041.jpg"


def too_many_responses(stim, sub_dict, stim_dict):
    """

    :param stim: the stimulus which has more than STIM_REPS responses
    :param sub_dict: A dictionary where key = subject Prolific ID, value=a dict where key=Pavlovia session number,
    value=the session's data (preprocessed)
    :param stim_dict:  A dictionary where key=stimulus image name, value=a dicitionary where key=subject's session
    DEMOGRAPHIC ORDER - NOT PAVLOVIA ID! and the value is a dataframe where each line contains information about a
    single trial (and one of them is the said stimulus)
    :return:
    sub_dict: After exceeding entries were REMOVED from subjects' DATAFRAMES (such that now the exceeding subjects
    have -1 stimuli in their data)
    stim_dict: After exceeding entries were REMOVED from stim list of dataframes.
    """
    sub_order = sorted(list(stim_dict[stim].keys()))
    subs_order_to_exclude = sub_order[process_raw_data.STIM_REPS:]
    for sub_sess_order in subs_order_to_exclude:
        sub_df = stim_dict[stim].pop(sub_sess_order)  # remove their response df from the image's response dict
        sub_id = sub_df[process_raw_data.PROLIFIC_ID].unique()[0]
        sub_sess_number = sub_df[process_raw_data.SEQ_NUM].unique()[0]
        df = sub_dict[sub_id][sub_sess_number]
        df = df[df[process_raw_data.STIM_ID] != stim]  # remove their response line for this stimulus from the subject dataframe
        sub_dict[sub_id][sub_sess_number] = df
    return sub_dict, stim_dict


def too_few_responses(stim, sub_dict, stim_dict):
    sub_order = sorted(list(stim_dict[stim].keys()))
    if len(sub_order) >= process_raw_data.STIM_REPS:
        print(f"ERROR: stim {stim} does not have too few responses")
        return
    for sub_sess_order in sub_order:
        sub_id = stim_dict[stim][sub_sess_order][process_raw_data.PROLIFIC_ID].unique()[0]
        sub_sess_number = stim_dict[stim][sub_sess_order][process_raw_data.SEQ_NUM].unique()[0]
        df = sub_dict[sub_id][sub_sess_number]
        df = df[df[process_raw_data.STIM_ID] != stim]  # remove their response line for this stimulus from the subject dataframe
        sub_dict[sub_id][sub_sess_number] = df
    stim_dict.pop(stim)  # remove this stimulus from the stimulus db altogether
    return sub_dict, stim_dict


def filter_stimuli(sub_dict, stim_dict):
    stim_list = list(stim_dict.keys())
    too_many = 0
    too_few = 0

    # for sanity check
    stim_subs_before = dict()
    subs_list = list(sub_dict.keys())
    for sub in subs_list:
        sessions = list(sub_dict[sub].keys())
        for sess in sessions:
            sess_stim = sub_dict[sub][sess].loc[:, process_raw_data.STIM_ID_EXTRA_COL].tolist()
            for stim in sess_stim:
                if stim not in list(stim_subs_before.keys()):
                    stim_subs_before[stim] = list()
                stim_subs_before[stim].append(sub)

    # actual work
    for stim in stim_list:
        resps = stim_dict[stim]
        if len(resps) > process_raw_data.STIM_REPS:  # stimulus was seen by more than STIM_REPS subjects; as per Chuyin- remove LIFO
            sub_dict, stim_dict = too_many_responses(stim, sub_dict, stim_dict)
            too_many += 1
        elif len(resps) < process_raw_data.STIM_REPS:  # stimulus was seen by less than STIM_REPS subjects: remove from analysis
            sub_dict, stim_dict = too_few_responses(stim, sub_dict, stim_dict)
            too_few += 1
        # else, stimulus was seen exactly STIM_REPS times, all is good
    print(f"{too_many} stimuli were seen by more than {process_raw_data.STIM_REPS} subjects; exceeding responses were removed LIFO")
    print(f"{too_few} stimuli were seen by less than {process_raw_data.STIM_REPS} subjects; stimuli were removed from all {too_few} participants")
    print(f"NOW ***{len(list(stim_dict.keys()))}*** stimuli are left to be analyzed!")

    # sanity check 1
    for stim in stim_dict.keys():
        resps = stim_dict[stim]
        if len(resps) > process_raw_data.STIM_REPS or len(resps) < process_raw_data.STIM_REPS:
            raise Exception("ERROR: stimuli were not filtered!!!!!! (a)")

    # sanity check 2
    stim_subs_after = dict()
    subs_list = list(sub_dict.keys())
    for sub in subs_list:
        sessions = list(sub_dict[sub].keys())
        for sess in sessions:
            sess_stim = sub_dict[sub][sess].loc[:, process_raw_data.STIM_ID_EXTRA_COL].tolist()
            for stim in sess_stim:
                if stim not in list(stim_subs_after.keys()):
                    stim_subs_after[stim] = list()
                stim_subs_after[stim].append(sub)
    for stim in list(stim_subs_after.keys()):
        if len(stim_subs_after[stim]) > process_raw_data.STIM_REPS:
            raise Exception("ERROR: stimuli were not filtered!!!!!! (b)")

    """
    Following Chuyin's method above, we deleted a stimulus from a subject's' run if the stimulus was already seen
    by process_raw_data.STIM_REPS other subjects BEFORE this subject has seen the stimulus. 
    This means that a subject could be in face "stripped" from all their stimuli because by the time they 
    saw them,  STIM_REPS other people have already seen them.
    As we follow Chuyin, this is what we did above. The code below checks how many subjects we lost due to this manipulation.
    """
    nullified_subs = list()
    subs_list = list(sub_dict.keys())

    for sub in subs_list:
        sessions = list(sub_dict[sub].keys())
        null_sess_list = list()
        for sess in sessions:
            if sub_dict[sub][sess].empty:  # if the subject's experiment dataframe is now empty
                null_sess_list.append(sess)
        if null_sess_list == sessions:
            poor_sub = sub_dict.pop(sub)
            nullified_subs.append(poor_sub)
    print(f"{len(nullified_subs)} subjects were removed following Chuyin's LIFO stimulus removal maintaining {process_raw_data.STIM_REPS} subjects per image.")
    print(f"**{len(sub_dict.keys())}** subjects will be analyzed.")

    return sub_dict, stim_dict


def unify_data(sub_dict):
    experiment_df = pd.concat([sub_dict[sub][sess] for sub in sub_dict.keys() for sess in sub_dict[sub].keys()])
    # stimulus duration by multiplying the duration (ms) of a single frame by the number of frames the stimulus lasted
    experiment_df[STIM_DUR_FROM_FRAMES] = experiment_df[process_raw_data.STIM_DUR_FRAMES] * experiment_df[process_raw_data.RR_FRAMEDUR]
    experiment_df.reset_index(inplace=True)
    print(f"{len(experiment_df[process_raw_data.PROLIFIC_ID].unique())} subjects in the analyzed dataset")
    return experiment_df


def replace_digits(word):
    """
    Replace digits with verbal descriptions of their numbers. e.g., replace "2" with "two" and "42" with "fourty-two"
    :param word: a given string
    :return: the same string, with the digits in it converted.
    """
    if not isinstance(word, str):  # word is nan, not a string
        return word

    if bool(re.search(r'\d', word)):  # if the word contains digits (number)
        word_split = word.split("-")
        word_split_result = list()
        for token in word_split:
            if token.isdigit():
                token = num2words.num2words(int(token))
            word_split_result.append(token)
        word_unified = '-'.join(word_split_result)
        return word_unified
    else:  # if the word does not contain digits
        return word


def remove_stop_words(word):
    """
    Nullify stop-words (replace them with nothing
    :param word: the given string
    :return: the same string w/o English stop words
    """
    if not isinstance(word, str):  # word is nan, not a string
        return word

    stop_words = set(stopwords.words('english'))
    word_split = word.split("-")
    word_split_result = list()
    for token in word_split:
        if token not in stop_words:
            word_split_result.append(token)
    word_unified = '-'.join(word_split_result)
    return word_unified


def process_responses(experiment_df, save_path, load=False):
    """
    This method processes the verbal responses given by subjects. It removes quotation marks and replaces spaces
    with hyphens (as per Chuyin et al. 2022).
    In addition, it replaces digits with words (e.g., 2 -> "two"), UNLIKE the original work by Chuyin.
    Then, it nullifies stop words (because that's what was originally done).
    Finally, it replaces nans with empty strings so all responses are of type str.
    :param experiment_df: The dataframe including all the experiment's responses
    :param save_path: the path to which data should be saved
    :param load: whether to load a pre-processed file or generate a new one
    :return: the resulting dataframe after pre-processing of all words.
    """
    print("parse words")
    file_name = "experiment_df_words_parsed.csv"
    if load == True:
        experiment_df = pd.read_csv(os.path.join(save_path, file_name))
    else:  # do ONCE
        for col in process_raw_data.WORDS:
            experiment_df[col] = experiment_df[col].str.lower()  # lowercase
            experiment_df[col] = experiment_df[col].str.replace('"', '')  # get rid of "these"
            experiment_df[col] = experiment_df[col].str.replace("'", '')  # get rid of 'these'
            experiment_df[col] = experiment_df[col].str.replace("`", '')  # get rid of `these`
            experiment_df[col] = experiment_df[col].str.replace('|', '')  # get rid of |
            experiment_df[col] = experiment_df[col].str.replace('_', '-')  # get rid of _
            experiment_df[col] = experiment_df[col].str.replace(' - ', '-')  # get rid of spaces around hyphens
            experiment_df[col] = experiment_df[col].str.replace('- ', '-')  # get rid of spaces around hyphens
            experiment_df[col] = experiment_df[col].str.replace(' -', '-')  # get rid of spaces around hyphens
            experiment_df[col] = experiment_df[col].str.replace(' ', '-')  # replace space between two words with a hyphen
            experiment_df[col] = experiment_df[col].str.rstrip('-')  # remove leading and trailing hyphens
            experiment_df[col] = experiment_df[col].str.lstrip('-')  # remove leading and trailing hyphens
            experiment_df[col] = experiment_df[col].str.replace("'", "")  # replace 'these'
            experiment_df[col] = experiment_df[col].apply(lambda w: replace_digits(w))  # replace digits with words
            # Chuyin's original work: they removed stop words!!!
            experiment_df[col] = experiment_df[col].apply(lambda w: remove_stop_words(w))
        experiment_df = experiment_df.replace('', np.nan)  # replace '' with nan
        experiment_df.replace(np.nan, '', inplace=True)
        experiment_df.to_csv(os.path.join(save_path, file_name), index=False)
    return experiment_df


def responses_spelling(experiment_df, save_path, load=False, conversion_file=False):
    print("correct spelling")
    file_name = "experiment_df_words_corrected.csv"
    conversion_file_name = "words_conversion_log.csv"
    if load == True:
        experiment_df = pd.read_csv(os.path.join(save_path, file_name))
    else:  # do ONCE
        experiment_df.replace(np.nan, '', inplace=True)
        if conversion_file == True:
            conversion = pd.read_csv(os.path.join(save_path, conversion_file_name))
        else:  # DO ONCE, THEN CHANGE conversion_file to TRUE and upload
            spell = SpellChecker(language='en')
            orig = list()  # for a df documenting the conversion
            corrected = list()  # for a df documenting the conversion
            for col in process_raw_data.WORDS:
                word_list = experiment_df[col].tolist()  # all words in column
                misspelled = spell.unknown(word_list)  # all the words the SpellChecker recognized as "misspelled"
                for word in misspelled:
                    if word == '' or len(word) == 1:
                        continue  # Either an empty word or a word describing a single letter in Sperling array
                    most_likely = spell.correction(word)  # Get the one `most likely` answer
                    if word != most_likely:  # if suggestion is different from the original word
                        orig.append(word)
                        corrected.append(most_likely)
            conversion = pd.DataFrame({"orig": orig, "SpellCheck": corrected}).to_csv(os.path.join(save_path, conversion_file_name), index=False)
            return  # manually go over and decide whether to (1) stay with the original (2) take the spellchecked version (3) alternative manual solution : add a "corrected" column!!
        for ind, row in conversion.iterrows():
            experiment_df = experiment_df.replace(row["orig"], row["corrected"])
        experiment_df.to_csv(os.path.join(save_path, file_name), index=False)
    return experiment_df


def lemmatize(experiment_df, save_path, load=False):
    file_name = "experiment_df_words_lemmatized.csv"
    lemm_file_name = "words_lemmatization_log.csv"
    experiment_df.replace(np.nan, '', inplace=True)
    print("lemmatize words")
    if load == True:
        conversion = pd.read_csv(os.path.join(save_path, lemm_file_name))
    else:
        nlp = spacy.load("en_core_web_trf")  # prefer accuracy over performance: https://spacy.io/usage/models
        orig = list()  # list all response words
        lemmatized = list()
        pos = list()
        for col in process_raw_data.WORDS:
            words = experiment_df[col].tolist()
            for word in words:
                if len(word) > 1:  # not a single letter
                    nlp_token = nlp(word)
                    lemm = [token.lemma_ for token in nlp_token][0]
                    pos = [token.pos_ for token in nlp_token][0]
                    # if lemm != word:  # commented as the lemmatization might miss spelling differences between 2 versions of the same word
                    orig.append(word)
                    lemmatized.append(lemm)
        conversion = pd.DataFrame({"orig": orig, "pos": pos, "lemmatized": lemmatized})
        conversion.to_csv(os.path.join(save_path, lemm_file_name), index=False)
        return
    conversion.replace(np.nan, '', inplace=True)
    for ind, row in conversion.iterrows():
        experiment_df = experiment_df.replace(row["orig"], row["approved"])
    experiment_df.to_csv(os.path.join(save_path, file_name), index=False)
    return experiment_df


def count_unique_words(experiment_df):
    word_list = [experiment_df[col].tolist() for col in process_raw_data.WORDS]
    word_list = [word for col in word_list for word in col if len(word) > 0]  # flatten, no nans
    word_set = set(word_list)  # unique values
    print(f"{len(word_list)} word responses in this dataset, out of which {len(word_set)} are unique words.")
    return


def assert_no_duplicate_resps_within_subject(experiment_df):
    print("make sure there are no duplicate words within the same subject response for a single picture")
    report_count = 0
    removal_count = 0
    for ind, row in experiment_df.iterrows():
        words = row[process_raw_data.WORDS].tolist()
        words = [w for w in words if len(w) > 0]  # no-responses are not duplicate responses
        if len(set(words)) < len(words):  # there is a repeating word within a single subject's single trial
            repeated = [(item, count) for item, count in collections.Counter(words).items() if count > 1]
            report_count += len(repeated)
            for rep in repeated:
                rep_word = rep[0]
                rep_cnt = rep[1]
                for col in process_raw_data.WORDS:
                    if rep_cnt == 1:
                        break  # no need to look for more appearances of this word
                    elif row[col] == rep_word and rep_cnt > 1:
                        experiment_df.loc[ind, col] = ''  # empty this cell
                        rep_cnt -= 1
                        removal_count += 1
    print(f"{report_count} words were used more than once in the same trial (describing the same image); removed {removal_count} repetitions")
    return experiment_df


def table_per_image(experiment_df, include_extra_col=None):
    result_dict = dict()
    image_list = list(experiment_df[process_raw_data.STIM_ID].unique())
    relevant_cols = process_raw_data.WORDS.copy()
    if include_extra_col is not None:
        relevant_cols.extend(include_extra_col)
    for image in image_list:
        image_df = experiment_df[experiment_df[process_raw_data.STIM_ID] == image][relevant_cols]  # each row = one subject response to this image
        image_df.replace('', np.nan, inplace=True)
        result_dict[image] = image_df.reset_index(drop=True, inplace=False)
    return result_dict


def image_stats(image_dict, save_path):
    """
    Save summary statistics of responses per image. NOTE that this is important for later analysis and collapsing
    across experiments.
    """
    print("Count empty and partial responses per image")
    stats_file_name = "words_per_image_stats.csv"
    raw_file_name = "words_per_image_raw.csv"
    empty_responses_per_image = list()
    partial_responses_per_image = list()
    words_per_image = list()
    image_list = list(image_dict.keys())
    for image in image_list:
        image_df = image_dict[image]
        # how many empty responses are there
        dropped_noresp = image_df.dropna(how='all', subset=process_raw_data.WORDS, inplace=False)  # no responses at all
        noresp_cnt = image_df.shape[0] - dropped_noresp.shape[0]
        empty_responses_per_image.append(noresp_cnt)
        # how many partial responses (not counting the no-responses here), meaning a line with a nan is a partial line
        dropped_partial_resp = dropped_noresp.dropna(how='any', subset=process_raw_data.WORDS, inplace=False)
        partial_cnt = dropped_noresp.shape[0] - dropped_partial_resp.shape[0]
        partial_responses_per_image.append(partial_cnt)
        # how many words in total were provided for this image
        word_cnt = dropped_noresp.shape[0] * len(process_raw_data.WORDS) - dropped_noresp.isna().sum().sum()
        words_per_image.append(word_cnt)
    summary_df = pd.DataFrame({process_raw_data.STIM_ID_EXTRA_COL: image_list,
                               "empty responses per stim": empty_responses_per_image,
                               "partial responses per stim": partial_responses_per_image,
                               "total response words per stim": words_per_image})
    summary_df.to_csv(os.path.join(save_path, raw_file_name))
    summary_df.describe().to_csv(os.path.join(save_path, stats_file_name))
    print(f"A total of {sum(empty_responses_per_image)} trials had no response; {sum([1 for x in empty_responses_per_image if x > 0])} stimuli had at least one no-response in them")
    print(f"A total of {sum(partial_responses_per_image)} trials had partial response (less than 5 words); {sum([1 for x in partial_responses_per_image if x > 0])} stimuli had at least one partial response in them")
    print(f"A total of {sum(words_per_image)} words were provided")  # sanity check
    return summary_df


def count_word_resps(image_df, word, ind):
    """
    We need to count how many out of STIM_REPS - 1 respondants (to equate within-image to betweeen-image) responded
    with the target word. For that, we will first see if we have STIM_REPS - 1 respondents at all, and then we will
    count accordingly.
    """
    relevant = image_df.dropna(how='all', subset=process_raw_data.WORDS, inplace=False)  # drop null-responses
    if relevant.shape[0] > process_raw_data.STIM_REPS - 1:
        # we have process_raw_data.STIM_REPS responses, and we need to remove one, as per Chuyin, remove it by index
        relevant = relevant.drop(relevant.index[ind])
    # otherwise, we don't exceed the number of responses we want to compare to, we're good
    word_reps = relevant.eq(word).sum().sum()  # how many times did word repeat for this image
    return word_reps


def calc_cumul_pcntg(counter_dict):
    """
    For each cell (dict value=count), count the sum of the cell value and all the values TO ITS RIGHT (=larger!! keys)
    :param counter_dict: the dictionary where for each cell there is some value count
    :return: the same dict but the counts are now cumulative
    """
    sum_dict_vals = sum(counter_dict.values())  # sum all values
    pcnt_dict = {p: counter_dict[p]/sum_dict_vals for p in counter_dict}  # same dictionary, converted to %
    cumul_dict = {p: 0 for p in counter_dict}  # init
    pctgs = sorted(list(counter_dict.keys()))  # ASCENDING order
    for ind in range(len(pctgs)):
        cumul_for_ind = 0
        for jnd in range(ind, len(pctgs)):  # from ind until the end (the right)
            cumul_for_ind += pcnt_dict[pctgs[jnd]]
        cumul_dict[pctgs[ind]] = cumul_for_ind
    return cumul_dict


# DEPRECATED! SEE calculate_image_aucs
def corrected_AUC_calculation(fpr, tpr):
    """
    This method calculates the X and Y axes of a step-like function, for ROC calculation purposes.
    The fpr(x) and tpr(y) in this calculation will always create a step function, such that from a
    certain point onward (some fpr value), the Y axis (tpr) changes from 0 to 1 (as tpr in this
    calculation is always an array of 0's up to a point from which it's all 1).
    Thus, the ROC is a step function, and the AUC should be calculated accordingly
    (e.g., word_auc = np.trapz(y=tpr, x=fpr)).

    However, methods that integrate y(x) (such as numpy's trapz) strech a line between two points,
    such that instead of "step" , we will receive a false function (with a diagonal line between the
    last point where y=0 and the first point where y=1).
    Thus, to solve this, and be able to integrate y(x) of our tpr(fpr) step function correctly, one needs
    to generate "extra" points based on the data to make an explicit step for trapz to integrate over
    (without "completing" the function with a diagonal line between two points).
    This is the corrected_AUC_calculation.

    :param fpr: x axis
    :param tpr: y axis
    :return: the corrected X and Y axes
    """
    x_axis = list()
    y_axis = list()
    for i in range(len(fpr) - 1):  # go over the axis (arbitrary, the x and y arrays are the same length)
        # insert the original x, y values that the fpr(x), tpr(y) lists had:
        x_axis.append(fpr[i])
        y_axis.append(tpr[i])
        # now, create a new point, based on the NEXT fpr(x) value with the same tpr(y) value (0 or 1)
        x_val = fpr[i + 1]
        y_val = tpr[i]
        x_axis.append(x_val)
        y_axis.append(y_val)
    # insert the last original element from those lists
    x_axis.append(fpr[-1])
    y_axis.append(tpr[-1])
    return x_axis, y_axis


def calculate_image_aucs(experiment_df, save_path, load=False):
    """
    Word IA is specific to image+word combination.
    """
    print("Calculating word+image AUCs")
    file_name = "experiment_image_aucs.pickle"
    rare_file_name = "words_rare_IA.pickle"
    word_count_file_name = "words_count_per_image"
    if load:
        # auc dict
        fl = open(os.path.join(save_path, file_name), 'rb')
        images_aucs_dict = pickle.load(fl)
        fl.close()
        # rare words dict
        fl = open(os.path.join(save_path, rare_file_name), 'rb')
        rare_word_count = pickle.load(fl)
        fl.close()
        # word count dict
        fl = open(os.path.join(save_path, word_count_file_name + ".pickle"), 'rb')
        word_count_dict = pickle.load(fl)
        fl.close()
        # words w/o AUC list
        image_only_rare_words_df = pd.read_csv(os.path.join(save_path, "image_no_auc_only_rare.csv"))
    else:
        print("Please wait, this will take a while")
        image_dict = table_per_image(experiment_df)
        image_stats(image_dict, save_path)  # counts how many empty/partial responses per image
        image_list = list(image_dict.keys())
        rare_word_count = dict()  # for each image, how many rare words it has
        images_aucs_dict = dict()  # the result dict
        word_count_dict = dict()  # all words, not only those with IA, how many times they appear in image
        image_only_rare_words = list()  # Images w/o AUCs as they only have rare words in them

        # prepare word-IA plot
        plt.clf()
        plt.figure()
        sns.set_style('whitegrid')

        for image in image_list:
            word_count_dict[image] = dict()  # for each IMAGE, key=a word that appeared in it, value=how many times
            rare_word_count[image] = 0
            image_df = image_dict[image]  # all responses for current image
            image_word_aucs = dict()  # key=word, val=[all aucs for this word]

            """
            For each image+word, count how many [other] subjects who responded to this image used this word.
            Note: thanks to "assert_no_duplicate_resps_within_subject", we know for sure that a word can repeat only
            in a different row (=different subject), and not within the same row (=same subject).
            Thus, to count how many times a word appeared other than for a subject of interest, we can just count
            how many times it appeared in image_df_responded and deduct 1 (this is "count_other_subs").
            """
            for ind, row in image_df.iterrows():
                if ind == process_raw_data.STIM_REPS:  # if I expect 10 responses, the last row index is 9
                    raise Exception(f"ERROR: TOO MANY RESPONSES FOR IMAGE {image}")
                for col in process_raw_data.WORDS:  # iterate word columns
                    word = row[col]
                    ### INITIALIZE THE CUMULATIVE PERCENTAGES FOR AUC CALCULATION AT THE IMAGE+WORD LEVEL
                    image_cnt_for_cumul_pcntg = {p: 0 for p in [i / (process_raw_data.STIM_REPS - 1) for i in range(process_raw_data.STIM_REPS)]}
                    other_images_cnt_for_cumul_pcntg = {p: 0 for p in [i / (process_raw_data.STIM_REPS - 1) for i in range(process_raw_data.STIM_REPS)]}
                    if word not in word_count_dict[image]:  # this is for the counter of how many times the word appeared in the image
                        word_count_dict[image][word] = 1
                    else:
                        word_count_dict[image][word] += 1
                    if pd.isna(word):  # null response is not a word, skip
                        continue
                    count_other_subs = image_df.eq(word).sum().sum() - 1  # -1 for other subjects, w/o this one

                    """
                    From Chuyin 2022:
                    ' Note that a word must be reported by at least one “other participant” 
                    (i.e., reported by two or more people in total) under the target image to have a valid Word IA value. 
                    The words that were reported by only one person are called “rarely reported words”. '
                    """
                    # If a target word is reported only by 1 subject, it is a rarely reported word, don't calculate IA
                    if count_other_subs == 0:
                        rare_word_count[image] += 1
                        continue

                    # Otherwise: for each other image, count the number of participants who reported this word:
                    # cnt for current image: THIS IS THE *WITHIN* image ratio
                    pcnt_other_subs = count_other_subs / (process_raw_data.STIM_REPS - 1)
                    image_cnt_for_cumul_pcntg[pcnt_other_subs] = 1
                    # cnt for other images: THESE ARE THE BEWTEEN image ratios
                    other_images = [im for im in image_list if im != image]
                    other_image_cnts = list()  # for each image, count how many subjects reported "word" for that image - see "count_word_resps"
                    for other_image in other_images:
                        other_image_df = image_dict[other_image]
                        count_other_image = count_word_resps(other_image_df, word, ind)
                        other_image_cnts.append(count_other_image)
                    other_image_pcntgs = [cnt / (process_raw_data.STIM_REPS - 1) for cnt in other_image_cnts]
                    other_image_pcntgs_cnt = Counter(other_image_pcntgs)
                    for pcnt in other_image_pcntgs_cnt:  # have for each % the number of images that had this % reported
                        other_images_cnt_for_cumul_pcntg[pcnt] = other_image_pcntgs_cnt[pcnt]

                    # Cumulative % for within-image and between-images word appearance
                    image_cumul_pcntg = calc_cumul_pcntg(image_cnt_for_cumul_pcntg)  # within
                    other_images_cumul_pcntg = calc_cumul_pcntg(other_images_cnt_for_cumul_pcntg)  # between

                    # ROC curve: y=TPR, x = FPR, in Chuyin: y=within-image, x=between-image
                    df = pd.DataFrame({"within image": image_cumul_pcntg, "other images": other_images_cumul_pcntg})
                    # plt.plot(df["other images"], df["within image"]) for debugging, shows the ROC
                    fpr, tpr = df["other images"].tolist()[::-1], df["within image"].tolist()[::-1]  # -1 to have the lists in ASCENDING order
                    """
                    The fpr(x) and tpr(y) in this calculation will always create a step function, such that from a 
                    certain point onward (some fpr value), the Y axis (tpr) changes from 0 to 1 (as tpr in this 
                    calculation is always an array of 0's up to a point from which it's all 1). 
                    Thus, the ROC is a step function, and the AUC should be calculated accordingly
                    (e.g., word_auc = np.trapz(y=tpr, x=fpr)).
                    
                    However, methods that integrate y(x) (such as numpy's trapz) strech a line between two points, 
                    such that instead of "step" , we will receive a false function (with a diagonal line between the 
                    last point where y=0 and the first point where y=1). 
                    Thus, to solve this, and be able to integrate y(x) of our tpr(fpr) step function correctly, one needs
                    to generate "extra" points based on the data to make an explicit step for trapz to integrate over 
                    (without "completing" the function with a diagonal line between two points). 
                    This is the "corrected_AUC_calculation". 
                    
                    ***HOWEVER***!!! In their work, Chuyin et al. (2022) DID NOT apply this correction. This can be seen
                    in their Figure 1 and Figure 1 supplementary material (1b), where you can see that a diagonal line
                    streches between two points in the black ROC curve (and not a "step"-like, vertical one). If you 
                    apply the corrected_AUC_calculation, IAs come out completely different. Thus, we decided to follow
                    the original way, done by Chuyin et al., and comment out the correction (corrected_AUC_calculation).
                    """
                    #x_axis, y_axis = corrected_AUC_calculation(fpr, tpr)  # Read the above comment
                    x_axis, y_axis = fpr, tpr
                    word_auc = np.trapz(y=y_axis, x=x_axis)
                    # PLOT IT for debugging (plt.show to see)
                    plt.plot(x_axis, y_axis)

                    # insert this word to a result image-dict
                    if word not in image_word_aucs:
                        image_word_aucs[word] = [word_auc]
                    else:  # append in order to average all AUCs for this image+word later
                        image_word_aucs[word].append(word_auc)
                    images_aucs_dict[image] = image_word_aucs  # insert to grand dict

            """ 
            If we are at this stage (still in a single image) and images_aucs_dict[image] does not exist, 
            this means that ALL WORDS for this image were RARE WORDS and so AUC was never calculated. 
            """
            if image not in images_aucs_dict:
                image_only_rare_words.append(image)

        print(f"{len(image_only_rare_words)} images do not have AUC as they contain only rare words.")

        """
        SAVE pickles to save time
        """
        # auc dict
        fl = open(os.path.join(save_path, file_name), 'wb')
        pickle.dump(images_aucs_dict, fl)
        fl.close()
        # rare words dict
        fl = open(os.path.join(save_path, rare_file_name), 'wb')
        pickle.dump(rare_word_count, fl)
        fl.close()

        # word counts dict
        word_count_df = pd.DataFrame.from_dict({(image, word): [word_count_dict[image][word]]
                                                for image in word_count_dict.keys()
                                                for word in word_count_dict[image].keys()}, orient='index')
        word_count_df.rename(columns={0: "word count in image"}, inplace=True)
        word_count_df.to_csv(os.path.join(save_path, word_count_file_name+".csv"))
        # pickle it as well
        fl = open(os.path.join(save_path, word_count_file_name+".pickle"), 'wb')
        pickle.dump(word_count_dict, fl)
        fl.close()
        # words without AUC list
        image_only_rare_words_df = pd.DataFrame({"Image": image_only_rare_words})
        image_only_rare_words_df.to_csv(os.path.join(save_path, "image_no_auc_only_rare.csv"))

        # plot - not for publication, just to see what the AUCs look like
        plt.plot([0, 1], [0, 1], '--')  # add diagonal line
        plot_title = f"Word-Image ROC curves (Word IA Calculation)"
        plt.title(plot_title)
        plt.xlabel("Other-Image Cumulative %")
        plt.ylabel("Within-Image Cumulative %")
        plt.savefig(os.path.join(save_path, f"roc_word_IA.png"), bbox_inches="tight")
        plt.close()

    return images_aucs_dict, rare_word_count, word_count_dict, image_only_rare_words_df


def calculate_word_IAs(images_aucs_dict, save_path):
    """
    According to Chuyin et al, word IA is simply the AVERAGE of all AUCs for this words' instances IN A CERTAIN IMAGE.
    """
    print("Calculating word IAs")
    file_name = "experiment_IA_word.csv"
    stats_file_name = "experiment_IA_word_stats.csv"
    word_IA_dict = dict()  # result dict
    for image in images_aucs_dict:  # for each image
        image_name = image.split(os.sep)[-1]  # take only the image name (not path)
        word_IA_dict[image_name] = dict()
        for word in images_aucs_dict[image]:  # for each word in that image
            word_IA = statistics.mean(images_aucs_dict[image][word])  # average all AUCs of this word to get word IA
            word_IA_dict[image_name][word] = word_IA
    # create and save word IA dataframe
    df = pd.DataFrame.from_dict({(image, word): [word_IA_dict[image][word]] for image in word_IA_dict.keys()
                            for word in word_IA_dict[image].keys()}, orient='index')
    df.rename(columns={0: "word IA"}, inplace=True)
    df.to_csv(os.path.join(save_path, file_name))
    # calculate stats on word IAs
    df.describe().to_csv(os.path.join(save_path, stats_file_name))
    return word_IA_dict


def calculate_word_freq_stats(experiment_df, save_path, word_IA_dict, load=False):
    """
    Creates a csv file where for each word in each image (=row), the following information is calculated:
    - how many times did the word appear in the image
    - in how many other images did the word appear
    - how many times did the word appear in other images total
    - is the word rare (by Chuyin definitions, i.e., appeared only once in the image of interest)
    - the word's IA score (if not rare)
    - the word's frequency in English (Zipf frequency, see https://pypi.org/project/wordfreq/
    """
    print("Calculating word frequency stats")
    file_name = "word_frequency_count.csv"
    file_name_norare = "word_frequency_count_norare.csv"
    if load:
        df = pd.read_csv(os.path.join(save_path, file_name))
    else:
        image_dict = table_per_image(experiment_df)
        image_list = list(image_dict.keys())
        curr_image = list()
        curr_word = list()
        word_image_cnt = list()
        word_total_freq_cnt = list()
        word_is_rare = list()
        curr_word_IA = list()
        count_in_image = list()
        word_freq_in_english_lang = list()

        for image in image_list:
            image_df = image_dict[image]
            for ind, row in image_df.iterrows():
                for col in process_raw_data.WORDS:  # iterate word columns
                    word = row[col]
                    if pd.isna(word):  # null response is not a word, skip
                        continue
                    curr_image.append(image)
                    curr_word.append(word)
                    count_other_subs = image_df.eq(word).sum().sum() - 1  # -1 for other subjects, w/o this one
                    count_in_image.append(image_df.eq(word).sum().sum()) # total # of appearances of word in current image
                    if count_other_subs == 0:
                        word_is_rare.append(1)
                        curr_word_IA.append(np.nan)  # no calculated IA for this word, we won't find it in the word_IA_dict
                    else:
                        word_is_rare.append(0)
                        if len(image) == len(SAMPLE_IMAGE) and len(next(iter(word_IA_dict))) == len(SAMPLE_IMAGE):  # keys are identical
                            curr_image_word_IA = word_IA_dict[image][word]
                        elif len(image) != len(SAMPLE_IMAGE) and len(next(iter(word_IA_dict))) != len(SAMPLE_IMAGE):
                            curr_image_word_IA = word_IA_dict[image][word]
                        else:  # keys are not identical: "image" includes the full path to the image, word_IA_dict keys are just the image names
                            curr_image_word_IA = word_IA_dict[image[-len(SAMPLE_IMAGE):]][word]
                        curr_word_IA.append(curr_image_word_IA)
                    # cnt other images:
                    word_image_cntr = 0
                    word_total_freq_cntr = 0
                    other_images = [im for im in image_list if im != image]
                    for other_image in other_images:
                        other_image_df = image_dict[other_image]
                        im_freq = other_image_df.eq(word).sum().sum()  # how many times did the word repeat in the df
                        word_total_freq_cntr += im_freq
                        if im_freq > 0:
                            word_image_cntr += 1
                    word_image_cnt.append(word_image_cntr)
                    word_total_freq_cnt.append(word_total_freq_cntr)
                    # https://pypi.org/project/wordfreq/  "Reasonable Zipf values are between 0 and 8"
                    word_freq_in_english_lang.append(zipf_frequency(word=word, lang='en'))  # count the frequency of the word in the English language

        result_dict = {"image": curr_image, "word": curr_word, "word freq in image": count_in_image,
                       "word appeared in # other images count": word_image_cnt,
                       "word total freq count (in all other images)": word_total_freq_cnt,
                       "word is rare": word_is_rare, "word IA": curr_word_IA,
                       "word freq in English language": word_freq_in_english_lang}
        df = pd.DataFrame(result_dict)
        # drop duplicate rows
        df.drop_duplicates(inplace=True)
        df.to_csv(os.path.join(save_path, file_name), index=False)
        """
        For correlation analysis between word IA and other measures, we will save a version of "word_frequency_count"
        that contains ONLY words that have an IA score to begin with.
        """
        df_norare = df[df["word is rare"] != 1]  # rare words do not have an IA score by definition
        df_norare.to_csv(os.path.join(save_path, file_name_norare), index=False)

    return df


def confidence_analysis(experiment_df, word_IA_dict, save_path, load=False):
    """
    Generate a csv where for each word (WITH AN IA score) in each image (row: word-image pair), there is the IA score
    of that word, and the mean confidence rating for that word in the image.
    """
    file_name = "word_IA_confidence.csv"
    file_name_norare = "word_IA_confidence_norare.csv"
    image_list = list()
    word_list = list()
    word_IA_list = list()
    mean_confidence_list = list()
    relevant_col_list = process_raw_data.WORDS + process_raw_data.WORDS_RATINGS
    col_dict = {process_raw_data.WORDS[i]: process_raw_data.WORDS_RATINGS[i] for i in
                range(len(process_raw_data.WORDS))}

    for image in word_IA_dict.keys():
        image_df = experiment_df[experiment_df[process_raw_data.STIM_ID].str.contains(image)][relevant_col_list]
        for word in word_IA_dict[image].keys():  # only words with word IA!!
            tmp_conf_list = list()  # to aggregate all confidence ratings for that word, and average at the end
            image_list.append(image)
            word_list.append(word)
            word_IA_list.append(word_IA_dict[image][word])
            for ind, row in image_df.iterrows():
                for col in col_dict.keys():
                    if row[col] == word:
                        tmp_conf_list.append(row[col_dict[col]])
            mean_conf = statistics.mean(tmp_conf_list)
            mean_confidence_list.append(mean_conf)

    result_df = pd.DataFrame({"image": image_list, "word": word_list, "word IA": word_IA_list,
                              "mean confidence rating": mean_confidence_list})
    result_df.drop_duplicates(inplace=True)
    result_df.to_csv(os.path.join(save_path, file_name), index=False)
    return


def count_no_response_trials(experiment_df):
    """
    Count how many trials had no response (meaning, subjects provided zero out of five words)
    Then count how many stimuli had at least one no-response in them. Then count partial trials.
    :param experiment_df: experiment data frame after all the word parsing, spelling, lemmatization
    and assertions were made.
    """
    word_df = experiment_df[process_raw_data.WORDS]
    word_df_empty_resps = word_df[word_df.apply(lambda x: min(x) == max(x), 1)]  # this is true only for empty rows
    print(f"A total of {word_df_empty_resps.shape[0]} trials had no response.")

    word_df_empty_resps_info = experiment_df.loc[word_df_empty_resps.index]
    num_of_images_with_empty_resps = len(word_df_empty_resps_info[process_raw_data.STIM_ID].unique())
    print(f"A total of {num_of_images_with_empty_resps} images had at least one no-response in them.")

    word_df_partial_resps = word_df[word_df.apply(lambda x: len(min(x)) == 0, 1)]  # this is true for empty/partial rows
    word_df_partial_resps_only = word_df_partial_resps[word_df_partial_resps.apply(lambda x: min(x) != max(x), 1)]  # only partial
    print(f"A total of {word_df_partial_resps_only.shape[0]} trials had partial responses.")

    word_df_partial_resps_info = experiment_df.loc[word_df_partial_resps_only.index]
    num_of_images_with_partial_resps = len(word_df_partial_resps_info[process_raw_data.STIM_ID].unique())
    print(f"A total of {num_of_images_with_partial_resps} images had at least one partial response in them.")
    return


def analyze_data(experiment_df, save_path):
    """
    This is where the heavy-lifting of data processing is done.
    *NOTE* : each step can either be calculated from scratch, or LOADED once the file has been saved.
    Additionally, starred steps (*) are ones where manual intervention is REQUIRED.

    It is done in the following steps:
    1. process_responses: responses are parsed (spaces, hyphens, replacing digits with words etc)

    2. (*) responses_spelling: this method runs a spell checker and saves a file with the suggested spelling correction
    for misspelled words (conversion_file). THIS REQUIRES MANUAL APPROVAL of the correction in the conversion file
    called "words_conversion_log.csv".
    Once going over it manually, adding a "corrected" column where the final spelling for each orig word, then
    change "conversion_file=True" and then the response table will be updated and saved (then you can use load=True).

    3. (*) lemmatize: this method runs a lemmatization process on the spell-checked words ,and saves a file with the
    suggested lemmas for ALL words in the dataset. THIS REQUIRES MANUAL APPROVAL of the correction in the lemma file.
    This is done by adding an "approved" column to the file "words_lemmatization_log.csv". Once it's done, change "load"
    to True.

    Then, the response matrix is ready for calculation of descriptives and IA.
    """

    print("---PREPARE DATA FOR ANALYSIS---")
    experiment_df = process_responses(experiment_df, save_path, load=False)
    experiment_df = responses_spelling(experiment_df, save_path, load=False, conversion_file=False)  # Conversion file True after adding a "corrected" column to the output file!
    experiment_df = lemmatize(experiment_df, save_path, load=False)  # "True" only after adding a "approved" column to the output file!
    experiment_df = assert_no_duplicate_resps_within_subject(experiment_df)  # self-explanatory
    count_unique_words(experiment_df)  # print how many words are overall, and how many of them are unique
    count_no_response_trials(experiment_df)  # count empty and partial responses
    print("---DATABASE IS READY: LET THE ANALYSIS BEGIN!---")
    images_aucs_dict, rare_word_count, word_count_dict, image_only_rare_words_df = calculate_image_aucs(experiment_df, save_path, load=False)  # ORIGINAL AUC CALCULATION
    word_IA_dict = calculate_word_IAs(images_aucs_dict, save_path)
    print("--- WORD FREQ ANALYSIS---")
    calculate_word_freq_stats(experiment_df, save_path, word_IA_dict, load=False)
    print("---CONFIDENCE ANALYSIS---")
    confidence_analysis(experiment_df, word_IA_dict, save_path, load=False)
    return


def manage_data_analysis(sub_dict, stim_dict, save_path):
    p = os.path.join(save_path, "analysis")
    if not os.path.isdir(p):
        try:
            os.mkdir(p)
        except Exception as e:
            raise e
    print("---EXCLUDE STIMULI & RESPONSES FROM ANALYSIS---")
    sub_dict_proc, stim_dict_proc = filter_stimuli(sub_dict, stim_dict)
    print(f"{len(stim_dict_proc.keys())} images in the analyzed dataset")
    print(f"{len(sub_dict_proc.keys())} subjects in the analyzed dataset")
    experiment_df = unify_data(sub_dict_proc)
    experiment_df.to_csv(os.path.join(p, "experiment_df_words_orig.csv"))
    analyze_data(experiment_df, p)
    return
