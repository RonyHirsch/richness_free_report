import os
import collections
import re
import pandas as pd
import numpy as np
import pingouin as pg
import statistics
import pickle
from collections import Counter
from spellchecker import SpellChecker
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import spacy
from nltk.corpus import wordnet
import num2words
import process_raw_data
from wordfreq import word_frequency, zipf_frequency

"""
Manage all the analyses of the gist experimental data. 
"""

__author__ = "Rony Hirschhorn"

STIM_DUR_FROM_FRAMES = "stim_dur_from_frames"
STIM_DUR_FROM_TS = "stim_dur_from_timestamps"
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
        if stim in ['resources/exp/blackwhite/im0000308.png', 'resources/exp/blackwhite/im0000628.png', 'resources/exp/blackwhite/im0000413.png', 'resources/exp/blackwhite/im0000631.png', 'resources/exp/blackwhite/im0000444.png']:
            b = 1
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
        experiment_df.to_csv(os.path.join(save_path, file_name))
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
            conversion = pd.DataFrame({"orig": orig, "SpellCheck": corrected}).to_csv(os.path.join(save_path, conversion_file_name))
            return  # manually go over and decide whether to (1) stay with the original (2) take the spellchecked version (3) alternative manual solution : add a "corrected" column!!
        for ind, row in conversion.iterrows():
            experiment_df = experiment_df.replace(row["orig"], row["corrected"])
        experiment_df.to_csv(os.path.join(save_path, file_name))
    return experiment_df


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


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
                if len(word) > 1:
                    nlp_token = nlp(word)
                    lemm = [token.lemma_ for token in nlp_token][0]
                    pos = [token.pos_ for token in nlp_token][0]
                    # if lemm != word:  # commented as the lemmatization might miss spelling differences between 2 versions of the same word
                    orig.append(word)
                    lemmatized.append(lemm)
        conversion = pd.DataFrame({"orig": orig, "pos": pos, "lemmatized": lemmatized})
        conversion.to_csv(os.path.join(save_path, lemm_file_name))
        return
    conversion.replace(np.nan, '', inplace=True)
    for ind, row in conversion.iterrows():
        experiment_df = experiment_df.replace(row["orig"], row["approved"])
    experiment_df.to_csv(os.path.join(save_path, file_name))
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
    Save summary statistics of responses per image.
    :param image_dict:
    :param save_path:
    :return:
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
    :param image_df:
    :param word:
    :return:
    """
    relevant = image_df.dropna(how='all', subset=process_raw_data.WORDS, inplace=False)  # drop null-responses
    if relevant.shape[0] > process_raw_data.STIM_REPS - 1:
        # we have process_raw_data.STIM_REPS responses, and we need to remove one, as per Chuyin, remove it by index
        relevant = relevant.drop(relevant.index[ind])
    # otherwise, we don't exceed the number of responses we want to compare to, we're good
    word_reps = relevant.eq(word).sum().sum()  # how many times did word repeat for this image
    return word_reps


def calc_cumul_pcntg(counter_dict):
    sum_dict_vals = sum(counter_dict.values())
    pcnt_dict = {p: counter_dict[p]/sum_dict_vals for p in counter_dict}  # same dictionary, converted to %
    cumul_dict = {p: 0 for p in counter_dict}
    pctgs = sorted(list(counter_dict.keys()))
    for ind in range(len(pctgs)):
        cumul_for_ind = 0
        for jnd in range(ind, len(pctgs)):
            cumul_for_ind += pcnt_dict[pctgs[jnd]]
        cumul_dict[pctgs[ind]] = cumul_for_ind
    return cumul_dict


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
    :param experiment_df:
    :return:
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
        image_stats(image_dict, save_path)
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
            word_count_dict[image] = dict()
            rare_word_count[image] = 0
            image_df = image_dict[image]  # all responses for current image
            image_word_aucs = dict()  # key=word, val=[all aucs for this word]

            # For each image+word, count how many [other] subjects who responded to this image used this word.
            # Note: thanks to "assert_no_duplicate_resps_within_subject", we know for sure that a word can repeat only
            # in a different row (=different subject), and not within the same row (=same subject).
            # Thus, to count how many times a word appeared other than for a subject of interest, we can just count
            # how many times it appeared in image_df_responded and deduct 1.
            for ind, row in image_df.iterrows():
                if ind == process_raw_data.STIM_REPS:
                    raise Exception(f"ERROR: TOO MANY RESPONSES FOR IMAGE {image}")
                for col in process_raw_data.WORDS:  # iterate word columns
                    word = row[col]
                    ### INITIALIZE THE CUMULATIVE PERCENTAGES FOR AUC CALCULATION AT THE IMAGE+WORD LEVEL
                    image_cnt_for_cumul_pcntg = {p: 0 for p in [i / (process_raw_data.STIM_REPS - 1) for i in range(process_raw_data.STIM_REPS)]}
                    other_images_cnt_for_cumul_pcntg = {p: 0 for p in [i / (process_raw_data.STIM_REPS - 1) for i in range(process_raw_data.STIM_REPS)]}
                    if word not in word_count_dict[image]:
                        word_count_dict[image][word] = 1
                    else:
                        word_count_dict[image][word] += 1

                    if pd.isna(word):  # null response is not a word, skip
                        continue
                    count_other_subs = image_df.eq(word).sum().sum() - 1  # -1 for other subjects, w/o this one
                    """
                    From Chuyin 2022:
                    "  Note that a word must be reported by at least one “other participant” 
                    (i.e., reported by two or more people in total) under the target image to have a valid Word IA value. 
                    The words that were reported by only one person are called “rarely reported words”. "
                    """
                    # If a target word is reported only by 1 subject, it is a rarely reported word, don't calculate IA
                    if count_other_subs == 0:
                        rare_word_count[image] += 1
                        continue

                    # Otherwise: for each other image, count the number of participants who reported this word:
                    # cnt for current image:
                    pcnt_other_subs = count_other_subs / (process_raw_data.STIM_REPS - 1)
                    image_cnt_for_cumul_pcntg[pcnt_other_subs] = 1
                    # cnt for other images:
                    other_images = [im for im in image_list if im != image]
                    other_image_cnts = list()  # for each image, count how many subjects reported "word" for that image
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
                    fpr, tpr = df["other images"].tolist()[::-1], df["within image"].tolist()[::-1]  # -1 to have the lists in ascending order
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
                    # PLOT IT
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
            In this case, we will set images_aucs_dict[image] to be nan. 
            """
            if image not in images_aucs_dict:
                image_only_rare_words.append(image)
                #images_aucs_dict[image] = np.nan

        print(f"{len(image_only_rare_words)} images do not have AUC as they contain only rare words.")

        # auc dict
        fl = open(os.path.join(save_path, file_name), 'ab')
        pickle.dump(images_aucs_dict, fl)
        fl.close()
        # rare words dict
        fl = open(os.path.join(save_path, rare_file_name), 'ab')
        pickle.dump(rare_word_count, fl)
        fl.close()
        # word counts dict
        word_count_df = pd.DataFrame.from_dict({(image, word): [word_count_dict[image][word]]
                                                for image in word_count_dict.keys()
                                                for word in word_count_dict[image].keys()}, orient='index')
        word_count_df.rename(columns={0: "word count in image"}, inplace=True)
        word_count_df.to_csv(os.path.join(save_path, word_count_file_name+".csv"))
        fl = open(os.path.join(save_path, word_count_file_name+".pickle"), 'ab')
        pickle.dump(word_count_dict, fl)
        fl.close()
        # words without AUC list
        image_only_rare_words_df = pd.DataFrame({"Image": image_only_rare_words})
        image_only_rare_words_df.to_csv(os.path.join(save_path, "image_no_auc_only_rare.csv"))

        # plot
        plt.plot([0, 1], [0, 1], '--')  # add diagonal line
        plot_title = f"Word-Image ROC curves (Word IA Calculation)"
        plt.title(plot_title)
        plt.xlabel("Other-Image Cumulative %")
        plt.ylabel("Within-Image Cumulative %")
        plt.savefig(os.path.join(save_path, f"roc_word_IA.png"), bbox_inches="tight")
        plt.close()

    return images_aucs_dict, rare_word_count, word_count_dict, image_only_rare_words_df


def calculate_word_IAs(images_aucs_dict, save_path):
    print("Calculating word IAs")
    file_name = "experiment_IA_word.csv"
    stats_file_name = "experiment_IA_word_stats.csv"
    word_IA_dict = dict()  # result dict
    for image in images_aucs_dict:
        image_name = image.replace("resources/exp/orig/", "")
        word_IA_dict[image_name] = dict()
        for word in images_aucs_dict[image]:
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


def calculate_image_IAs(word_IAs_dict, word_count_dict, save_path):
    print("Calculating image IAs")
    file_name = "experiment_IA_image.csv"
    stats_file_name = "experiment_IA_image_stats.csv"
    image_IA_dict = dict()
    image_word_cnt_dict = dict()  # how many NON-NAN WORDS were provided for this image
    total = len(process_raw_data.WORDS) * process_raw_data.STIM_REPS  # total responses for an image

    if len(next(iter(word_IAs_dict))) == len(SAMPLE_IMAGE) and len(next(iter(word_count_dict))) != len(SAMPLE_IMAGE):
        word_count_dict = {k[-len(SAMPLE_IMAGE):]: word_count_dict[k] for k in word_count_dict.keys()}

    for image in word_IAs_dict:
        image_word_IA_list = list()
        for word in word_IAs_dict[image]:
            image_word_IA_list.append(word_IAs_dict[image][word])  # aggregate all word IAs
        image_IA_dict[image] = statistics.mean(image_word_IA_list)  # average all word IAs for this image
        # now, count how many nans (no responses) in this image
        key_list = [type(k) for k in word_count_dict[image].keys()]
        if float in key_list:  # this means "nan" is one of the "words" for this image
            nan_loc = key_list.index(float)  # which key index it is
            nans = word_count_dict[image][list(word_count_dict[image].keys())[nan_loc]]
            image_word_cnt_dict[image] = total - nans
        else:
            image_word_cnt_dict[image] = sum(word_count_dict[image].values())
    print(f"{len(image_IA_dict.keys())} images have calculated image IA")
    # create and save word IA dataframe
    df = pd.DataFrame.from_dict({k: [image_IA_dict[k]] for k in image_IA_dict}).T
    df.rename(columns={0: "image IA"}, inplace=True)

    nans_df = pd.DataFrame.from_dict({k: [image_word_cnt_dict[k]] for k in image_IA_dict}).T
    nans_df.rename(columns={0: "Number of word responses in image"}, inplace=True)
    result_df = pd.merge(df, nans_df, left_index=True, right_index=True)
    result_df.to_csv(os.path.join(save_path, file_name))
    # calculate stats on word IAs
    df.describe().to_csv(os.path.join(save_path, stats_file_name))
    return image_IA_dict


def image_IA_and_rarely_reported(image_IA_dict, rare_word_count, save_path):
    print("correlation between image IA and the number of rarely reported words in an image")
    if len(next(iter(rare_word_count))) != len(SAMPLE_IMAGE) and len(next(iter(image_IA_dict))) == len(SAMPLE_IMAGE):
        rare_word_count = {k[-len(SAMPLE_IMAGE):]: rare_word_count[k] for k in rare_word_count.keys()}
    result_df = pd.DataFrame({"image IA": image_IA_dict, "rare word count": rare_word_count})
    result_df.to_csv(os.path.join(save_path, "image_IA_rare_word_cnt.csv"))
    corr = pg.corr(x=result_df["image IA"], y=result_df["rare word count"], alternative='two-sided', method='pearson')
    print(corr)
    # plot
    sns.set_style("whitegrid")
    sns.scatterplot(x=result_df["image IA"], y=result_df["rare word count"])
    plt.title("Image IA as a Function of Rare Word Count", fontsize=14)
    plt.xlabel("Image IA")
    plt.ylabel("Number of Rare Words")
    plt.savefig(os.path.join(save_path, f"image_IA_rare_word_cnt.png"))
    return


def calculate_word_freq_stats(experiment_df, save_path, word_IA_dict, load=False):
    print("Calculating word frequency stats")
    file_name = "word_frequency_count.csv"
    stats_file_name = "word_frequency_count_stats.csv"
    stats_rare = "word_frequency_count_stats_rare.csv"
    stats_norare = "word_frequency_count_stats_non_rare.csv"
    if load:
        df = pd.read_csv(os.path.join(save_path, file_name))
    else:
        # for each rare word, in how many different images did it appear, how many times did it repeat total, is it rare
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
        df.to_csv(os.path.join(save_path, file_name))
        # calculate stats on word IAs
        df.describe().to_csv(os.path.join(save_path, stats_file_name))

    # plot a barplot with X-axis being the number of times a word appeared in a specific image and the Y being the word IA
    sns.set_style("whitegrid")
    sns.boxplot(x="word freq in image", y="word IA", data=df, palette="Blues")
    plt.title("Word IA as a Function of Frequency", fontsize=14)
    plt.xlabel("Word IA")
    plt.ylabel("Frequency of Word within Target Image")
    plt.savefig(os.path.join(save_path, f"word_frequency_count.png"))

    # stats: on average, in how many different images did a rare/nonrare word appear?
    rare_df = df[df["word is rare"] == 1]
    rare_df.describe().to_csv(os.path.join(save_path, stats_rare))
    non_rare_df = df[df["word is rare"] == 0]
    non_rare_df.describe().to_csv(os.path.join(save_path, stats_norare))
    return df


def calc_rare_word_per_image(word_count_dict, save_path):
    """
    For each image, how many rare words it has (and stats on that)
    :param word_count_dict:
    :param save_path:
    :return:
    """
    rare_word_count_per_image = "word_rare_count.csv"
    rare_word_count_per_image_stats = "word_rare_count_stats.csv"
    image_list = list()
    image_num_of_rare_words = list()
    for image in word_count_dict.keys():
        image_cntr = 0
        for word in word_count_dict[image].keys():
            if word_count_dict[image][word] == 1:
                image_cntr += 1
        image_list.append(image)
        image_num_of_rare_words.append(image_cntr)

    df = pd.DataFrame({"image": image_list, "rare word count": image_num_of_rare_words})
    df.to_csv(os.path.join(save_path, rare_word_count_per_image))
    df.describe().to_csv(os.path.join(save_path, rare_word_count_per_image_stats))
    res_dict = {image_list[i]: image_num_of_rare_words[i] for i in range(len(image_num_of_rare_words))}
    return res_dict


def analyze_image_presentation_duration(experiment_df, image_IA_dict, save_path):
    print("Calculating presentation duration stats")
    stats_file_name = "image_pres_dur_stats.csv"
    experiment_df_no_rr_outliers = experiment_df[experiment_df[STIM_DUR_FROM_FRAMES] < 150]  # this is in ms, and way too much for a skipped frame/err
    grand_desc = experiment_df_no_rr_outliers.describe()[STIM_DUR_FROM_FRAMES]
    stats_df = pd.DataFrame({STIM_DUR_FROM_FRAMES: grand_desc})
    stats_df.to_csv(os.path.join(save_path, stats_file_name))

    # plot histogram of stim duration
    sns.set_style("whitegrid")
    sns.histplot(data=experiment_df_no_rr_outliers, x=STIM_DUR_FROM_FRAMES, binwidth=1)
    plt.title("Stimulus Presentation Duration in ms", fontsize=14)
    plt.xlabel("Presentation Duration (ms)")
    plt.ylabel("Count (trials)")
    plt.savefig(os.path.join(save_path, f"image_pres_dur_hist.png"))

    exp_tables = table_per_image(experiment_df, include_extra_col=[STIM_DUR_FROM_FRAMES])

    # stimulus average duration and the number of words provided for it
    result_dict = {"image": list(), "number of words": list(), "average presentation duration": list(), "image IA": list()}
    for image in exp_tables:
        if len(image) != len(SAMPLE_IMAGE) and len(next(iter(image_IA_dict))) == len(SAMPLE_IMAGE):
            image_name = image[-len(SAMPLE_IMAGE):]
        else:
            image_name = image
        image_df = exp_tables[image]
        words_provided = image_df[process_raw_data.WORDS].count().sum()  # how many non-nan words are there
        avg_duration = image_df[STIM_DUR_FROM_FRAMES].mean()
        if image_name in image_IA_dict.keys():
            image_IA = image_IA_dict[image_name]
        else:
            image_IA = np.nan
        result_dict["image"].append(image)
        result_dict["number of words"].append(words_provided)
        result_dict["average presentation duration"].append(avg_duration)
        result_dict["image IA"].append(image_IA)
    result_df = pd.DataFrame(result_dict)
    print('correlation between the average presentation duration of an image and the number of reported words')
    correl = pg.corr(x=result_df["number of words"], y=result_df["average presentation duration"], alternative='two-sided', method='pearson')
    print(correl)
    print('correlation between the average presentation duration of an image and the image IA')
    correl = pg.corr(x=result_df["image IA"], y=result_df["average presentation duration"], alternative='two-sided', method='pearson')
    print(correl)
    return


def confidence_analysis(experiment_df, word_IA_dict, save_path, load=False):
    file_name = "word_IA_confidence.csv"

    if load:
        result_df = pd.read_csv(os.path.join(save_path, file_name))
    else:
        image_list = list()
        word_list = list()
        word_IA_list = list()
        mean_confidence_list = list()
        relevant_col_list = process_raw_data.WORDS + process_raw_data.WORDS_RATINGS
        col_dict = {process_raw_data.WORDS[i]: process_raw_data.WORDS_RATINGS[i] for i in range(len(process_raw_data.WORDS))}

        for image in word_IA_dict.keys():
            image_df = experiment_df[experiment_df[process_raw_data.STIM_ID].str.contains(image)][relevant_col_list]
            for word in word_IA_dict[image].keys():  # only words with word IA!!
                tmp_conf_list = list()
                image_list.append(image)
                word_list.append(word)
                word_IA_list.append(word_IA_dict[image][word])
                for ind, row in image_df.iterrows():
                    for col in col_dict.keys():
                        if row[col] == word:
                            tmp_conf_list.append(row[col_dict[col]])
                mean_conf = statistics.mean(tmp_conf_list)
                mean_confidence_list.append(mean_conf)

        result_df = pd.DataFrame({"image": image_list, "word": word_list, "word IA": word_IA_list, "mean confidence rating": mean_confidence_list})
        result_df.to_csv(os.path.join(save_path, file_name))

    print('correlation between word IA and the average confidence rating of word')
    correl = pg.corr(x=result_df["word IA"], y=result_df["mean confidence rating"], alternative='two-sided', method='pearson')
    print(correl)
    # plot
    plt.clf()
    plt.figure()
    sns.set_style('whitegrid')
    sns.regplot(x=result_df["mean confidence rating"], y=result_df["word IA"], x_jitter=.05)
    plt.xticks(np.arange(1, 6, 1.0))
    plt.title("Image IA as a Function of Mean Confidence Rating", fontsize=14)
    plt.xlabel("Mean Confidence Rating of Word")
    plt.ylabel("Word IA")
    plt.savefig(os.path.join(save_path, f"word_IA_confidence.png"))

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
    print("---PREPARE DATA FOR ANALYSIS---")
    experiment_df = process_responses(experiment_df, save_path, load=False)
    experiment_df = experiment_df[experiment_df.columns.drop(list(experiment_df.filter(regex='Unnamed')))]
    experiment_df = responses_spelling(experiment_df, save_path, load=False, conversion_file=True)  # Conversion file True after adding a "corrected" column to the output file!
    experiment_df = experiment_df[experiment_df.columns.drop(list(experiment_df.filter(regex='Unnamed')))]
    experiment_df = lemmatize(experiment_df, save_path, load=True)  # "True" only after adding a "approved" column to the output file!
    experiment_df = assert_no_duplicate_resps_within_subject(experiment_df)
    count_unique_words(experiment_df)
    count_no_response_trials(experiment_df)
    print("---DATABASE IS READY: LET THE ANALYSIS BEGIN!---")
    images_aucs_dict, rare_word_count, word_count_dict, image_only_rare_words_df = calculate_image_aucs(experiment_df, save_path, load=False)  # ORIGINAL AUC CALCULATION
    word_IA_dict = calculate_word_IAs(images_aucs_dict, save_path)
    image_IA_dict = calculate_image_IAs(word_IA_dict, word_count_dict, save_path)
    print("---RARE WORD ANALYSIS---")
    calculate_word_freq_stats(experiment_df, save_path, word_IA_dict, load=False)
    rare_word_count = calc_rare_word_per_image(word_count_dict, save_path)
    image_IA_and_rarely_reported(image_IA_dict, rare_word_count, save_path)
    print("---CONFIDENCE ANALYSIS---")
    confidence_analysis(experiment_df, word_IA_dict, save_path, load=False)
    print("---PRESENTATION DURATION ANALYSIS---")
    analyze_image_presentation_duration(experiment_df, image_IA_dict, save_path)
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
