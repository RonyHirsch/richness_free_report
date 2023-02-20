import os
import pandas as pd
import numpy as np
import process_raw_data
import analyze_data

COL_SOA = "values.soa"
COL_IMG = "values.img_file"
COL_RESP = "response"
COL_TRIAL_CODE = "trialcode"
WORD_CODES = [f"d{i}" for i in range(1, 6)]
RATE_CODES = [f"rb{i}" for i in range(1, 6)]
COL_BLOCK_CODE = "blockcode"
COL_BLOCK_NUM = "blocknum"
COL_TRIAL_NUM = "trialnum"
DATE = "date"
TIME = "time"

CONF_MAP = {"Don't Know": 1, "Guess": 2, "Maybe": 3, "Confident": 4, "Very Confident": 5}


def reshape_chuyin(file_path, save_path):
    """
    Don't edit content at all, just reshape the dataframe s.t. each subject's trial is exactly 1 line
    :param file_path:
    :param save_path:
    :return:
    """
    raw_data = pd.read_csv(file_path)
    processed_data = raw_data[raw_data[COL_BLOCK_CODE] != "practice_block"]  # only experimental stimuli
    processed_data = processed_data[processed_data[COL_SOA] == 67]  # SOA: just 67 ms
    processed_data.rename(columns={"subject": process_raw_data.PROLIFIC_ID, COL_IMG: process_raw_data.STIM_ID},
                          inplace=True)
    processed_data = processed_data[[DATE, TIME, process_raw_data.PROLIFIC_ID, COL_TRIAL_CODE, process_raw_data.STIM_ID, COL_RESP, COL_BLOCK_NUM, COL_TRIAL_NUM]]
    # df result: stim name, sub name, words, word ratings
    result_list = list()
    result_cols = [DATE, TIME, process_raw_data.STIM_ID,
                   process_raw_data.PROLIFIC_ID] + process_raw_data.WORDS + process_raw_data.WORDS_RATINGS
    for stim in processed_data[process_raw_data.STIM_ID].unique():
        image_df = processed_data[processed_data[process_raw_data.STIM_ID] == stim]
        for sub in image_df[process_raw_data.PROLIFIC_ID].unique():
            sub_df = image_df[image_df[process_raw_data.PROLIFIC_ID] == sub]
            sub_date = sub_df[DATE].tolist()[0]
            sub_time = sub_df[TIME].tolist()[0]
            words = sub_df[sub_df[COL_TRIAL_CODE].isin(WORD_CODES)][COL_RESP].tolist()
            ratings = sub_df[sub_df[COL_TRIAL_CODE].isin(RATE_CODES)][COL_RESP].tolist()
            trial_list = [sub_date, sub_time, stim, sub] + words + ratings
            result_list.append(trial_list)
    result_df = pd.DataFrame(result_list, columns=result_cols)
    result_df.to_csv(os.path.join(save_path, "gist_batch_v1_all_mt_reshaped.csv"))
    return result_df


def too_many_responses_CHUYIN(stim_df):
    sub_ids = stim_df[process_raw_data.PROLIFIC_ID].tolist()
    sorted_df = stim_df.sort_values([DATE, TIME])
    to_drop = sorted_df.shape[0] - process_raw_data.STIM_REPS
    sorted_after_drop = sorted_df.head(-to_drop)
    sub_ids_after_drop = sorted_after_drop[process_raw_data.PROLIFIC_ID].tolist()
    subs_to_drop = [s for s in sub_ids if s not in sub_ids_after_drop]
    return subs_to_drop


def process_dataframe(df, save_path, load=False):
    if not load:
        for col in process_raw_data.WORDS_RATINGS:
            df.replace({col: CONF_MAP}, inplace=True)
        df.to_csv(os.path.join(save_path, "gist_batch_v1_all_mt_numeric_ratings.csv"))

    # REMOVE REPEATING STIMULI (PROCESSING OF RAW DATA)
    stimuli = list(df[process_raw_data.STIM_ID].unique())
    too_many = 0
    too_few = 0
    for stim in stimuli:
        stim_df = df[df[process_raw_data.STIM_ID] == stim]
        if stim_df.shape[0] > process_raw_data.STIM_REPS:  # stimulus was seen by more than STIM_REPS subjects; as per Chuyin - remove LIFO
            subs_to_drop = too_many_responses_CHUYIN(stim_df)
            df = df[~df[process_raw_data.PROLIFIC_ID].isin(subs_to_drop)]  # all subjects not in the drop list
            too_many += 1
        elif stim_df.shape[0] < process_raw_data.STIM_REPS:  # stimulus was seen by less than STIM_REPS subjects: remove from analysis
            df = df[df[process_raw_data.STIM_ID] != stim]
            too_few += 1
        # else, stimulus was seen exactly STIM_REPS times, all is good
    print(
        f"{too_many} stimuli had more than {process_raw_data.STIM_REPS} responses; exceeding responses were removed LIFO")
    print(
        f"{too_few} stimuli had more than {process_raw_data.STIM_REPS} responses; stimuli were removed from all {too_few} participants")

    # sanity check
    for stim in stimuli:
        stim_df = df[df[process_raw_data.STIM_ID] == stim]
        if stim_df.shape[0] > process_raw_data.STIM_REPS or stim_df.shape[0] < process_raw_data.STIM_REPS:
            print("ERROR: stimuli were not filtered")
            return
    df.to_csv(os.path.join(save_path, "gist_batch_v1_all_mt_10_resps.csv"), index=False)
    df.drop(columns=[DATE, TIME], inplace=True)
    return df


def transform_chuyin(file_path, save_path, load=True):
    import sys
    orig_stdout = sys.stdout
    f = open(os.path.join(save_path, 'processing_log.txt'), 'w')
    sys.stdout = f  # reroute console prints to file logging

    if load:
        reshaped_df = pd.read_csv(file_path)
        reshaped_df.drop(columns=[x for x in reshaped_df.columns if "Unnamed" in x], inplace=True)
    #reshaped_df = reshape_chuyin(file_path, save_path)
    #reshaped_df = pd.read_csv(os.path.join(file_path, "gist_batch_v1_all_mt_reshaped.csv"))
    #processed_df = pd.read_csv(os.path.join(file_path, "gist_batch_v1_all_mt_10_resps.csv"))
    processed_df = process_dataframe(reshaped_df, save_path, load)
    analyze_data.analyze_data(processed_df, save_path)

    sys.stdout = orig_stdout  # bring it back
    f.close()


if __name__ == "__main__":

    transform_chuyin(
        file_path=r"..\gist_batch_v1_all_mt_numeric_ratings.csv",
        save_path=r"..\processed")