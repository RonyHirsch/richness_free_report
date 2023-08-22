import os
import pandas as pd

"""
Parse and process the data of the control experiments. 

In the first control experiment,  participants needed to choose which stimulus was richer (comparison).
In the second control experiment, participants needed to rate the perceived richness of the stimulus (intact or blurry) 
on a slider. 

The parsing is very similar in both experiments, and yields a csv file called "subject_df.csv", 
where each row=participant, and columns are information about their behavior during the experiment. 

Further data analysis was then performed in JASP.
"""

__author__ = "Rony Hirschhorn"


EXP_NAME = "b_v_i_experiment"
CSV = ".csv"
PROLIFIC_ID = "id"
SEQ_NUM = "participant"  # The subject's serial number is used to set the stimulus sequence that they were presented with
DEMOG_SUB = "Participant id"
DEMOG_AGE = "Age"
DEMOG_SEX = "Sex"
DEMOG_NATIONALITY = "Nationality"
DEMOG_TIME_TAKEN_SEC = "Time taken"

RELEVANT_COLS_COMPARISON = ["participant", "OS", "id", "browser", "xResolution", "yResolution", "frame_rate_curr",
                            "single_frame", "trial_dur_in_frames", "exp_stim_dur_frames", "stim_frames_actual", "single_frame_dur",
                            "exp_stim_left", "exp_stim_right", "exp_correct", "resp_keyboard.keys", "resp_keyboard.corr",
                            "resp_keyboard.rt", "exp_trials.thisRepN"]

RELEVANT_COLS_SLIDER = ["participant", "OS", "id", "browser", "xResolution", "yResolution", "frame_rate_curr",
                        "single_frame", "trial_dur_in_frames", "exp_stim_dur_frames", "stim_frames_actual", "single_frame_dur",
                        "exp_stim", "trial_slider.response", "trial_slider.rt", "exp_trials.thisRepN"]


def load_subs(data_path):
    sub_dict = dict()
    # files are Pavlovia files; some Prolific subjects might have attempted to do the experiment more than once
    file_names = [f for f in os.listdir(data_path) if f.endswith(CSV) and EXP_NAME in f]
    for f in file_names:
        file_data = pd.read_csv(os.path.join(data_path, f))
        file_id = file_data[PROLIFIC_ID].unique()[0]  # there's only 1 Prolific ID per Pavlovia data file
        if file_id not in sub_dict:
            sub_dict[file_id] = dict()
        sub_pavlovia_order = file_data[SEQ_NUM][0]  # in a single file there is a single participant number
        sub_dict[file_id][sub_pavlovia_order] = file_data
    print(f"{len(sub_dict.keys())} individual subjects (Prolific IDs) found in database.")
    return sub_dict


def dempgraphic_stats(demographic_data, save_path):
    """
    Calculate and save basic descriptives about the subjects' dempgraphic background.
    :param demographic_data: A dataframe containing demographic information about subjects --> AFTER filter_subs was run
    :param save_path: path to save outputs in (folder)
    :return: Nothing; saves the data in csv files under save_path
    """
    cnt_sex = demographic_data.groupby(DEMOG_SEX).count()[DEMOG_SUB]
    cnt_nation = demographic_data.groupby(DEMOG_NATIONALITY).count()
    cnt_nation = cnt_nation.iloc[:, 1:2]
    cnt_nation.rename({DEMOG_SUB: "count"}, axis=1, inplace=True)
    cnt_nation.to_csv(os.path.join(save_path, "demog_nation.csv"))
    demographic_data.loc[:, DEMOG_AGE] = pd.to_numeric(demographic_data[DEMOG_AGE])  # make sure age is a number
    mean_age = demographic_data[DEMOG_AGE].mean()
    std_age = demographic_data[DEMOG_AGE].std()
    min_age = demographic_data[DEMOG_AGE].min()
    max_age = demographic_data[DEMOG_AGE].max()
    timed_subs = demographic_data[demographic_data[DEMOG_TIME_TAKEN_SEC].notnull()]
    mean_minutes = timed_subs[DEMOG_TIME_TAKEN_SEC].mean() / 60
    std_minutes = timed_subs[DEMOG_TIME_TAKEN_SEC].std() / 60
    pd.DataFrame({"age mean": [mean_age], "age std": [std_age], "age min": [min_age], "age max": [max_age],
                  "time (minutes) mean": [mean_minutes], "time (minutes) std": [std_minutes],
                  cnt_sex.index[0]: [cnt_sex[cnt_sex.index[0]]], cnt_sex.index[1]: [cnt_sex[cnt_sex.index[1]]]}).to_csv(os.path.join(save_path, "demog_age_sex.csv"))
    return


def process_df(data, is_comparison=False):
    if is_comparison:
        sub_cols = ["frame_rate_curr", "single_frame", "trial_dur_in_frames", "exp_stim_dur_frames",
                    "stim_frames_actual",
                    "single_frame_dur", "exp_stim_left", "exp_stim_right", "exp_correct", "resp_keyboard.keys",
                    "resp_keyboard.corr", "resp_keyboard.rt", "exp_trials.thisRepN"]
    else:
        sub_cols = ["frame_rate_curr",  "single_frame", "trial_dur_in_frames", "exp_stim_dur_frames", "stim_frames_actual",
                    "single_frame_dur", "exp_stim", "trial_slider.response", "trial_slider.rt", "exp_trials.thisRepN"]
    data = data.dropna(subset=sub_cols, how='all').reset_index(drop=True, inplace=False)  # remove rows where all these columns are empty
    # this block corrects for excess pre-experiment data rows
    first = data.index[data["exp_trials.thisRepN"] == 0].tolist()[0]  # get first trial index
    pre_first = data.loc[:first, :].shape[0]
    if pre_first > 2:
        data = data.iloc[first-1:].reset_index(drop=True, inplace=False)

    cols = data.columns
    for index, row in data.iterrows():
        if index % 2 == 0:
            for col in cols:
                if pd.isna(data.loc[index, col]):  # for every column where the value is empty
                    data.at[index, col] = data.loc[index + 1, col]  # take it from the subsequent row
        else:  # this is the second row of the same trial from above
            continue
    # now we took all the information from subsequent rows, remove redundant rows
    result = data[data.index % 2 == 0].reset_index(drop=True)
    return result


def process_practice_df(data):
    prac_cols = ["practice_correct", "practice_resp_keyboard.keys", "practice_resp_keyboard.corr",
                 "practice_resp_keyboard.rt", "practice_trials.thisRepN"]
    data = data.dropna(subset=prac_cols, how='all').reset_index(drop=True, inplace=False)
    for index, row in data.iterrows():
        if index % 2 == 0:
            for col in data.columns:
                if pd.isna(data.loc[index, col]):  # for every column where the value is empty
                    data.at[index, col] = data.loc[index + 1, col]  # take it from the subsequent row
        else:  # this is the second row of the same trial from above
            continue
    # now we took all the information from subsequent rows, remove redundant rows
    result = data[data.index % 2 == 0].reset_index(drop=True)
    return result


def unify_subjects(subs_dict, save_path, is_comparison):
    if is_comparison:
        relevant_cols = RELEVANT_COLS_COMPARISON
    else:
        relevant_cols = RELEVANT_COLS_SLIDER
    subs_prolific_ids = list(subs_dict.keys())
    # extract experimental data
    sub_data_list = list()
    for sub in subs_prolific_ids:
        for sess in subs_dict[sub]:  # we assume there's 1 such session per subject anyway
            data = subs_dict[sub][sess]
            try:
                data_relevant = data[relevant_cols]
            except Exception:
                print(f"This subject did not complete the experiment: {sub}")
                subs_dict.pop(sub)
                continue
            data_processed = process_df(data_relevant, is_comparison)
            sub_data_list.append(data_processed)
    subject_df = pd.concat(sub_data_list)
    subject_df.to_csv(os.path.join(save_path, "subject_df.csv"), index=False)
    return subject_df


def manage_comparison(data_path, demog_data_path, save_path):
    """
    The comparison experiment, choosing which image was richer.

    data_path: the path to the folder containing individual csvs (raw data outputs), each belongs to one participant.
    demog_data_path: the demographic data table provided by prolific with information about all the people who
    signed up for this experiment.
    save_path: the path to which subject_df csv file will be saved.
    """
    print("---LOADING SUBJECT DATA FILES---")
    subs_raw_all = load_subs(data_path)
    print("---PRE-PROCESSING: PARSING SUBJECTS---")
    demographic_data = pd.read_csv(demog_data_path)
    # now, sync the demographic data table s.t it will include all the subjects in the data dictionary
    demographic_data = demographic_data[demographic_data[DEMOG_SUB].isin(list(subs_raw_all.keys()))].reset_index(drop=True)
    demographic_data = demographic_data.drop(demographic_data.columns[[0]], axis=1)

    print(f"***{demographic_data.shape[0]}*** subjects are included in the experiment's database.")
    dempgraphic_stats(demographic_data, save_path)
    subject_df = unify_subjects(subs_raw_all, save_path, is_comparison=True)
    print(f"***{len(subject_df['participant'].unique())}*** subjects are included in the experiment's analysis.")

    """
    Prepare additional dfs for data analysis in JASP. The original subject_df is already saved by this point.
    """
    # FOR JASP ANALYSIS: make it so that we have only one row per participant
    subject_df_agg_lr = subject_df.groupby(["participant", "exp_correct"]).mean().reset_index()
    subject_df_agg_lr["resp_keyboard.corr"] = 100 * subject_df_agg_lr["resp_keyboard.corr"]  # turn proportion into %
    subject_df_row_per_sub = subject_df_agg_lr.pivot_table(index="participant", columns="exp_correct").reset_index()
    subject_df_row_per_sub.columns = [f"{col[1]}_{col[0]}" for col in subject_df_row_per_sub.columns]  # flatten column names
    subject_df_row_per_sub.to_csv(os.path.join(save_path, f"subject_df_pcnt_correct_sides.csv"), index=False)
    return


def manage_slider(data_path, demog_data_path, save_path):
    """
    The slider experiment.

    data_path: the path to the folder containing individual csvs (raw data outputs), each belongs to one participant.
    demog_data_path: the demographic data table provided by prolific with information about all the people who
    signed up for this experiment.
    save_path: the path to which subject_df csv file will be saved.
    """
    print("---LOADING SUBJECT DATA FILES---")
    subs_raw_all = load_subs(data_path)
    print("---PRE-PROCESSING: PARSING SUBJECTS---")
    demographic_data = pd.read_csv(demog_data_path)
    # now, sync the demographic data table s.t it will include all the subjects in the data dictionary
    demographic_data = demographic_data[demographic_data[DEMOG_SUB].isin(list(subs_raw_all.keys()))].reset_index(drop=True)
    demographic_data = demographic_data.drop(demographic_data.columns[[0]], axis=1)
    pavlovia_subs = list(subs_raw_all.keys())
    for participant in pavlovia_subs:
        if participant not in demographic_data[DEMOG_SUB].tolist():
            print(f"Subject {participant} does not have demographic data; removed")
            subs_raw_all.pop(participant)
    print(f"***{demographic_data.shape[0]} = {len(subs_raw_all.keys())}*** subjects are included in the experiment's database.")
    dempgraphic_stats(demographic_data, save_path)
    subject_df = unify_subjects(subs_raw_all, save_path, is_comparison=False)
    print(f"***{len(subject_df['participant'].unique())}*** subjects are included in the experiment's analysis.")

    """
    Prepare additional dfs for data analysis in JASP. The original subject_df is already saved by this point.
    """
    # pre-processing: extract stimulus
    subject_df["stim_id"] = subject_df["exp_stim"].str.rsplit('/', n=1).str[-1].str.rsplit('.', n=1).str[0]
    subject_df["version"] = subject_df["exp_stim"].str.rsplit('/', n=2).str[-2]
    # remove redundant columns
    subject_df_filtered = subject_df.drop(columns=["exp_trials.thisRepN", "exp_stim_dur_frames", "trial_dur_in_frames", "frame_rate_curr"], inplace=False)
    # group by subject, separately for each version
    subject_df_grouped = subject_df_filtered.groupby(["participant", "version"]).mean().reset_index()
    subject_df_row_per_sub = subject_df_grouped.pivot_table(index="participant", columns="version").reset_index()
    subject_df_row_per_sub.columns = [f"{col[1]}_{col[0]}" for col in subject_df_row_per_sub.columns]  # flatten column names
    # save for JASP repeated-measures ANOVA
    subject_df_row_per_sub.to_csv(os.path.join(save_path, f"subject_df_per_version.csv"), index=False)

    # now explore the first trial only
    sub_df_first_trial = subject_df[subject_df["exp_trials.thisRepN"] == 0]
    sub_df_first_trial.to_csv(os.path.join(save_path, f"subject_df_first_trial.csv"), index=False)
    return


if __name__ == "__main__":
    manage_comparison(data_path=r"..\raw",
                  demog_data_path=r"..\prolific_export.csv",
                  save_path=r"..\processed")
    manage_slider(data_path=r"..\raw",
                    demog_data_path=r"..\prolific_export.csv",
                    save_path=r"..\processed")
