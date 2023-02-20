import os
import pandas as pd
import numpy as np
import process_raw_data
import plotter


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

RELEVANT_COLS = ["participant", "OS", "id", "browser", "xResolution", "yResolution", "frame_rate_curr",
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


def process_df(data):
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


def unify_subjects(subs_dict, save_path):
    subs_prolific_ids = list(subs_dict.keys())
    sub_data_list = list()
    for sub in subs_prolific_ids:
        for sess in subs_dict[sub]:  # we assume there's 1 such session per subject anyway
            data = subs_dict[sub][sess]
            try:
                data_relevant = data[RELEVANT_COLS]
            except Exception:
                print(f"This subject did not complete the experiment: {sub}")
                subs_dict.pop(sub)
                continue
            data_processed = process_df(data_relevant)
            sub_data_list.append(data_processed)
    subject_df = pd.concat(sub_data_list)
    subject_df.to_csv(os.path.join(save_path, "subject_df.csv"), index=False)
    return subject_df


def analyze_data(subject_df, save_path):

    # STEP 1: comparing intact to blurry within-subject (overall)
    subject_df["image_type"] = np.where(subject_df.exp_stim.str.contains("orig"), "intact", "blurry")
    grouped = subject_df.groupby(["participant", "image_type"]).mean().reset_index(inplace=False).loc[:, ["participant", "image_type", "trial_slider.response", "stim_frames_actual"]]
    grouped.to_csv(os.path.join(save_path, "subject_ratings_by_type.csv"), index=False)

    # this is a cosmetic conversion s.t. this could be analyzed by JASP
    trial_list = list()
    for t in ["intact", "blurry"]:
        grouped_cnt = grouped[grouped["image_type"] == t]
        grouped_cnt.drop("image_type", axis=1, inplace=True)
        grouped_cnt.rename(columns={"trial_slider.response": f"resp_{t}", "stim_frames_actual": f"stim_frames_{t}"}, inplace=True)
        trial_list.append(grouped_cnt)
    trial_list = [df.set_index("participant") for df in trial_list]
    trial_df = pd.concat(trial_list, axis=1).reset_index(drop=False, inplace=False)
    trial_df.to_csv(os.path.join(save_path, "subject_ratings_by_type_columns.csv"), index=False)
    # plot
    plotter.plot_raincloud_from_df_new(df=trial_df, col1="resp_intact", col1_name="Intact", col1_color="#003459",
                                       col2="resp_blurry", col2_name="Blurry", col2_color="#007EA7",
                                       title="Average Rating for Intact and Blurry Images",
                                       x_name="Group", y_name="Average Rating",
                                       save_path=save_path, save_name="subject_ratings_by_type_columns",
                                       alpha_1=0.85, alpha_2=0.85, min=0, max=1.04, interval=0.1,
                                       lines=True, line_w=0.6, scat_size=16)

    # STEP 2: take only the first trial for independent samples t-test
    subject_df_first = subject_df[subject_df["exp_trials.thisRepN"] == 0]
    subject_df_first.to_csv(os.path.join(save_path, "first_trial_ratings.csv"), index=False)
    # for plotting
    subject_df_first_intact = subject_df_first[subject_df_first["image_type"] == "intact"]["trial_slider.response"].tolist()
    subject_df_first_blur = subject_df_first[subject_df_first["image_type"] == "blurry"]["trial_slider.response"].tolist()
    d = {"first_intact_rating": subject_df_first_intact, "first_blurry_rating": subject_df_first_blur}
    first_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.items()]))
    plotter.plot_raincloud_from_df_new(df=first_df, col1="first_intact_rating", col1_name="Intact", col1_color="#003459",
                                       col2="first_blurry_rating", col2_name="Blurry", col2_color="#007EA7",
                                       title="First Trial Rating for Intact and Blurry Images",
                                       x_name="Group", y_name="First Trial Rating",
                                       save_path=save_path, save_name="first_trial_ratings",
                                       alpha_1=0.85, alpha_2=0.85, min=0, max=1.04, interval=0.1,
                                       lines=False, scat_size=16)
    return


def manage_analysis(data_path, demog_data_path, save_path, sequence_path):
    print("---LOADING SUBJECT DATA FILES---")
    subs_raw_all = load_subs(data_path)
    sequences = process_raw_data.load_seqs(sequence_path)
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
    subject_df = unify_subjects(subs_raw_all, save_path)
    print(f"***{len(subject_df['participant'].unique())}*** subjects are included in the experiment's analysis.")
    print("---ANALYSIS: ANALYZE DATA---")
    analyze_data(subject_df, save_path)
    return


if __name__ == "__main__":
    # PILOT 22-11-09
    manage_analysis(data_path=r"..\raw",
                    demog_data_path=r"..\prolific_export_...csv",
                    save_path=r"..\processed",
                    sequence_path=r"..\sequences")
