import os
import pandas as pd
import numpy as np

"""
Manage all pre-processing of the gist experimental data. 
"""

__author__ = "Rony Hirschhorn"

EXP_NAME = "gist"
CSV = ".csv"
SEQ_NUM = "participant"  # The subject's serial number is used to set the stimulus sequence that they were presented with
SEQ_FILE = "sequence"  # The actual sequence file that was loaded
OS = "OS"
HEIGHT = "screenHeight"
WITDH = "screenWidth"
RR_COMP = "frameRate"
RR_ACTUAL = "frame_rate_curr"
RR_FRAMEDUR = "single_frame"
PROLIFIC_ID = "id"
MASK_NAME = "mask_name"
MASK_DUR = "mask_duration"
WORDS = [f"word_{i}.text" for i in range(1, 6)]
WORDS_RATINGS = [f"word_{i}_slider.response" for i in range(1, 6)]
WORDS_RATINGS_RTS = [f"word_{i}_slider.rt" for i in range(1, 6)]
TRIAL_START = "trial_start"
STIM_START = "stim_start"
STIM_END = "stim_end"
STIM_DUR_FRAMES = "calc_dur_in_frames"
STIM_ID = "trial_stimulus"
STIM_ID_EXTRA_COL = "stim_name"
TRIAL_NUMBER = "trials.thisIndex"
SEQ_LOADER = "sequence_loader"
PRACTICE = "practice"
INSTRUCTIONS = "instructions"
MASK = "mask"
REMOVED_PAVLOVIA_SESS = list()
REMOVED_PROLIFIC_SUBS = list()
STIM_REPS = 10
DEMOG_LANG = "First Language"
DEMOG_SUB = "Participant id"  #  ****** IN OLD PROLIFIC VERSIONS *******: "participant_id"
DEMOG_DATA_EXPIRED = "DATA EXPIRED"
DEMOG_NATIONALITY = "Nationality"
DEMOG_SEX = "Sex"
DEMOG_AGE = "Age"  #  ****** IN OLD PROLIFIC VERSIONS *******: "age"
DEMOG_TIME_COMPLETE = "completed_date_time"
DEMOG_TIME_TAKEN_SEC = "Time taken"  #  ****** IN OLD PROLIFIC VERSIONS *******: "time_taken"
# not used so for removal
END_BUTTON = "end_button"
REDU_1 = "image_name"
REDU_2 = "duration_sec"
SEQUENCE_NUMBER = "seq_num"
EXP_NAME_COL = "expName"
TRIALS = "trials."
POST_PRACTICE = "post_practice_mouse"
STIM_FRAMES_OLD = "stimulus_frames"


def load_subs(data_path):
    sub_dict = dict()
    # files are Pavlovia files; some Prolific subjects might have attempted to do the experiment more than once
    file_names = [f for f in os.listdir(data_path) if f.endswith(CSV) and EXP_NAME in f]
    for f in file_names:
        file_data = pd.read_csv(os.path.join(data_path, f))
        if file_data.empty:
            continue  # no subject information in that file - probably an in-&-out
        file_id = file_data[PROLIFIC_ID].unique()[0]  # there's only 1 Prolific ID per Pavlovia data file
        if file_id not in sub_dict:
            sub_dict[file_id] = dict()
        sub_pavlovia_order = file_data[SEQ_NUM][0]  # in a single file there is a single participant number
        sub_dict[file_id][sub_pavlovia_order] = file_data
    print(f"{len(sub_dict.keys())} individual subjects (Prolific IDs) found in database.")
    return sub_dict


def load_seqs(sequence_path):
    seq_dict = {f: pd.read_csv(os.path.join(sequence_path, f)) for f in os.listdir(sequence_path)}
    return seq_dict


def check_sess_responses(session_df):
    try:
        trial_df = session_df[session_df[STIM_ID_EXTRA_COL].notna()]  # Experimental trials information
        word_df = trial_df[WORDS]
        if word_df.isnull().all().all():
            return False
    except Exception as e:  # this means that there was no stim ID extra column which happens if the first trial was never fully completed
        return False
    return True


def filter_sess(subs_dict, sequences):
    """
    This method goes over each subject's Pavlovia files (experimental sessions) and filters them. Technically,
    each subject (recruited with Prolific) was supposed to have a single Pavlovia file associated with them.
    Sometimes, the same subject has attempted to enter the experiment multiple times due to an error online. A subject
    might have also entered the experiment and left before any data was collected. This method is supposed to filter
    those sessions such that subjects will only remain with sessions in which experimental data was collected from them.
    :param subs_dict: A dictionary where key = subject Prolific ID, value=a dict where key=Pavlovia session number,
    value=the session's data.
    :param sequences: The stimuli in this experiment were not randomly drawn; I pre-generated stimulus sequence
    files, each file corresponding to a single subject's experimental run. The idea was to make sure that each
    stimulus image is seen by 10 different subjects (=appears in 10 different sequence files). Thus, each Pavlovia run
    (=session file) is associated with a single sequence file, based on the subject's Pavlovia order (%).
    :return: subs_dict, after filtering-out sessions with no experimental data.
    """
    subs_list = list(subs_dict.keys())
    for sub in subs_list:
        sub_sessions = subs_dict[sub]
        # Generate a dictionary (sub_seqs) where key=session (Pavlovia) number, value=session's sequence file name
        try:
            sub_seqs = {s: sub_sessions[s][SEQ_FILE].dropna(axis=0).unique()[0] for s in sub_sessions}
        except Exception:  # Subject session file doesn't even have a "sequence" column - nothing was run or even loaded
            to_pop = list()
            for s in sub_sessions:
                if SEQ_FILE not in sub_sessions[s].columns:
                    to_pop.append(s)
                    REMOVED_PAVLOVIA_SESS.append(sub_sessions[s])
            p = [subs_dict[sub].pop(s) for s in to_pop]
            sub_seqs = {s: sub_sessions[s][SEQ_FILE].dropna(axis=0).unique()[0] for s in sub_sessions}

        # Check the sessions associated with each subject: filter out those w/o *experimental* data
        # Remove Pavlovia data files that have no experimental data in them
        for sess_num in sorted(list(sub_seqs.keys())):
            if STIM_ID not in sub_sessions[sess_num].columns:  # No data was even collected in this session
                sub_seqs.pop(sess_num)
                r = subs_dict[sub].pop(sess_num)  # This does not count as an additional session
                REMOVED_PAVLOVIA_SESS.append(r)
                continue
            if sub_sessions[sess_num][STIM_ID].dropna(axis=0).unique().shape[0] == 0:  # No experimental stimuli
                sub_seqs.pop(sess_num)
                r = subs_dict[sub].pop(sess_num)
                REMOVED_PAVLOVIA_SESS.append(r)
                continue

        # Check the remaining sessions:
        # If subject has exactly one session containing experimental data, all is well at this point
        if len(sub_seqs) > 1:  # Subject has more than one session containing *experimental* data
            if len(sub_seqs) == 2:
                invalid_sess_flag = 0  # flag to raise if one of them is invalid
                seq = 0  # session number of the invalid session
                if list(sub_seqs.values())[0] != list(sub_seqs.values())[1]:  # Not the same sequence was loaded
                    for sess_num in sorted(list(sub_seqs.keys())):  # Go over them by order
                        sess_str = sub_seqs[sess_num][-7:]
                        selected_seq = list(sequences[sess_str][STIM_ID_EXTRA_COL])  # Planned stim
                        presented_seq = list(sub_sessions[sess_num][STIM_ID_EXTRA_COL].dropna(axis=0))  # Presented stim
                        if len(selected_seq) != len(presented_seq) and not invalid_sess_flag:
                            # A partial session (the first that was encountered), raise flag
                            invalid_sess_flag = 1
                            seq = sess_num
                        elif len(selected_seq) == len(presented_seq) and invalid_sess_flag:
                            # A valid, complete session, after we already encountered an invalid one - get rid of it
                            sub_seqs.pop(seq)
                            r = subs_dict[sub].pop(seq)
                            REMOVED_PAVLOVIA_SESS.append(r)
                        # When there are 2 sessions and BOTH PARTIAL, we don't do anything at this stage

                    if not invalid_sess_flag:  # Subject has 2 sessions, both valid (complete), DIFFERENT SEQUENCES!
                        # choose the FIRST ONE as the valid one
                        for sess_num in sorted(list(sub_seqs.keys())):  # Go over them by order
                            if sorted(list(sub_seqs.keys())).index(sess_num) > 0:  # if this is the SECOND one, remove it
                                sub_seqs.pop(sess_num)
                                r = subs_dict[sub].pop(sess_num)
                                REMOVED_PAVLOVIA_SESS.append(r)


                else:  # Subject ID has 2 sessions, both have the **same** loaded stimulus sequence: didn't happen
                    raise Exception(f"ERROR: Subject {sub} did the experiment twice, with the same sequence!")
            else:  # More than 2 sessions containing experimental data for the same subject: didn't happen
                raise Exception(f"ERROR: Subject {sub}has MORE than two sessions containing experimental data!")
    print(f"{len(REMOVED_PAVLOVIA_SESS)} runs were removed due to containing no experimental data.")

    # For each session - check if any response was made. If no verbal responses were given, remove session.
    cnt = 0
    for sub in subs_list:
        sub_sessions = subs_dict[sub]
        sub_seqs = {s: sub_sessions[s][SEQ_FILE].dropna(axis=0).unique()[0] for s in sub_sessions}
        for sess_num in sorted(list(sub_seqs.keys())):
            if not check_sess_responses(sub_sessions[sess_num]):
                sub_seqs.pop(sess_num)
                r = subs_dict[sub].pop(sess_num)
                REMOVED_PAVLOVIA_SESS.append(r)
                cnt += 1
    print(f"{cnt} runs were removed due to containing no responses.")

    # Subjects' sessions are actually pre-set by stimulus sequences. Ideally, have we had 1 completed run per sequence
    # we'd have the exact coverage we wanted. As it didn't happen, we will now check all the sequences that repeated
    # between different subjects (i.e., 2 different Prolific IDs that saw the same stimulus sequence): if one of the
    # repetitions is a PARTIAL run, we will discard it.
    cnt = 0
    seq_name_dict = {s: 0 for s in sequences}
    for sub in subs_list:
        sub_sessions = subs_dict[sub]
        sub_sess_names = list(sub_sessions.keys())
        for sess in sub_sess_names:
            seq = sub_sessions[sess][SEQ_FILE].dropna(axis=0).unique()[0][-7:]
            seq_name_dict[seq] += 1
    seqs_exceed = [k for k, v in seq_name_dict.items() if v > 1]
    for seq in seqs_exceed:  # for all sequences that repeated across multiple subjects
        for sub in subs_list:
            sub_sessions = subs_dict[sub]
            sub_sess_names = list(sub_sessions.keys())
            for sess in sub_sess_names:
                if seq == sub_sessions[sess][SEQ_FILE].dropna(axis=0).unique()[0][-7:]:  # if this is a repeated seq
                    if not check_sess_stim(sub_sessions[sess], sequences):  # if this is a PARTIAL run
                        r = subs_dict[sub].pop(sess)  # remove it
                        REMOVED_PAVLOVIA_SESS.append(r)
                        cnt += 1
    print(f"{cnt} runs were removed due to being partial runs of a repeating stimulus sequence.")
    return subs_dict


def check_sess_stim(session_df, sequences):
    # check all planned experimental stimuli were presented
    seq_num = session_df[SEQ_FILE].dropna(axis=0).unique()[0][-7:]
    selected_seq = list(sequences[seq_num][STIM_ID_EXTRA_COL])  # Stim to be presented
    presented_seq = list(session_df[STIM_ID_EXTRA_COL].dropna(axis=0))  # Stim that were presented
    if len(selected_seq) != len(presented_seq):  # Not all planned stimuli were presented (the other direction can't technically occur)
            return False  # session is incomplete
    if len(selected_seq) == len(presented_seq):  # The planned number of stimuli were presnted
        if presented_seq != selected_seq:  # But not all planned stimuli were presented
            return False
    return True


def filter_subs(subs_dict, sequences, demog_data_path):
    """
    After filter_sess was run, we remove from our subject database all subjects who have no experimental sessions left.
    :param subs_dict: A dictionary where key = subject Prolific ID, value=a dict where key=Pavlovia session number,
    value=the session's data --> AFTER filter_sess was run
    :param sequences: The stimuli in this experiment were not randomly drawn; I pre-generated stimulus sequence
    files, each file corresponding to a single subject's experimental run. The idea was to make sure that each
    stimulus image is seen by 10 different subjects (=appears in 10 different sequence files). Thus, each Pavlovia run
    (=session file) is associated with a single sequence file, based on the subject's Pavlovia order (%).
    :param demog_data_path: path to the demographic data csv file
    :return:
    subs_dict: the same dict, but w/o sessions that are not valid and w/o subjects that have no/partial data
    demographic_data: A dataframe containing demographic information about the remaining subjects
    """
    subs_removed = 0

    subs_list = list(subs_dict.keys())
    for sub in subs_list:
        sub_sessions = subs_dict[sub]
        sub_sess_names = list(sub_sessions.keys())
        if len(sub_sess_names) == 0:  # This subject has NO experimental sessions
            r = subs_dict.pop(sub)  # Remove it
            REMOVED_PROLIFIC_SUBS.append(r)
            continue  # move on to the next subject
    print(f"{len(REMOVED_PROLIFIC_SUBS)} subjects were removed due to having no experimental data. "
          f"{len(subs_dict.keys())} subjects left in database.")
    subs_removed += len(REMOVED_PROLIFIC_SUBS)

    cnt = 0
    subs_list = list(subs_dict.keys())
    for sub in subs_list:
        sub_cnt = 0
        sub_sessions = subs_dict[sub]
        sub_sess_names = list(sub_sessions.keys())
        for sess in sub_sess_names:
            if not check_sess_stim(subs_dict[sub][sess], sequences):
                sub_cnt += 1
        if sub_cnt > 1:
            raise Exception("ERROR: check manually")  # Luckily, that did not happen: no 2 partial sessions for the same subject.
        if sub_cnt > 0:
            cnt += 1
            r = subs_dict.pop(sub)  # Remove it
            REMOVED_PROLIFIC_SUBS.append(r)
    print(f"{cnt} subjects were removed for not completing the experiment (partial data).")
    subs_removed += cnt


    demographic_data = pd.read_csv(demog_data_path)
    # SPECIAL CASE (WEIRD, HAPPENED ONCE): subject with full pavlovia data (experiment), but no demongraphic data (prolific table)
    cnt = 0
    for sub in subs_list:
        subject_demographics = demographic_data[demographic_data[DEMOG_SUB] == sub]
        if subject_demographics.empty:  # this subject has NO demographic data
            if sub in subs_dict.keys():  # if this subject exists in the subject dictionary
                r = subs_dict.pop(sub)  # Remove it
                REMOVED_PROLIFIC_SUBS.append(r)
                cnt += 1
    subs_removed += cnt
    print(f"{cnt} subjects were removed for not having demographic data.")

    # now, sync the demographic data table s.t it will include all the subjects in the data dictionary
    demographic_data = demographic_data[demographic_data[DEMOG_SUB].isin(list(subs_dict.keys()))].reset_index(drop=True)
    #demographic_data = demographic_data.drop(demographic_data.columns[[0]], axis=1)
    print(f"***{demographic_data.shape[0]}*** subjects are included in the experiment's database. A total of {subs_removed} subjects were excluded.")
    return subs_dict, demographic_data


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
    ages = pd.to_numeric(demographic_data[DEMOG_AGE], errors='coerce')  # some have a "DATA_EXPIRED" entry instead of age, this turns them to nan
    mean_age = np.nanmean(ages)
    std_age = np.nanstd(ages)
    min_age = np.nanmin(ages)
    max_age = np.nanmax(ages)
    mean_minutes = np.nanmean(demographic_data[DEMOG_TIME_TAKEN_SEC]) / 60
    std_minutes = np.nanstd(demographic_data[DEMOG_TIME_TAKEN_SEC]) / 60
    pd.DataFrame({"age mean": [mean_age], "age std": [std_age], "age min": [min_age], "age max": [max_age],
                  "time (minutes) mean": [mean_minutes], "time (minutes) std": [std_minutes],
                  cnt_sex.index[0]: [cnt_sex[cnt_sex.index[0]]], cnt_sex.index[1]: [cnt_sex[cnt_sex.index[1]]]}).to_csv(os.path.join(save_path, "demog_age_sex.csv"))
    return


def count_stim_reps(subs_dict, sequences, save_path):
    """
    Counts how many times each stimulus was repeated.
    :param subs_dict: A dictionary where key = subject Prolific ID, value=a dict where key=Pavlovia session number,
    value=the session's data --> AFTER filter_sess AND filter_subs were run
    :param sequences: The stimuli in this experiment were not randomly drawn; I pre-generated stimulus sequence
    files, each file corresponding to a single subject's experimental run. The idea was to make sure that each
    stimulus image is seen by 10 different subjects (=appears in 10 different sequence files). Thus, each Pavlovia run
    (=session file) is associated with a single sequence file, based on the subject's Pavlovia order (%).
    :param save_path: path to save outputs in (folder)
    :return: all_stims: A dataframe with a single column, containing all stimulus names
    """
    all_stims = pd.concat(list(sequences.values()), ignore_index=True).drop_duplicates(ignore_index=True)
    stim_dict_ids = {s: list() for s in all_stims[STIM_ID_EXTRA_COL].tolist()}
    stim_dict_cnt = {s: 0 for s in all_stims[STIM_ID_EXTRA_COL].tolist()}
    seq_name_dict = {s: 0 for s in sequences}
    subs_list = list(subs_dict.keys())
    for sub in subs_list:
        sub_sessions = subs_dict[sub]
        sub_sess_names = list(sub_sessions.keys())
        for sess in sub_sess_names:
            presented = list(sub_sessions[sess][STIM_ID_EXTRA_COL].dropna(axis=0))  # Stim that were presented
            for stim in presented:
                stim_dict_ids[stim].append(sub)  # no single subject saw the same stimulus twice: see filter_sess
                stim_dict_cnt[stim] += 1

    for sub in subs_list:
        sub_sessions = subs_dict[sub]
        sub_sess_names = list(sub_sessions.keys())
        for sess in sub_sess_names:
            seq = sub_sessions[sess][SEQ_FILE].dropna(axis=0).unique()[0][-7:]
            seq_name_dict[seq] += 1
    seqs_exceed = [k for k, v in seq_name_dict.items() if v > 1]
    print(f"{len(seqs_exceed)} sequences were repeated by more than 1 subject.")

    rep_list = list()
    rep_num_list = list()
    many_rep_list = list()
    for stim in all_stims[STIM_ID_EXTRA_COL].tolist():
        if stim_dict_cnt[stim] < STIM_REPS:
            rep_list.append(stim)
            rep_num_list.append(stim_dict_cnt[stim])
        if stim_dict_cnt[stim] > STIM_REPS:
            many_rep_list.append(stim)
    missing_stim_df = pd.DataFrame({STIM_ID_EXTRA_COL: rep_list, "reps": rep_num_list})
    if not missing_stim_df.empty:
        print(f"{len(rep_list)} stimuli were seen by less than {STIM_REPS} subjects.")
        missing_stim_df.to_csv(os.path.join(save_path, "missing_stim.csv"))
    else:
        print(f"All stimuli were seen by at least {STIM_REPS} different subjects.")
    print(f"{len(many_rep_list)} stimuli were seen by more than {STIM_REPS} subjects.")
    return all_stims


def unify_stim_rows(stim_df):
    stim_df = stim_df.reset_index(drop=True)  # align indexes with number of rows
    cols = stim_df.columns
    for index, row in stim_df.iterrows():
        if index % 2 == 0:
            for col in cols:
                if pd.isna(stim_df.loc[index, col]):  # for every column where the value is empty
                    stim_df.at[index, col] = stim_df.loc[index+1, col]  # take it from the subsequent row
        else:  # this is the second row of the same trial from above
            continue
    # now we took all the information from subsequent rows, remove redundant rows
    result = stim_df[stim_df.index % 2 == 0].reset_index(drop=True)
    return result


def get_stim_resps(subs_db, demographic_data, all_stim_images, save_path):
    """
    Parse subs_db to get rid of redundant information in each subject's experimental dataframe, and create a dictionary
    containing the same information, but per image - so we would have an easier time analyzing data per stimulus
    (across all subjects who saw it).
    :param subs_db: A dictionary where key = subject Prolific ID, value=a dict where key=Pavlovia session number,
    value=the session's data --> AFTER filter_sess AND filter_subs were run
    :param demographic_data: A dataframe containing demographic information about subjects --> AFTER filter_subs was run
    :param all_stim_images: A dataframe with a single column, containing all stimulus names
    :param save_path: path to save outputs in (folder)
    :return:
    sub_dict: subs_db, after pre-processing of each subject's session dataframe
    stim_dict: A dictionary where key=stimulus image name, value=a dicitionary where key=subject's session
    DEMOGRAPHIC ORDER - NOT PAVLOVIA ID! and the value is a dataframe where each line contains information about a
    single trial (and one of them is the said
    image)
    """
    p = os.path.join(save_path, "processed_data")
    if not os.path.isdir(p):
        try:
            os.mkdir(p)
        except Exception as e:
            raise e

    subs_list = list(subs_db.keys())
    sub_dict = dict()  # processed data dictionary, same structure as subs_db
    stim_dict = {s: dict() for s in all_stim_images[STIM_ID_EXTRA_COL].tolist()}

    for sub in subs_list:
        sub_dict[sub] = dict()
        sub_index = demographic_data[demographic_data[DEMOG_SUB] == sub].index[0]  # subject order
        sub_sessions = subs_db[sub]
        sub_sess_names = list(sub_sessions.keys())
        for sess in sub_sess_names:
            sess_data = sub_sessions[sess]
            experimental_trials = sess_data[(sess_data[STIM_ID].notnull()) | sess_data[STIM_ID_EXTRA_COL].notnull()]
            cols_to_remove = [c for c in experimental_trials.columns if
                              (c.startswith(SEQ_LOADER) or c.startswith(PRACTICE) or c.startswith(INSTRUCTIONS)
                               or c.startswith(MASK) or c.startswith(END_BUTTON)
                               or c.startswith(TRIALS) or c.startswith(POST_PRACTICE))] + [SEQ_FILE, REDU_1, REDU_2,
                                                                                           SEQUENCE_NUMBER,
                                                                                           EXP_NAME_COL, STIM_FRAMES_OLD]
            relevant = experimental_trials.drop(columns=cols_to_remove, inplace=False, errors='ignore')
            relevant = unify_stim_rows(relevant)
            relevant.to_csv(os.path.join(p, f"{sess}.csv"))
            sub_dict[sub][sess] = relevant
            presented = list(relevant[STIM_ID_EXTRA_COL])
            for stim in presented:  # insert by order
                stim_dict[stim][sub_index] = relevant

    return sub_dict, stim_dict


def manage_preprocessing(data_path, demog_data_path, save_path, sequence_path):
    """
    Manages all the raw data pre-processing
    :param data_path: path to where the raw data files reside (folder)
    :param demog_data_path: path to the demographic data csv file
    :param save_path: path to save outputs in (folder)
    :param sequence_path: path to the folder which contains all the experimental stimulus sequences (folder)
    :return:
    sub_dict_processed: A dictionary where key=subject, value=a dictionary where key=subject's session id (order)
    and the value is a dataframe where each line contains information about a single trial
    stim_dict: A dictionary where key=stimulus image name, value=a dicitionary where key=subject's session id (order)
    and the value is a dataframe where each line contains information about a single trial (and one of them is the said
    image)
    demographic_data: the dataframe containing subjects' demographic information
    """
    print("---LOADING SUBJECT DATA FILES---")
    subs_raw_all = load_subs(data_path)
    sequences = load_seqs(sequence_path)
    print("---PRE-PROCESSING: PARSING SUBJECT SESSIONS---")
    subs_filtered_sessions = filter_sess(subs_raw_all, sequences)
    print("---PRE-PROCESSING: PARSING SUBJECTS---")
    subs_db, demographic_data = filter_subs(subs_filtered_sessions, sequences, demog_data_path)
    dempgraphic_stats(demographic_data, save_path)
    print("---PRE-PROCESSING: HANDLE STIM SEQUENCES---")
    all_stim_images = count_stim_reps(subs_db, sequences, save_path)
    print("---PRE-PROCESSING: STIMULUS IMAGES---")
    sub_dict_processed, stim_dict = get_stim_resps(subs_db, demographic_data, all_stim_images, save_path)
    return sub_dict_processed, stim_dict, demographic_data

