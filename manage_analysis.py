import os
import sys
import process_raw_data
import analyze_data

"""
Manage all pre-processing and analysis of the gist experimental data. 
"""

__author__ = "Rony Hirschhorn"


def manage_analysis(data_path, save_path, sequence_path, demog_data_path, stdout_file=False):
    """
    data_path: path to the folder where all the raw data csv files are (the outputs of the online experiment,
    1 file per subject)
    save_path: path to the folder where all the processed data will be saved.
    sequence_path: path to the folder containing all the possible stimulus sequences. Each participant run was executing
    a single sequence (one csv file), which controlled the order of the presented stimuli for that participant. This
    was set based on % on the order they started the experiment.
    demog_data_path: path to a csv file, which is the prolific demographic data table.
    stdout_file: it "True", then all the print() operations will write all the messages to a txt file, which will then
    be saved in save_path.
    """
    if stdout_file:
        orig_stdout = sys.stdout
        f = open(os.path.join(save_path, 'processing_log.txt'), 'w')
        sys.stdout = f  # reroute console prints to file logging

    print("****************************PRE-PROCESSING RAW DATA****************************")
    sub_dict, stim_dict, demographic_data = process_raw_data.manage_preprocessing(data_path=data_path,
                                                                                  save_path=save_path,
                                                                                  sequence_path=sequence_path,
                                                                                  demog_data_path=demog_data_path)

    print("****************************ANALYZE PRE-PROCESSED DATA****************************")
    analyze_data.manage_data_analysis(sub_dict, stim_dict, save_path)

    if stdout_file:
        sys.stdout = orig_stdout  # bring it back
        f.close()
    return


if __name__ == "__main__":
    manage_analysis(
        data_path=r"..\data",
        save_path=r"..\processed",
        sequence_path=r"..\all_seqs",
        demog_data_path=r"..\prolific_export.csv",
        stdout_file=True)
