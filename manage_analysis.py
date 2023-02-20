import os
import sys
import process_raw_data
import analyze_data
import pandas as pd

"""
Manage all pre-processing and analysis of the gist experimental data. 
"""

__author__ = "Rony Hirschhorn"


def filter_out_images(sub_dict, stim_dict, save_path, filter_file):
    filter_images = pd.read_csv(filter_file)
    image_list = filter_images.loc[:, "Image Number"].tolist()
    print(f"{len(stim_dict.keys())} stimuli before filtering")
    for image in image_list:
        stim_dict = {im: v for im, v in stim_dict.items() if image not in im}  # filter out image from dict
        for sub in sub_dict:
            for sess in sub_dict[sub]:
                df = sub_dict[sub][sess]
                df = df[~df["stim_name"].str.contains(image)]  # filter our image trials
                sub_dict[sub][sess] = df
    print(f"{len(stim_dict.keys())} stimuli after filtering")
    return sub_dict, stim_dict


def manage_analysis(data_path, save_path, sequence_path, demog_data_path, filter_file=None, stdout_file=False):
    if stdout_file:
        orig_stdout = sys.stdout
        f = open(os.path.join(save_path, 'processing_log.txt'), 'w')
        sys.stdout = f  # reroute console prints to file logging

    print("****************************PRE-PROCESSING RAW DATA****************************")
    sub_dict, stim_dict, demographic_data = process_raw_data.manage_preprocessing(data_path=data_path,
                                                                                  save_path=save_path,
                                                                                  sequence_path=sequence_path,
                                                                                  demog_data_path=demog_data_path)

    if filter_file:
        print("****************************FILTERING OUT BLURRED IMAGES****************************")
        sub_dict, stim_dict = filter_out_images(sub_dict, stim_dict, save_path, filter_file)

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
        sequence_path=r"..\sequences",
        filter_file=None,
        demog_data_path=r"..\prolific_export_....csv",
        stdout_file=True)