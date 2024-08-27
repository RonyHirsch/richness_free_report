import os
import pandas as pd
import plotter

existence_tags = ["existInImage", "conceptual word", "nonexist:  confusion", "nonexist: right gist non-existent item", "nonexist: no connection at all"]
existence_colors = {"existInImage": "#FDECEF",
                    "conceptual word": "#D59AB3",
                    "nonexist:  confusion": "#975E7B",
                    "nonexist: right gist non-existent item": "#612940",
                    "nonexist: no connection at all": "#E1E0E0"
                    }
existence_names = {"existInImage": "Exist",
                    "conceptual word": "Conceptual",
                    "nonexist:  confusion": "Confusion",
                    "nonexist: right gist non-existent item": "Insertion",
                    "nonexist: no connection at all": "Unrelated"
                    }


def confidence_per_existence_tag_exploratory(exist_path, result_save_path):
    study = "exploratory"

    # create save_path
    study_save_path = os.path.join(result_save_path, study)
    if not os.path.exists(study_save_path):
        os.makedirs(study_save_path)

    # load df
    word_exist_df = pd.read_csv(exist_path)

    # a single column denoting the type of existence tag given to this word in this image
    word_exist_df["tag"] = word_exist_df[existence_tags].idxmax(axis=1)
    word_exist_df = word_exist_df[["version", "image", "word", "word IA", "mean confidence rating", "tag"]]

    # plot
    plotter.plot_raincloud(df=word_exist_df, data_col_name="mean confidence rating", group_col_name="tag",
                           group_order=existence_tags, group_color_dict=existence_colors,
                           save_path=study_save_path, save_name=f"all_confidence_per_tag", y_title="Mean Confidence",
                           x_title="Category", group_name_dict=existence_names, marker_size=50, marker_alpha=0.3,
                           marker_spread=0.15, group_spacing=0.5, violin_width=0.35, violin_alpha=0.7, ymin=1.0,
                           ymax=5.05, yskip=1)
    word_exist_df.to_csv(os.path.join(study_save_path, f"all_confidence_per_tag.csv"), index=False)

    return word_exist_df


def confidence_per_existence_tag_prereg(exist_path, result_save_path, conf_df_path):
    study = "preregistered"

    # create save_path
    study_save_path = os.path.join(result_save_path, study)
    if not os.path.exists(study_save_path):
        os.makedirs(study_save_path)

    # load word tag df
    word_exist_df = pd.read_csv(exist_path)
    # a single column denoting the type of existence tag given to this word in this image
    word_exist_df["tag"] = word_exist_df[existence_tags].idxmax(axis=1)

    # load confidence dfs
    confidence_dfs = list()
    for experiment in ["blackwhite", "blurred", "replication"]:
        token = experiment if experiment == "blackwhite" else ("intact" if experiment == "replication" else "blur")
        conf_df = pd.read_csv(os.path.join(conf_df_path, f"word_IA_confidence_{token}.csv"))
        conf_df = conf_df.loc[:, ~conf_df.columns.str.contains('^Unnamed')]
        conf_df["image"] = conf_df["image"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])  # only image name
        conf_df["version"] = experiment
        confidence_dfs.append(conf_df)
    confidence_df = pd.concat(confidence_dfs)

    # merge confidence with word IA dataframes
    combined_df = pd.merge(
        left=word_exist_df,  # all words with IA scores
        right=confidence_df,  # all words with confidence
        how='left',
        left_on=['word', 'image', 'version'],
        right_on=['word', 'image', 'version'])

    combined_df = combined_df[["version", "image", "word", "word IA_x", "mean confidence rating", "tag"]]
    combined_df = combined_df.rename(columns={"word IA_x": "word IA"})

    # plot
    plotter.plot_raincloud(df=combined_df, data_col_name="mean confidence rating", group_col_name="tag",
                           group_order=existence_tags, group_color_dict=existence_colors,
                           save_path=study_save_path, save_name=f"all_confidence_per_tag", y_title="Mean Confidence",
                           x_title="Category", group_name_dict=existence_names, marker_size=50, marker_alpha=0.3,
                           marker_spread=0.15, group_spacing=0.5, violin_width=0.35, violin_alpha=0.7, ymin=1.0,
                           ymax=5.05, yskip=1)
    combined_df.to_csv(os.path.join(study_save_path, f"all_confidence_per_tag.csv"), index=False)

    return combined_df


if __name__ == "__main__":
    result_save_path = r"...\confidence"

    # word IA and existence scores path (for exploratory and preregistered studies)
    exist_exp_path = r"...\exploratory\aggregation\word_nonexist\words_per_image_all_new.csv"
    # confidence per existence taq
    expl_df = confidence_per_existence_tag_exploratory(exist_exp_path, result_save_path)
    expl_df["study"] = "exploratory"

    # same for preregistered (word IA sheet doesn't include confidence)
    exist_prereg_path = r"...\preregistered\aggregation\word_nonexist\word_frequency_count_all_new.csv"
    conf_df_path = r"...\preregistered\aggregation\word_conf"
    prereg_df = confidence_per_existence_tag_prereg(exist_prereg_path, result_save_path, conf_df_path)
    prereg_df["study"] = "preregistered"

