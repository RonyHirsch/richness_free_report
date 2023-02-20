import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from random import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def ks_test(data_1, data_2, alternative):
    """
    This test compares two sample distributions. It is a non-parametric test, and as such, can be applied to compare
    any two distributions regardless of whether you assume normal or uniform. The idea is that if two samples belong
    to each other, their empirical cumulative distribution functions (ECDFs) must be quite similar.
    The discrete, 2-sample KS test evaluates distributions' similarity by measuring the differences between the ECDFs.

    We use scipy's two-sided KS: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html

    Note that the alternative hypotheses describe the CDFs of the underlying distributions, not the observed values.
    For example, suppose x1 ~ F and x2 ~ G. If F(x) > G(x) for all x, the values in x1 tend to be less than those in x2.
    If the KS statistic is small or the p-value is high, then we cannot reject the null hypothesis in favor of the alternative.

    :param data_1: one set of data
    :param data_2: a second set of data
    :param alternative: ‘two-sided’, ‘less’, ‘greater’
    :return:
    """
    ks_statistic, pval = stats.ks_2samp(data1=data_1, data2=data_2, alternative=alternative)
    #print(f"KS statistic: {ks_statistic}, p={pval}")  # debugging
    return ks_statistic, pval


def shuffle_groups(group1, group2):
    df1 = pd.DataFrame({"data": group1, "group": [1] * len(group1)})
    df2 = pd.DataFrame({"data": group2, "group": [2] * len(group2)})
    df = pd.concat([df1, df2], axis=0).reset_index(drop=True, inplace=False)
    shuffle(df.loc[:, "group"])  # in place: https://docs.python.org/3/library/random.html#random.shuffle
    group1_new = df[df["group"] == 1]["data"].tolist()
    group2_new = df[df["group"] == 2]["data"].tolist()
    return group1_new, group2_new


def ks_test_permutation(group1, group2, name, save_path, num_perms=1000):
    result = list()
    result_cols = ["iteration", "KS", "p-value"]
    original_ks, original_pval = ks_test(data_1=group1, data_2=group2, alternative="two-sided")
    result.append(["original", original_ks, original_pval])

    # permutation test: create distribution
    # start tracking progress
    for i in tqdm(range(num_perms)):
        group1, group2 = shuffle_groups(group1, group2)
        ks_statistic, pval = ks_test(data_1=group1, data_2=group2, alternative="two-sided")
        result.append([i, ks_statistic, pval])
    result_df = pd.DataFrame(result, columns=result_cols)
    result_df.to_csv(os.path.join(save_path, f"{name.replace(' ', '')}.csv"))

    # plot the distribution
    plt.clf()
    sns.set_style("whitegrid")
    sns.histplot(data=result_df["KS"], color="#B0BBBF", stat="frequency", kde=True, line_kws={"color": "#536065"})
    plt.axvline(x=original_ks, color="red")
    plt.title(f"{name.title()} Histogram", fontsize=21)
    plt.savefig(os.path.join(save_path, f"{name.replace(' ', '')}.png"), bbox_inches="tight")
    plt.show()
    return


def plot_cumulative(group1, group1_name, group2, group2_name, name, save_path):
    plt.clf()
    palette = {group1_name: "#24423F", group2_name: "#DB504A"}
    sns.set_style("whitegrid")
    df1 = pd.DataFrame({"data": group1, "group": [group1_name] * len(group1)})
    df2 = pd.DataFrame({"data": group2, "group": [group2_name] * len(group2)})
    df = pd.concat([df1, df2], axis=0).reset_index(drop=True, inplace=False)
    sns.histplot(data=df, x="data", stat="percent", hue="group", cumulative=True, kde=True, palette=palette)
    plt.title(f"{name.title()} Cumulative Histogram")
    plt.savefig(os.path.join(save_path, f"{name.replace(' ', '')}_cumulative.png"), bbox_inches="tight")
    plt.show()
    return


def manage_comparisons(data_path, save_path):

    wordIA_file = pd.read_csv(os.path.join(data_path, f"experiment_IA_word_intactVblurry.csv"))
    imageIA_file = pd.read_csv(os.path.join(data_path, f"experiment_IA_image_intactVblurry.csv"))

    """
    # Two-sample Kolmogorov-Smirnov test for word IA (after independent sample ttest)
    print("----------------KS test for word IA----------------", flush=True)
    wordIA_intact = wordIA_file[wordIA_file["Group"] == "Intact"].loc[:, "word IA"].tolist()
    wordIA_blur = wordIA_file[wordIA_file["Group"] == "Blurry"].loc[:, "word IA"].tolist()
    plot_cumulative(group1=wordIA_intact, group1_name="intact", group2=wordIA_blur, group2_name="blurry",
                    name="word IA", save_path=save_path)
    ks_test_permutation(group1=wordIA_intact, group2=wordIA_blur, name="word IA", save_path=save_path)

    # Two-sample Kolmogorov-Smirnov test for image IA (after paired sample ttest)
    print("----------------KS test for image IA----------------", flush=True)
    imageIA_intact = imageIA_file.loc[:, "image IA intact"].tolist()
    imageIA_blur = imageIA_file.loc[:, "image IA blurry"].tolist()
    plot_cumulative(group1=imageIA_intact, group1_name="intact", group2=imageIA_blur, group2_name="blurry",
                    name="image IA", save_path=save_path)
    ks_test_permutation(group1=imageIA_intact, group2=imageIA_blur, name="image IA", save_path=save_path)
    """

    # plot
    import ptitprince as pt
    pal = sns.color_palette(["#003459", "#007EA7"])

    # word IA raincloud
    import plotter
    wordIA_intact = wordIA_file[wordIA_file["Group"] == "Intact"]
    wordIA_intact = wordIA_intact.rename(columns={"word IA": "word IA Intact"})
    intact_list = wordIA_intact["word IA Intact"].tolist()
    wordIA_blurry = wordIA_file[wordIA_file["Group"] != "Intact"]
    wordIA_blurry = wordIA_blurry.rename(columns={"word IA": "word IA Blurry"})
    blurry_list = wordIA_blurry["word IA Blurry"].tolist()
    pad_list = [np.nan for i in range(len(intact_list)-len(blurry_list))]
    blurry_list.extend(pad_list)
    wordIA_new = pd.DataFrame({"word IA Intact": intact_list, "word IA Blurry": blurry_list})

    plt.clf()
    plt.figure()
    plotter.plot_raincloud_from_df_new(df=wordIA_new, col1="word IA Intact", col1_name="Intact", col1_color="#003459",
                                       col2="word IA Blurry", col2_name="Blurry", col2_color="#007EA7",
                                       title="Word IA in Intact and Blurry Images", x_name="Group", y_name="Word IA",
                                       save_path=save_path, save_name="wordIA_comp", alpha_1=0.85, alpha_2=0.85,
                                       min=0.75, max=1.04, interval=0.05, lines=False)
    """
    pt.half_violinplot(x="word IA", y="Group", data=wordIA_file, palette=pal, bw=.2, cut=0.,
                       scale="area", width=.6, inner=None, alpha=0.85)
    sns.stripplot(x="word IA", y="Group", data=wordIA_file, palette=pal, edgecolor="white",
                       size=2, jitter=1, zorder=0, alpha=0.55)
    sns.boxplot(x="word IA", y="Group", data=wordIA_file, color="black", linewidth=2, width=.2, zorder=10,
                showcaps=True, boxprops={'facecolor': 'none', "zorder": 10},
                showfliers=False, whiskerprops={'linewidth': 2, "zorder": 10},
                saturation=1)
    plt.title("Word IA in Intact and Blurry Images", fontsize=20)
    plt.xlabel("Word IA", fontsize=15)
    plt.ylabel("Group", fontsize=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(10, 6)
    plt.savefig(os.path.join(save_path, f"wordIA_comp.png"), bbox_inches="tight")
    """

    # image IA raincloud
    plt.clf()
    plt.figure()
    plotter.plot_raincloud_from_df_new(df=imageIA_file, col1="image IA intact", col1_name="Intact", col1_color="#003459",
                                   col2="image IA blurry", col2_name="Blurry", col2_color="#007EA7",
                                   title="Image IA in Intact and Blurry Images", x_name="Group", y_name="Image IA",
                                   save_path=save_path, save_name="imageIA_comp", alpha_1=0.85, alpha_2=0.85,
                                   min=0.75, max=1.04, interval=0.05, lines=True)


    # word freq count
    plt.clf()
    plt.figure()
    freq_file = pd.read_csv(os.path.join(data_path, f"word_frequency_count_intactVblurry.csv"))
    sns.set_theme(style="white")
    freq_file_nonan = freq_file[freq_file["word IA"].notna()].reset_index(drop=True, inplace=False)
    freq_file_nonan_intact = freq_file_nonan[freq_file_nonan["Group"] == "Intact"]
    freq_file_nonan_blurry = freq_file_nonan[freq_file_nonan["Group"] != "Intact"]
    sns.scatterplot(data=freq_file_nonan, x="word IA", y="word freq in English language", palette=pal, hue="Group")
    # add regression line
    sns.regplot(data=freq_file_nonan_intact, x="word IA", y="word freq in English language", color="#003459",
                scatter=False)
    sns.regplot(data=freq_file_nonan_blurry, x="word IA", y="word freq in English language", color="#007EA7",
                scatter=False)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title("Word Frequency in English by Word IA", fontsize=21)
    plt.xlabel("Word IA", fontsize=19)
    plt.ylabel("Frequency in English", fontsize=19)
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(8, 6)
    plt.savefig(os.path.join(save_path, f"word_freq_IA.png"), DPI=1000)
    return


def calculate_image_aucs_with_rare(experiment_df, save_path, load=False):
    import pickle
    import analyze_data
    import process_raw_data
    from collections import Counter
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
        image_dict = analyze_data.table_per_image(experiment_df)
        analyze_data.image_stats(image_dict, save_path)
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
                for col in process_raw_data.WORDS:  # iterate word columns
                    word = row[col]
                    ### INITIALIZE THE CUMULATIVE PERCENTAGES FOR AUC CALCULATION AT THE IMAGE+WORD LEVEL
                    image_cnt_for_cumul_pcntg = {p: 0 for p in [i / (process_raw_data.STIM_REPS - 1) for i in
                                                                range(process_raw_data.STIM_REPS)]}
                    other_images_cnt_for_cumul_pcntg = {p: 0 for p in [i / (process_raw_data.STIM_REPS - 1) for i in
                                                                       range(process_raw_data.STIM_REPS)]}
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
                        #continue # ***THIS IS COMMENTED OUT SUCH THAT RARE WORDS GET AUC AND IA AS WELL!!!***

                    # Otherwise: for each other image, count the number of participants who reported this word:
                    # cnt for current image:
                    pcnt_other_subs = count_other_subs / (process_raw_data.STIM_REPS - 1)
                    image_cnt_for_cumul_pcntg[pcnt_other_subs] = 1
                    # cnt for other images:
                    other_images = [im for im in image_list if im != image]
                    other_image_cnts = list()  # for each image, count how many subjects reported "word" for that image
                    for other_image in other_images:
                        other_image_df = image_dict[other_image]
                        count_other_image = analyze_data.count_word_resps(other_image_df, word, ind)
                        other_image_cnts.append(count_other_image)
                    other_image_pcntgs = [cnt / (process_raw_data.STIM_REPS - 1) for cnt in other_image_cnts]
                    other_image_pcntgs_cnt = Counter(other_image_pcntgs)
                    for pcnt in other_image_pcntgs_cnt:  # have for each % the number of images that had this % reported
                        other_images_cnt_for_cumul_pcntg[pcnt] = other_image_pcntgs_cnt[pcnt]

                    # Cumulative % for within-image and between-images word appearance
                    image_cumul_pcntg = analyze_data.calc_cumul_pcntg(image_cnt_for_cumul_pcntg)  # within
                    other_images_cumul_pcntg = analyze_data.calc_cumul_pcntg(other_images_cnt_for_cumul_pcntg)  # between

                    # ROC curve: y=TPR, x = FPR, in Chuyin: y=within-image, x=between-image
                    df = pd.DataFrame({"within image": image_cumul_pcntg, "other images": other_images_cumul_pcntg})
                    # plt.plot(df["other images"], df["within image"]) for debugging, shows the ROC
                    fpr, tpr = df["other images"].tolist()[::-1], df["within image"].tolist()[
                                                                  ::-1]  # -1 to have the lists in ascending order
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
                    # x_axis, y_axis = corrected_AUC_calculation(fpr, tpr)  # Read the above comment
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
                # images_aucs_dict[image] = np.nan

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
        word_count_df.to_csv(os.path.join(save_path, word_count_file_name + ".csv"))
        fl = open(os.path.join(save_path, word_count_file_name + ".pickle"), 'ab')
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


def alternative_IA_calculation(data_path, save_path):
    import analyze_data
    print("---PREPARE DATA FOR ANALYSIS---")
    experiment_df = pd.read_csv(os.path.join(data_path, "experiment_df_words_lemmatized.csv"))
    cols = experiment_df.columns.tolist()
    start_ind = cols.index("participant")
    experiment_df = experiment_df[cols[start_ind:]]  # first column
    experiment_df.replace(np.nan, '', inplace=True)  # replace nans with 0
    experiment_df = analyze_data.assert_no_duplicate_resps_within_subject(experiment_df)
    analyze_data.count_unique_words(experiment_df)
    analyze_data.count_no_response_trials(experiment_df)
    print("---DATABASE IS READY: LET THE ANALYSIS BEGIN!---")
    images_aucs_dict, rare_word_count, word_count_dict, image_only_rare_words_df = calculate_image_aucs_with_rare(experiment_df,
                                                                                                        save_path,
                                                                                                        load=False)  # ORIGINAL AUC CALCULATION
    word_IA_dict = analyze_data.calculate_word_IAs(images_aucs_dict, save_path)
    image_IA_dict = analyze_data.calculate_image_IAs(word_IA_dict, word_count_dict, save_path)
    analyze_data.confidence_analysis(experiment_df, word_IA_dict, save_path, load=False)
    return


if __name__ == "__main__":
    manage_comparisons(data_path=r"E:\Richness\Chuyin\01 replication and blurry\Data\Comparison\0_original",
                       save_path=r"E:\Richness\Chuyin\01 replication and blurry\Data\Comparison\0_original")
    #manage_comparisons(data_path=r"D:\Richness\Chuyin\01 replication and blurry\Data\Comparison\2_filtered",
    #                   save_path=r"D:\Richness\Chuyin\01 replication and blurry\Data\Comparison\2_filtered")
    #manage_comparisons(data_path=r"D:\Richness\Chuyin\01 replication and blurry\Data\Comparison\filtered",
    #                   save_path=r"D:\Richness\Chuyin\01 replication and blurry\Data\Comparison\filtered")
    # ROUND 1
    #alternative_IA_calculation(data_path=r"D:\Richness\Chuyin\01 replication and blurry\Data\Round 1 intact\PerceptionExperiment_2022-03-11b\processed\0_analysis_original",
    #                           save_path=r"D:\Richness\Chuyin\01 replication and blurry\Data\Round 1 intact\PerceptionExperiment_2022-03-11b\processed\analysis_with_rare")
    # ROUND 2
    #alternative_IA_calculation(data_path=r"D:\Richness\Chuyin\01 replication and blurry\Data\Round 2 blur\PerceptionExperiment_2022-04-19\processed\0_analysis_original",
    #                           save_path=r"D:\Richness\Chuyin\01 replication and blurry\Data\Round 2 blur\PerceptionExperiment_2022-04-19\processed\analysis_with_rare")