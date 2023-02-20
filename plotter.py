import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches


def plot_raincloud_from_df_new(df, col1, col1_name, col1_color, col2, col2_name, col2_color, title, x_name, y_name,
                           save_path, save_name, alpha_1=0.9, alpha_2=0.5, min=0.75, max=1.04, interval=0.05,
                               lines=True, line_w=0.2, scat_size=3):
    plt.clf()
    plt.figure()
    sns.set_theme(style="white")
    position1 = 0.9
    position2 = 1.5

    labels = list()

    df = df.dropna(inplace=False)  # get rid of nans

    # scatter
    y_list1 = df[col1].tolist()
    scat_x1 = (np.ones(len(y_list1)) * position1) + (np.random.rand(len(y_list1)) * 0.2 / 2.)
    scat_x1 = [x - (0.2 / 4.) for x in scat_x1]

    # column 1
    position1_list = [position1] * len(y_list1)
    violin = plt.violinplot(y_list1, positions=[position1 - (0.2 / 2.)], showmeans=False, showextrema=False, showmedians=False)
    b = violin['bodies'][0]  # single violin = single body
    # set alpha
    b.set_alpha(alpha_1)
    m = np.mean(b.get_paths()[0].vertices[:, 0])  # get the center
    # modify the paths to not go further right than the center
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
    # set the violin color
    b.set_color(col1_color)
    # add violin edges
    b.set_edgecolor("black")
    b.set_linewidth(1)

    # then scatter
    plt.scatter(x=scat_x1, y=y_list1, marker="o", color=col1_color, alpha=1, s=scat_size)

    # complete with a boxplot
    plt.boxplot(y_list1, positions=[position1], notch=False, patch_artist=True,
                boxprops=dict(facecolor="none", linewidth=1.6),
                whiskerprops={'linewidth': 1.6},
                medianprops=dict(color='black', linewidth=1.6), showfliers=False)
    labels.append((mpatches.Patch(color=col1_color), col1_name))

    # column 2
    if len(col2) > 0:
        # scatter
        y_list2 = df[col2].tolist()
        scat_x2 = (np.ones(len(y_list2)) * position2) + (np.random.rand(len(y_list2)) * 0.2 / 2.)
        scat_x2 = [x - (0.2 / 4.) for x in scat_x2]

        if lines:
            xs = [scat_x1, scat_x2]
            ys = [y_list1, y_list2]
            plt.plot(xs, ys, color="darkgray", linewidth=line_w)


        position2_list = [position2] * len(y_list2)
        violin = plt.violinplot(y_list2, positions=[position2 - (0.2 / 2.)], showmeans=False, showextrema=False,
                                showmedians=False)
        b = violin['bodies'][0]  # single violin = single body
        # set alpha
        b.set_alpha(alpha_2)
        m = np.mean(b.get_paths()[0].vertices[:, 0])  # get the center
        # modify the paths to not go further right than the center
        b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
        # set the violin color
        b.set_color(col2_color)
        # add violin edges
        b.set_edgecolor("black")
        b.set_linewidth(1)

        # then scatter
        plt.scatter(x=scat_x2, y=y_list2, marker="o", color=col2_color, alpha=1, s=scat_size)

        # complete with a boxplot
        plt.boxplot(y_list2, positions=[position2], notch=False, patch_artist=True,
                    boxprops=dict(facecolor="none", linewidth=1.6),
                    whiskerprops={'linewidth': 1.6},
                    medianprops=dict(color='black', linewidth=1.6), showfliers=False)
        labels.append((mpatches.Patch(color=col2_color), col2_name))


    # general plotting
    if len(col2) > 0:
        plt.xlim(position1 - 0.4, position2 + 0.18)
        plt.xticks(ticks=[position1 - 0.1, position2 - 0.1], labels=[col1_name, col2_name], fontsize=13)
    else:
        plt.xlim(position1 - 0.4, position1 + 0.2)
        plt.xticks(ticks=[position1 - 0.1], labels=[col1_name], fontsize=13)

    plt.yticks(ticks=np.arange(min, max, interval), fontsize=16)
    plt.title(title, fontsize=21)
    plt.xlabel(x_name, fontsize=19)
    plt.ylabel(y_name, fontsize=19)
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(8, 6)

    plt.savefig(os.path.join(save_path, f"{save_name}.png"), DPI=1000)

    return


def plot_raincloud_from_df(df, col1, col1_name, col1_color, col2, col2_name, col2_color, title, x_name, y_name,
                           save_path, save_name, alpha_1=0.9, alpha_2=0.5, min=0.75, max=1.04, interval=0.05):
    plt.clf()
    plt.figure()
    sns.set_theme(style="white")
    position1 = 1
    position2 = 1.3
    position3 = 1.7
    position4 = 1.7
    labels = list()

    df = df.dropna(inplace=False)  # get rid of nans

    y_list1 = df[col1].tolist()
    scat_x1 = (np.ones(len(y_list1)) * position1) + (np.random.rand(len(y_list1)) * 0.2 / 2.)
    y_list2 = df[col2].tolist()
    scat_x2 = (np.ones(len(y_list2)) * position2) + (np.random.rand(len(y_list2)) * 0.2 / 2.)
    xs = [scat_x1, scat_x2]
    ys = [y_list1, y_list2]
    plt.plot(xs, ys, color="darkgray", linewidth=0.2)


    position1_list = [position1] * len(y_list1)
    violin = plt.violinplot(y_list1, positions=[position3], showmeans=False, showextrema=False, showmedians=False)
    b = violin['bodies'][0]  # single violin = single body
    # set alpha
    b.set_alpha(alpha_1)
    m = np.mean(b.get_paths()[0].vertices[:, 0])  # get the center
    # modify the paths to not go further right than the center
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
    # set the violin color
    b.set_color(col1_color)
    # add violin edges
    b.set_edgecolor("black")
    b.set_linewidth(1)
    # then scatter
    plt.scatter(x=scat_x1, y=y_list1, marker="o", color=col1_color, alpha=1, s=7)
    # complete with a boxplot
    plt.boxplot(y_list1, positions=[position3 + 0.12], notch=False, patch_artist=True, boxprops=dict(facecolor=col1_color, alpha=0.7), medianprops=dict(color='black', linewidth=0.5), showfliers=False)
    labels.append((mpatches.Patch(color=col1_color), col1_name))


    position1_list = [position1] * len(y_list2)
    violin = plt.violinplot(y_list2, positions=[position4], showmeans=False, showextrema=False, showmedians=False)
    b = violin['bodies'][0]  # single violin = single body
    # set alpha
    b.set_alpha(alpha_2)
    m = np.mean(b.get_paths()[0].vertices[:, 0])  # get the center
    # modify the paths to not go further right than the center
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
    # set the violin color
    b.set_color(col2_color)
    # add violin edges
    b.set_edgecolor("black")
    b.set_linewidth(1)
    # then scatter
    plt.scatter(x=scat_x2, y=y_list2, marker="o", color=col2_color, alpha=1, s=7)
    # complete with a boxplot
    plt.boxplot(y_list2, positions=[position4 + 0.30], notch=False, patch_artist=True, boxprops=dict(facecolor=col2_color, alpha=0.7), medianprops=dict(color='black', linewidth=0.7), showfliers=False)
    labels.append((mpatches.Patch(color=col2_color), col2_name))

    plt.xlim(0, 2.3)
    plt.xticks(ticks=[], labels=[])
    plt.yticks(ticks=np.arange(min, max, interval), fontsize=16)
    plt.title(title, fontsize=21, pad=5)
    plt.xlabel(x_name, fontsize=19, labelpad=5)
    plt.ylabel(y_name, fontsize=19, labelpad=5 - 1.5)

    markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='') for color in [col1_color, col2_color]]
    plt.legend(markers, [col1_name, col2_name], numpoints=1, prop={'size': 15})

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(8, 6)
    plt.savefig(os.path.join(save_path, f"RAINCLOUD_{save_name}.png"), DPI=1000)

    return


def plot_scatter_from_df(df, x_col, x_col_name, y_col, y_col_name, color, size, title, save_path, save_name,
                         alpha, y_min, y_max, y_interval, x_min, x_max, x_interval):
    plt.clf()
    plt.figure()
    sns.set_theme(style="white")
    # basic scatter plot
    plt.scatter(x=df[x_col], y=df[y_col], marker="o", color=color, alpha=alpha, s=size)
    # obtain m (slope) and b(intercept) of linear regression line
    m, b = np.polyfit(df[x_col], df[y_col], 1)

    # add linear regression line to scatterplot
    plt.plot(df[x_col], m * df[x_col] + b, color=color, linewidth=4)

    plt.xlim(x_min, x_max)
    label_list = [str(x) for x in range(x_min, x_max+1, x_interval)] + [""]
    plt.xticks(ticks=np.arange(x_min, x_max+2, x_interval), labels=label_list, fontsize=16)
    plt.yticks(ticks=np.arange(y_min, y_max, y_interval), fontsize=16)
    plt.title(title, fontsize=21, pad=5)
    plt.xlabel(x_col_name, fontsize=19, labelpad=5)
    plt.ylabel(y_col_name, fontsize=19, labelpad=5-1.5)

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(8, 6)
    plt.savefig(os.path.join(save_path, f"SCATTER_{save_name}.png"), DPI=1000)
    return




#if __name__ == "__main__":
    #df = pd.read_csv(r"C:\Users\ronyhirschhorn\Documents\TAU\Richness\COMPARISON\original_analysis_intact_v_blurred_image_IA.csv")
    #plot_raincloud_from_df(df, col1="image IA original intact", col1_name="Intact", col1_color="#003459",
    #                       col2="image IA original scrambled", col2_name="Blurry", col2_color="#007EA7",
    #                       title="Image IA in Intact and Blurry Images", x_name="Image Version", y_name="Image IA",
    #                       save_path=r"C:\Users\ronyhirschhorn\Documents\TAU\Richness\COMPARISON", save_name="original_analysis_intact_v_blurred_image_IA")