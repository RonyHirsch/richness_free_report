import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np

"""
Plotting module.
"""

__author__ = "Rony Hirschhorn"


# font params
TITLE_SIZE = 20
AXIS_SIZE = 25
TICK_SIZE = 20
LABEL_PAD = 8


def plot_corr(df, x_col, x_name, y_col, y_name, color, save_name, save_path, title, xmin=0.7, xmax=1.01, xskip=0.05,
              ymin=1.0, ymax=7.01, yskip=1, marker_size=80, marker_alpha=0.15):
    plt.clf()
    plt.figure()
    sns.set_theme(style="white")

    df_x = df[x_col].tolist()
    df_y = df[y_col].tolist()
    plt.scatter(x=df_x, y=df_y, color=color, marker="o", s=marker_size, alpha=marker_alpha, edgecolor=color)
    # add a correlation line
    m, c = np.polyfit(df_x, df_y, 1)
    x_line = np.array([min(df_x), max(df_x)])
    y_line = m * x_line + c
    plt.plot(x_line, y_line, color=color, linewidth=4, linestyle='-')
    # cosmetics
    plt.xticks(ticks=[x for x in np.arange(xmin, xmax, xskip)], fontsize=TICK_SIZE)
    plt.yticks([y for y in np.arange(ymin, ymax, yskip)], fontsize=TICK_SIZE)
    plt.title(title, fontsize=AXIS_SIZE + 3, pad=LABEL_PAD + 5)
    plt.ylabel(y_name, fontsize=AXIS_SIZE, labelpad=LABEL_PAD)
    plt.xlabel(x_name, fontsize=AXIS_SIZE, labelpad=LABEL_PAD + 5)
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(15, 12)
    plt.savefig(os.path.join(save_path, f"{save_name}.svg"), format="svg", dpi=1000, bbox_inches='tight', pad_inches=0.01)
    del figure
    plt.close()

    return


def plot_raincloud(df, data_col_name, group_col_name, group_order, group_color_dict, save_path,
                   save_name, y_title, x_title, group_name_dict=None, marker_size=50, marker_alpha=0.25,
                   marker_spread=0.2, group_spacing=0.5, violin_width=0.35, violin_alpha=0.65,
                   ymin=0.5, ymax=1.05, yskip=0.1):

    # X axis params
    stim_xs = {item: idx * group_spacing for idx, item in enumerate(group_order)}

    for group in group_order:
        if group_col_name is not None:
            df_group = df[df[group_col_name] == group]
        else:
            df_group = df
        x_loc = stim_xs[group]
        y_vals = df_group[data_col_name]
        # plot violin
        violin = plt.violinplot(y_vals, positions=[x_loc], widths=violin_width, showmeans=True, showextrema=False, showmedians=False)
        # make it a half-violin plot (only to the LEFT of center)
        for b in violin['bodies']:
            # get the center
            m = np.mean(b.get_paths()[0].vertices[:, 0])
            # modify the paths to not go further right than the center
            b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
            b.set_color(group_color_dict[group])
            b.set_alpha(violin_alpha)
            b.set_edgecolor(group_color_dict[group])

        # change the color of the mean lines (showmeans=True)
        violin['cmeans'].set_color("black")
        violin['cmeans'].set_linewidth(4)

        # control the length
        m = np.mean(violin['cmeans'].get_paths()[0].vertices[:, 0])
        violin['cmeans'].get_paths()[0].vertices[:, 0] = np.clip(violin['cmeans'].get_paths()[0].vertices[:, 0], -np.inf, m)

        # then scatter
        scat_x = (np.ones(len(y_vals)) * (x_loc + marker_spread/3.5)) + (np.random.rand(len(y_vals)) * marker_spread)
        plt.scatter(x=scat_x, y=y_vals, marker="o", color=group_color_dict[group], s=marker_size, alpha=marker_alpha, edgecolor=group_color_dict[group])

    # cosmetics
    if group_name_dict:
        plt.xticks(ticks=[(idx * group_spacing) for idx, item in enumerate(group_order)],
                   labels=[group_name_dict[item] for item in group_order], fontsize=TICK_SIZE)
    else:
        plt.xticks(ticks=[(idx * group_spacing) for idx, item in enumerate(group_order)],
                   labels=[item for item in group_order], fontsize=TICK_SIZE)
    plt.yticks([y for y in np.arange(ymin, ymax, yskip)], fontsize=TICK_SIZE)
    plt.title("")
    plt.ylabel(y_title, fontsize=AXIS_SIZE, labelpad=LABEL_PAD)
    plt.xlabel(x_title, fontsize=AXIS_SIZE, labelpad=LABEL_PAD + 5)

    # save plot
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(15, 12)
    plt.savefig(os.path.join(save_path, f"{save_name}.svg"), format="svg", dpi=1000, bbox_inches='tight', pad_inches=0.01)
    del figure
    plt.close()
    return


def plot_raincloud_connected(df, data_col_name, group_col_name, group_order, group_color_dict, save_path,
                   save_name, y_title, x_title, group_name_dict=None, marker_size=50, marker_alpha=0.25,
                   marker_spread=0.2, group_spacing=0.5, violin_width=0.35, violin_alpha=0.65,
                   ymin=0.5, ymax=1.05, yskip=0.1):

    # X axis params
    stim_xs = {item: idx * group_spacing for idx, item in enumerate(group_order)}
    x_list = list()
    y_list = list()
    left_flag = True
    for group in group_order:
        if group_col_name is not None:
            df_group = df[df[group_col_name] == group]
        else:
            df_group = df
        x_loc = stim_xs[group]
        y_vals = df_group[data_col_name].tolist()
        # plot violin
        violin = plt.violinplot(y_vals, positions=[x_loc], widths=violin_width, showmeans=True, showextrema=False, showmedians=False)

        if left_flag:
            # make it a half-violin plot (only to the LEFT of center)
            for b in violin['bodies']:
                # get the center
                m = np.mean(b.get_paths()[0].vertices[:, 0])
                # modify the paths to not go further right than the center
                b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
                b.set_color(group_color_dict[group])
                b.set_alpha(violin_alpha)
                b.set_edgecolor(group_color_dict[group])
        else:
            # make it a half-violin plot (only to the RIGHT of center)
            for b in violin['bodies']:
                # get the center
                m = np.mean(b.get_paths()[0].vertices[:, 0])
                # modify the paths to not go further left than the center
                b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
                b.set_color(group_color_dict[group])
                b.set_alpha(violin_alpha)
                b.set_edgecolor(group_color_dict[group])

        # change the color of the mean lines (showmeans=True)
        violin['cmeans'].set_color("black")
        violin['cmeans'].set_linewidth(4)

        # control the length
        m = np.mean(violin['cmeans'].get_paths()[0].vertices[:, 0])
        violin['cmeans'].get_paths()[0].vertices[:, 0] = np.clip(violin['cmeans'].get_paths()[0].vertices[:, 0], -np.inf, m)

        # then scatter
        if left_flag:
            scat_x = (np.ones(len(y_vals)) * (x_loc + marker_spread/3.5)) + (np.random.rand(len(y_vals)) * marker_spread)
        else:
            scat_x = (np.ones(len(y_vals)) * (x_loc - marker_spread / 3.5)) - (np.random.rand(len(y_vals)) * marker_spread)
        plt.scatter(x=scat_x, y=y_vals, marker="o", color=group_color_dict[group], s=marker_size, alpha=marker_alpha, edgecolor=group_color_dict[group])

        # because we want to add subject lines:
        x_list.append(scat_x)
        y_list.append(y_vals)

        # reset flag:
        left_flag = False

    # Now subject lines
    for sub in range(len(x_list[0])):
        m_list = [x_list[l][sub] for l in range(len(x_list))]
        n_list = [y_list[l][sub] for l in range(len(x_list))]
        plt.plot(m_list, n_list, color="lightgray", linewidth=0.75, alpha=1)

    # cosmetics
    if group_name_dict:
        plt.xticks(ticks=[(idx * group_spacing) for idx, item in enumerate(group_order)],
                   labels=[group_name_dict[item] for item in group_order], fontsize=TICK_SIZE-2)
    else:
        plt.xticks(ticks=[(idx * group_spacing) for idx, item in enumerate(group_order)],
                   labels=[item for item in group_order], fontsize=TICK_SIZE-2)
    plt.yticks([y for y in np.arange(ymin, ymax, yskip)], fontsize=TICK_SIZE-2)
    plt.title("")
    plt.ylabel(y_title, fontsize=AXIS_SIZE-2, labelpad=LABEL_PAD)
    plt.xlabel(x_title, fontsize=AXIS_SIZE-2, labelpad=LABEL_PAD+2)

    # save plot
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(15, 12)
    plt.savefig(os.path.join(save_path, f"{save_name}.svg"), format="svg", dpi=1000, bbox_inches='tight', pad_inches=0.01)
    del figure
    plt.close()
    return

