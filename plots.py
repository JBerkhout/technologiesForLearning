import config
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr

from pre_processing import get_topics_names_dict, get_theme_names_dict

# from models import r_count
from accuracy import get_accuracy, accuracy_per_topic
from systematic_deviation import sys_high_low_official, sys_spread_official, sys_dev_ordering
from validity import pearson_per_student_formatted
from variability import read_topic_variability_statistics
from reliability import compute_student_reliability

INPUT_PATH = "data_v2.xlsx"
NR_TOPICS = 22
NR_TOPICS_PER_THEME = (6, 3, 4, 3, 6)

# change colors and run the line below to change the colors used in the plot
COLORS = ['#ff1969', '#ffbd59', '#00c2cb', '#3788d4', '#044aad', '#000000']
mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color', COLORS)


def plot_per_reviewer(metric):
    y_label = metric.capitalize()
    if metric == "inaccuracy":
        out = get_accuracy(INPUT_PATH)
    elif metric == "validity":
        out = pearson_per_student_formatted(INPUT_PATH)
    elif metric == "reliability":
        out = compute_student_reliability(INPUT_PATH)
    elif metric == "systematic high/low peer bias":
        out = sys_high_low_official(INPUT_PATH)
    elif metric == "systematic broad/narrow peer bias":
        out = sys_spread_official(INPUT_PATH)
    elif metric == "systematic problems in ordering":
        out = sys_dev_ordering(INPUT_PATH)
        y_label = "Rank correlation"  # atm with average ordering
    else:
        print("Metrics " + metric + " was not recognized...")
        return

    assert out.ndim == 1
    assert len(out) == config.r_count

    users = [i for i in range(1, len(out) + 1)]
    plt.bar(users, out)  # , color=COLOR)
    plt.title('Bar plot of ' + metric + ' for each peer reviewer')
    plt.xlabel('ID of peer reviewer')
    plt.ylabel(y_label)
    plt.grid()
    plt.show()


def plot_per_topic(metric, with_names=True, grouped="not"):
    if metric == "variability":
        y_values = read_topic_variability_statistics()
    elif metric == "inaccuracy over time":
        y_values = accuracy_per_topic(INPUT_PATH)
    elif metric == "subject knowledge":
        print("Metrics " + metric + " has not been implemented yet in a suitable format...")
        return
    else:
        print("Metrics " + metric + " was not recognized...")
        return

    assert y_values.ndim == 1
    assert len(y_values) == NR_TOPICS

    topic_ids = [i for i in range(1, len(y_values) + 1)]
    if not with_names:
        x_values = topic_ids
    else:
        topic_nrs = pd.DataFrame(topic_ids)  # a dataframe is needed to use the dictionary (df has only 1 column)
        x_values = topic_nrs[0].map(get_topics_names_dict())

    if grouped == "not":
        plt.bar(x_values, y_values)
        if with_names:
            plt.xticks(rotation=90)

    elif grouped == "theme":
        # specifying which topics belong to which theme, and ensuring corresponding good spacing
        x_ticks = np.array([])
        fig, ax = plt.subplots()
        for theme_nr in range(len(NR_TOPICS_PER_THEME)):
            nr_previous = sum(NR_TOPICS_PER_THEME[:theme_nr])
            cur_y = y_values[nr_previous:nr_previous+NR_TOPICS_PER_THEME[theme_nr]]
            cur_x = theme_nr + np.arange(NR_TOPICS_PER_THEME[theme_nr]) + nr_previous
            print(x_ticks)
            print(cur_x)
            x_ticks = np.append(x_ticks, cur_x)
            ax.bar(cur_x, cur_y, label=get_theme_names_dict().get(theme_nr + 1))

        ax.set_xticks(np.array(x_ticks))
        ax.set_xticklabels(x_values, rotation='vertical')
        ax.legend()

    plt.title('Bar plot of ' + metric + ' for each topic')
    plt.xlabel('ID of topic')
    plt.ylabel(metric.capitalize())
    plt.grid()
    plt.show()


def plot_correlation_metrics_with_acc():
    # (in)accuracy used for 'true' grades for quality
    metrics = ["validity", "reliability", "systematic high/low peer bias", "systematic broad/narrow peer bias", "systematic problems in ordering"]
    acc = get_accuracy(INPUT_PATH)

    out = []
    p_values = []
    for metric in metrics:
        if metric == "validity":
            values = pearson_per_student_formatted("data_v2.xlsx")
        elif metric == "reliability":
            values = compute_student_reliability("data_v2.xlsx")
        elif metric == "systematic high/low peer bias":
            values = sys_high_low_official("data_v2.xlsx")
        elif metric == "systematic broad/narrow peer bias":
            values = sys_spread_official("data_v2.xlsx")
        elif metric == "systematic problems in ordering":
            values = sys_dev_ordering("data_v2.xlsx")
        else:
            print("Metrics " + metric + " was not recognized...")
            return

        single_2d_arr = np.vstack([values, acc])
        without_nan = single_2d_arr[:, ~np.any(np.isnan(single_2d_arr), axis=0)]

        corr, p = pearsonr(without_nan[0], without_nan[1])
        out.append(np.abs(corr))
        p_values.append(p)

        corr, p = pearsonr(np.abs(without_nan[0]), without_nan[1])
        out.append(np.abs(corr))
        p_values.append(p)

    metrics_labels = (np.array([[metric.capitalize(), "Absolute " + metric] for metric in metrics])).flatten()
    bars = plt.bar(metrics_labels, out)  # , color=COLOR)
    for bar_id in range(len(bars)):
        plt.text(bars[bar_id].get_x(), bars[bar_id].get_height() + .005, "p-value: " + "{:.2e}".format(p_values[bar_id]))
    plt.title('Bar plot of absolute correlation with accuracy for each metric per reviewer', fontsize=25)
    plt.xlabel('Metric', fontsize=20)
    plt.xticks(rotation=90, fontsize=15)
    plt.ylabel('Absolute correlation', fontsize=20)
    plt.grid()
    plt.show()


def plot_correlation_combined_metrics_with_acc():
    # not yet implemented metrics not yet added     e.g. validity and reliability
    metrics = ["validity and reliability", "systematic deviations"]
    acc = get_accuracy(INPUT_PATH)

    out = []
    p_values = []
    for metric in metrics:
        if metric == "validity and reliability":
            weights = [0.2887592359980593, 0.7098815979902631]
            values = weights[0]*pearson_per_student_formatted("data_v2.xlsx") + weights[1]*compute_student_reliability("data_v2.xlsx")
        elif metric == "systematic deviations":
            weights = [0.8140871113424963, 0.6365323717653268, 0.17101072421329433]  # values of correlations found earlier, not dynamic yet
            values = weights[0]*np.abs(sys_high_low_official("data_v2.xlsx")) + weights[1]*sys_spread_official("data_v2.xlsx") + weights[2]*np.abs(sys_dev_ordering())
        else:
            print("Metrics " + metric + " was not recognized...")
            return

        single_2d_arr = np.vstack([values, acc])
        without_nan = single_2d_arr[:, ~np.any(np.isnan(single_2d_arr), axis=0)]

        corr, p = pearsonr(without_nan[0], without_nan[1])
        out.append(np.abs(corr))
        p_values.append(p)

    bars = plt.bar([metric.capitalize() for metric in metrics], out)  # , color=COLOR)
    for bar_id in range(len(bars)):
        plt.text(bars[bar_id].get_x(), bars[bar_id].get_height() + .005, "p-value: " + "{:.2e}".format(p_values[bar_id]))
    plt.title('Bar plot of absolute correlation with accuracy for combined metrics per reviewer')
    plt.xlabel('Metric')
    plt.ylabel('Absolute correlation')
    plt.grid()
    plt.show()
