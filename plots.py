import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

from pre_processing import get_topics_names_dict, get_theme_names_dict

from accuracy import get_accuracy, accuracy_per_topic
from systematic_deviation import sys_high_low_official, sys_spread_official, sys_dev_ordering
from variability import read_topic_variability_statistics

INPUT_PATH = "data_v2.xlsx"
NR_REVIEWERS = 44
NR_TOPICS = 22
NR_TOPICS_PER_THEME = (6, 3, 4, 3, 6)

# change colors and run the line below to change the colors used in the plot
COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink', 'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold']
mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color', COLORS)


def plot_per_reviewer(metric):
    y_label = metric.capitalize()
    if metric == "inaccuracy":
        out = get_accuracy(INPUT_PATH)
    elif metric == "validity":
        print("Metrics " + metric + " has not been implemented yet...")
        return
    elif metric == "reliability":
        print("Metrics " + metric + " has not been implemented yet...")
        return
    elif metric == "systematic high/low peer bias":
        out = sys_high_low_official()
    elif metric == "systematic broad/narrow peer bias":
        out = sys_spread_official()
    elif metric == "systematic problems in ordering":
        out = sys_dev_ordering()
        y_label = "Rank correlation"  # atm with average ordering
    else:
        print("Metrics " + metric + " was not recognized...")
        return

    assert out.ndim == 1
    assert len(out) == NR_REVIEWERS

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


