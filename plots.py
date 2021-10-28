import matplotlib.pyplot as plt
import pandas as pd

from pre_processing import get_topics_names_dict

from accuracy import get_accuracy, accuracy_per_topic
from systematic_deviation import sys_high_low_official, sys_spread_official
from variability import read_topic_variability_statistics

INPUT_PATH = "data_v2.xlsx"
COLOR = 'lightseagreen'
NR_REVIEWERS = 44
NR_TOPICS = 22


def plot_per_reviewer(metric):
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
        print("Metrics " + metric + " has not been implemented yet...")
        return
    else:
        print("Metrics " + metric + " was not recognized...")
        return

    assert out.ndim == 1
    assert len(out) == NR_REVIEWERS

    users = [i for i in range(1, len(out) + 1)]
    plt.bar(users, out, color=COLOR)
    plt.title('Bar plot of ' + metric + ' for each peer reviewer')
    plt.xlabel('ID of peer reviewer')
    plt.ylabel(metric.capitalize())
    plt.grid()
    plt.show()


def plot_per_topic(metric, with_names=True):
    if metric == "variability":
        out = read_topic_variability_statistics()
    elif metric == "inaccuracy over time":
        out = accuracy_per_topic(INPUT_PATH)
    elif metric == "subject knowledge":
        print("Metrics " + metric + " has not been implemented yet in a suitable format...")
        return
    else:
        print("Metrics " + metric + " was not recognized...")
        return

    assert out.ndim == 1
    assert len(out) == NR_TOPICS

    topic_ids = [i for i in range(1, len(out) + 1)]
    if not with_names:
        plt.bar(topic_ids, out, color=COLOR)
    else:
        topic_nrs = pd.DataFrame(topic_ids)  # a dataframe is needed to use the dictionary (df has only 1 column)
        plt.bar(topic_nrs[0].map(get_topics_names_dict()), out, color=COLOR)
        plt.xticks(rotation=90)
    plt.title('Bar plot of ' + metric + ' for each topic')
    plt.xlabel('ID of topic')
    plt.ylabel(metric.capitalize())
    plt.grid()
    plt.show()
