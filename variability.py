from typing import List, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pre_processing import get_short_rubrics_names_dict, get_topics_names_dict


# Compute the variability statistics on all topics and return this in a list of dicts.
def compute_variability_statistics() -> List[Dict[str, float]]:
    data_dict = pd.read_excel("data_v2.xlsx", None)
    output = []

    for topic in range(1, 23):
        tab_name = 'topic' + str(topic)
        df = data_dict[tab_name]

        for rubric in range(1, 9):
            rubric_grades = df['Grade' + str(rubric)].tolist()
            new_dict = {"topic": topic,
                        "rubric": rubric,
                        "mean": np.mean(rubric_grades),
                        "std": np.std(rubric_grades),
                        "variance": np.var(rubric_grades),
                        "q1": np.percentile(rubric_grades, 25),
                        "q3": np.percentile(rubric_grades, 75)
                        }
            output.append(new_dict)

    return output


# Method to analyze the variability statistics. Returns list with dicts.
def read_rubric_variability_statistics() -> List[Dict[str, float]]:
    # Let's infer some information from the statistics.xlsx
    variability_data = pd.read_excel("variability_statistics.xlsx", None)
    df = variability_data['Sheet1']
    output = []

    for rubric in range(1, 9):
        rubric_selection = df.loc[df['rubric'] == rubric]

        # First we compute the total mean value that is given for each rubric.
        means_list = rubric_selection['mean'].tolist()
        total_mean = np.mean(means_list)

        # Now let's see if students agree more with each other (less variability) on certain rubrics.
        std_list = rubric_selection['std'].tolist()
        mean_std = np.mean(std_list)
        # So this value above is the average standard deviation for each rubric.

        # Let's do the same for variance, in case we want to use that instead of std.
        variance_list = rubric_selection['variance'].tolist()
        mean_variance = np.mean(variance_list)
        # So this value above is the average variance for each rubric.

        # Now let's save all this information per rubric in a dictionary.
        rubric_variablility = {
            "rubric": rubric,
            "total_mean": total_mean,
            "mean_std": mean_std,
            "mean_variance": mean_variance}
        output.append(rubric_variablility)

    return output


# Method to analyze the variability statistics. Returns list with dicts.
def read_topic_variability_statistics(correct_format=True):
    # Let's infer some information from the statistics.xlsx
    variability_data = pd.read_excel("variability_statistics.xlsx", None)
    df = variability_data['Sheet1']
    output = []

    for topic in range(1, 23):
        topic_selection = df.loc[df['topic'] == topic]

        # First we compute the total mean value that is given for each topic.
        means_list = topic_selection['mean'].tolist()
        total_mean = np.mean(means_list)

        # Now let's see if students agree more with each other (less variability) on certain topics.
        std_list = topic_selection['std'].tolist()
        mean_std = np.mean(std_list)
        # So this value above is the average standard deviation for each topic.

        # Let's do the same for variance, in case we want to use that instead of std.
        variance_list = topic_selection['variance'].tolist()
        mean_variance = np.mean(variance_list)
        # So this value above is the average variance for each topic.

        # Now let's save all this information per rubric in a dictionary.
        topic_variability = {
            "topic": topic,
            "total_mean": total_mean,
            "mean_std": mean_std,
            "mean_variance": mean_variance}
        output.append(topic_variability)

    if correct_format:
        formatted_output = np.zeros(len(output))
        for i_topic in range(len(output)):
            formatted_output[i_topic] = output[i_topic]["mean_variance"]
        return formatted_output

    return output


# Save statistics to Excel file.
def save_statistics_excel() -> None:
    out = compute_variability_statistics()
    df = pd.DataFrame.from_dict(out)
    df.to_excel('variability_statistics.xlsx')


# Save variability per rubric to Excel file.
def save_rubric_variability_excel() -> None:
    out = read_rubric_variability_statistics()
    df = pd.DataFrame.from_dict(out)
    df.to_excel('rubric_variability.xlsx')


# Save variability per topic to Excel file.
def save_topic_variability_excel() -> None:
    out = read_topic_variability_statistics(False)
    df = pd.DataFrame.from_dict(out)
    df.to_excel('topic_variability.xlsx')


def plot_rubric_variability() -> None:
    out = read_rubric_variability_statistics()
    df = pd.DataFrame.from_dict(out)
    rubric_names = df.rubric.map(get_short_rubrics_names_dict())
    plt.bar(rubric_names, df.mean_variance)
    plt.title('Bar plot of variability per rubric')
    plt.xlabel('Rubric')
    plt.ylabel('Variance')
    plt.show()


def plot_topic_variability() -> None:
    out = read_topic_variability_statistics(False)
    df = pd.DataFrame.from_dict(out)
    topic_names = df.topic.map(get_topics_names_dict())
    plt.bar(topic_names, df.mean_variance)
    plt.title('Bar plot of variability per topic')
    plt.xticks(rotation='vertical')
    plt.xlabel('Topic')
    plt.ylabel('Variance')
    plt.show()


def plot_topic_variability_theme_grouped() -> None:
    out = read_topic_variability_statistics(False)
    df = pd.DataFrame.from_dict(out)
    topic_names = df.topic.map(get_topics_names_dict())
    # topic_categories = [topic_name[0:3] for topic_name in topic_names]
    # topic_category_division = [[topic_cat[0], topic_cat[2]] for topic_cat in topic_categories]

    # (manually) specifying which topics belong to which theme
    y_t1 = df.mean_variance[0:6]
    y_t2 = df.mean_variance[6:9]
    y_t3 = df.mean_variance[9:13]
    y_t4 = df.mean_variance[13:16]
    y_t5 = df.mean_variance[16:22]

    # ensuring correct spacing between topics
    x_t1 = np.arange(len(y_t1))
    x_t2 = 1 + np.arange(len(y_t2)) + len(y_t1)
    x_t3 = 2 + np.arange(len(y_t3)) + len(y_t1) + len(y_t2)
    x_t4 = 3 + np.arange(len(y_t4)) + len(y_t1) + len(y_t2) + len(y_t3)
    x_t5 = 4 + np.arange(len(y_t5)) + len(y_t1) + len(y_t2) + len(y_t3) + len(y_t4)

    fig, ax = plt.subplots()
    ax.bar(x_t1, y_t1, color='r', label="1 Student Modelling")
    ax.bar(x_t2, y_t2, color='b', label="2 Assessment")
    ax.bar(x_t3, y_t3, color='g', label="3 Adaptation")
    ax.bar(x_t4, y_t4, color='y', label="4 Intelligent Tutoring Systems")
    ax.bar(x_t5, y_t5, color='black', label="5 Big Educational Data")
    ax.set_title('Bar plot of variability per topic grouped per theme')
    ax.set_ylabel('Variance')
    ax.set_xlabel('Topic')
    ax.set_xticks(np.concatenate((x_t1, x_t2, x_t3, x_t4, x_t5)))
    ax.set_xticklabels(topic_names, rotation='vertical')
    ax.legend()
    plt.show()


def plot_topic_variability_day_grouped() -> None:
    out = read_topic_variability_statistics(False)
    df = pd.DataFrame.from_dict(out)
    topic_names = df.topic.map(get_topics_names_dict())

    # (manually) specifying which topics belong to which presentation day
    y_d1 = df.mean_variance[0:3]
    y_d2 = df.mean_variance[3:6]
    y_d3 = df.mean_variance[6:9]
    y_d4 = df.mean_variance[9:13]
    y_d5 = df.mean_variance[13:16]
    y_d6 = df.mean_variance[16:19]
    y_d7 = df.mean_variance[19:22]

    # ensuring correct spacing between topics
    x_t1 = np.arange(len(y_d1))
    x_t2 = 1 + np.arange(len(y_d2)) + len(y_d1)
    x_t3 = 2 + np.arange(len(y_d3)) + len(y_d1) + len(y_d2)
    x_t4 = 3 + np.arange(len(y_d4)) + len(y_d1) + len(y_d2) + len(y_d3)
    x_t5 = 4 + np.arange(len(y_d5)) + len(y_d1) + len(y_d2) + len(y_d3) + len(y_d4)
    x_t6 = 5 + np.arange(len(y_d6)) + len(y_d1) + len(y_d2) + len(y_d3) + len(y_d4) + len(y_d5)
    x_t7 = 6 + np.arange(len(y_d7)) + len(y_d1) + len(y_d2) + len(y_d3) + len(y_d4) + len(y_d5) + len(y_d7)

    fig, ax = plt.subplots()
    ax.bar(x_t1, y_d1, color='r', label="Day 1")
    ax.bar(x_t2, y_d2, color='b', label="Day 2")
    ax.bar(x_t3, y_d3, color='g', label="Day 3")
    ax.bar(x_t4, y_d4, color='y', label="Day 4")
    ax.bar(x_t5, y_d5, color='black', label="Day 5")
    ax.bar(x_t6, y_d6, color='purple', label="Day 6")
    ax.bar(x_t7, y_d7, color='orange', label="Day 7")
    ax.set_title('Bar plot of variability per topic grouped per presentation day')
    ax.set_ylabel('Variance')
    ax.set_xlabel('Topic')
    ax.set_xticks(np.concatenate((x_t1, x_t2, x_t3, x_t4, x_t5, x_t6, x_t7)))
    ax.set_xticklabels(topic_names, rotation='vertical')
    ax.legend()
    plt.show()


# MAIN
#save_statistics_excel()
#save_topic_variability_excel()
#save_rubric_variability_excel()