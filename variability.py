import pandas as pd
import numpy as np


def compute_variability_statistics():
    data_dict = pd.read_excel("data_v2.xlsx", None)
    output = []

    for topic in range(1, 23):
        tab_name = 'topic' + str(topic)
        df = data_dict[tab_name]

        for rubric in range(1, 9):
            rubric_grades = df['Grade'+ str(rubric)].tolist()
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


def read_rubric_variability_statistics():
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


def read_topic_variability_statistics():
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
        # So this value above is the average standard deviation for each rubric.

        # Let's do the same for variance, in case we want to use that instead of std.
        variance_list = topic_selection['variance'].tolist()
        mean_variance = np.mean(variance_list)
        # So this value above is the average variance for each rubric.

        # Now let's save all this information per rubric in a dictionary.
        rubric_variablility = {
                        "topic": topic,
                        "total_mean": total_mean,
                        "mean_std": mean_std,
                        "mean_variance": mean_variance}
        output.append(rubric_variablility)

    return output


def save_statistics_excel():
    out = compute_variability_statistics()
    df = pd.DataFrame.from_dict(out)
    df.to_excel('variability_statistics.xlsx')


def save_rubric_variability_excel():
    out = read_rubric_variability_statistics()
    df = pd.DataFrame.from_dict(out)
    df.to_excel('rubric_variability.xlsx')


def save_topic_variability_excel():
    out = read_topic_variability_statistics()
    df = pd.DataFrame.from_dict(out)
    df.to_excel('topic_variability.xlsx')


# MAIN
# save_statistics_excel()
# save_topic_variability_excel()