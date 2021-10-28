from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

from pre_processing import get_true_grade_sets, get_review_amount_list


# Validity can be computed with the Pearson product moment correlation
# It is based on the assumption that the data is normally distributed.
# Will be in range [-1, 1] with 0 being no correlation at all.

# THIS IS FUNCTION IS NOT DONE YET
def compute_pearson_per_student():
    data_dict = pd.read_excel("data_v2.xlsx", None)
    output = {}
    teacher_grades_per_topic = get_true_grade_sets("data_v2.xlsx")
    reviews_per_student = get_review_amount_list(data_dict)

    for student in range(1, 45):
        reviewed_topics = reviews_per_student[student]
        print(reviewed_topics)

    #
    # for topic in range(1, 23):
    #     tab_name = 'topic' + str(topic)
    #     df = data_dict[tab_name]
    #
    #     for rubric in range(1, 9):
    #         rubric_grades = df['Grade' + str(rubric)].tolist()
    #         new_dict = {"topic": topic,
    #                     "rubric": rubric,
    #                     "mean": np.mean(rubric_grades),
    #                     "std": np.std(rubric_grades),
    #                     "variance": np.var(rubric_grades),
    #                     "q1": np.percentile(rubric_grades, 25),
    #                     "q3": np.percentile(rubric_grades, 75)
    #                     }
    #         output.append(new_dict)
    #
    # return output


# Compute the validity using the Pearson correlation for each reviewer. Returns a dict with keys being the topic and
# the value being a tuple with the correlation value and the two tailed p value.
def compute_pearson_per_topic() -> Dict[int, Tuple[float, float]]:
    data_dict = pd.read_excel("data_v2.xlsx", None)
    output = {}
    teacher_grades_per_topic = get_true_grade_sets("data_v2.xlsx")

    for topic in range(1, 23):
        tab_name = 'topic' + str(topic)
        df = data_dict[tab_name]

        teacher_array_single = np.array(teacher_grades_per_topic[topic - 1])
        teacher_array_total = np.array([])

        correlations_per_topic = np.array([])
        for index, row in df.iterrows():
            teacher_array_total = np.concatenate((teacher_array_total, teacher_array_single))

            review_array = np.array([])
            for rubric in range(1, 9):
                grade_array = np.array([row['Grade' + str(rubric)]])
                review_array = np.concatenate((review_array, grade_array))

            correlations_per_topic = np.concatenate((correlations_per_topic, review_array))
            if topic == 13 or topic == 18:
                teacher_array_total[0] = teacher_array_total[0] - 0.0001

        correlation = np.array(scipy.stats.pearsonr(correlations_per_topic, teacher_array_total))
        output[topic] = correlation

    return output


def plot_topic_validity() -> None:
    data = compute_pearson_per_topic()

    topic_correlation_values = np.array([])
    for topic in range(1, 23):
        topic_correlation_values = np.concatenate((topic_correlation_values, np.array([data[topic][0]])))

    topics = [i for i in range(1, 22 + 1)]  # 22 = nr of topics, not dynamic yet...
    plt.bar(topics, topic_correlation_values)
    plt.title('Bar plot of validity measured with the Pearson Correlation Coefficient')
    plt.xlabel('Topic')
    plt.ylabel('Pearson Correlation Coefficient')  # see the to do above
    plt.show()

# MAIN
# plot_topic_validity()
