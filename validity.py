from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

from pre_processing import get_true_grade_sets, get_review_amount_list


# Validity can be computed with the Pearson product moment correlation
# It is based on the assumption that the data is normally distributed.
# Will be in range [-1, 1] with 0 being no correlation at all.


def pearson_per_student_formatted():
    pearson_p_values = list(compute_pearson_per_student().values())
    pearson_values = np.array(list(zip(*pearson_p_values))[0])
    return pearson_values


# Compute the pearson correlation coefficient per student, and adds (Nan, Nan) to tuple if student did not write reviews
def compute_pearson_per_student() -> Dict[int, Tuple[float, float]]:
    data_dict = pd.read_excel("data_v2.xlsx", None)
    output = {}
    teacher_grades_per_topic = get_true_grade_sets("data_v2.xlsx")
    reviews_per_student = get_review_amount_list(data_dict)

    for student in range(1, 45):
        teacher_array_total = np.array([])
        correlations_per_student = np.array([])
        review_array = np.array([])

        # If the student did not write any reviews, the correlation is Nan with p value Nan
        # This is only the case for student 18.
        if not reviews_per_student[student]:
            correlation = np.array((np.NaN, np.NaN))
        else:
            for topic in reviews_per_student[student]:
                tab_name = 'topic' + str(topic)
                df = data_dict[tab_name]
                student_row = df.loc[df['User'] == student]

                teacher_array_single = teacher_grades_per_topic[topic - 1]
                teacher_array_total = np.concatenate((teacher_array_total, teacher_array_single))

                for rubric in range(1, 9):
                    grade_array = student_row['Grade' + str(rubric)]
                    review_array = np.concatenate((review_array, grade_array))

            correlations_per_student = np.concatenate((correlations_per_student, review_array))
            correlation = np.array(scipy.stats.pearsonr(correlations_per_student, teacher_array_total))

        output[student] = correlation

    return output


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


# Plot a bar graph with the validity per topic
def plot_topic_validity() -> None:
    data = compute_pearson_per_topic()

    topic_correlation_values = np.array([])
    for topic in range(1, 23):
        topic_correlation_values = np.concatenate((topic_correlation_values, np.array([data[topic][0]])))


    topics = [i for i in range(1, 22 + 1)]  # 22 = nr of topics, not dynamic yet...
    x_pos = np.arange(len(topics))

    plt.bar(topics, topic_correlation_values)
    plt.title('Bar plot of validity measured with the Pearson Correlation Coefficient')
    plt.xlabel('Topic')
    plt.xticks(x_pos + 1, topics)
    plt.ylabel('Pearson Correlation Coefficient')
    plt.show()


# Plot a bar graph with the validity per student
def plot_student_validity() -> None:
    data = compute_pearson_per_student()

    topic_correlation_values = np.array([])
    for topic in range(1, 23):
        if topic != 18:
            topic_correlation_values = np.concatenate((topic_correlation_values, np.array([data[topic][0]])))
    print(topic_correlation_values)

    topics = [i for i in range(1, 22 + 1)]  # 22 = nr of topics, not dynamic yet...
    # We get rid of 18 because this student did not fill in any reviews.
    # This is not done dynamically, but for now, this is fine.
    topics.pop(17)
    x_pos = np.arange(len(topics))
    print(x_pos)

    print("topics", topics)
    print("topic_correlation_values", topic_correlation_values)
    plt.bar(x_pos + 1, topic_correlation_values)
    plt.title('Bar plot of validity measured with the Pearson Correlation Coefficient')
    plt.xlabel('Student')
    plt.xticks(x_pos + 1, topics)
    plt.ylabel('Pearson Correlation Coefficient')
    plt.show()

# MAIN
# plot_topic_validity()
# plot_student_validity()
