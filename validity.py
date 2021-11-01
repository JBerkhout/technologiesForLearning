import math
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

from pre_processing import get_true_grade_sets, get_review_amount_list
from models import r_count

# Validity can be computed with the Pearson product moment correlation
# It is based on the assumption that the data is normally distributed.
# Will be in range [-1, 1] with 0 being no correlation at all.


def pearson_per_student_formatted():
    pearson_p_values = list(compute_pearson_per_student().values())
    pearson_values = np.array(list(zip(*pearson_p_values))[0])
    return pearson_values


# Returns a list of the amount of reviews that are written for each topic.
def amount_of_reviews() -> List[int]:
    data_dict = pd.read_excel("data_v2.xlsx", None)

    review_amount_per_topic = []
    for topic in range(1, 23):
        tab_name = 'topic' + str(topic)
        df = data_dict[tab_name]
        review_amount = len(df["User"].tolist())
        review_amount_per_topic.append(review_amount)
    return review_amount_per_topic



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



# Compute the pearson correlation coefficient per student, and adds (Nan, Nan) to tuple if student did not write reviews
def compute_pearson_per_rubric() -> Dict[int, Tuple[float, float]]:
    data_dict = pd.read_excel("data_v2.xlsx", None)
    output = {}
    review_amounts = amount_of_reviews()
    df = data_dict['true_grades']

    for rubric in range(1, 9):
        total_teacher_array = np.array([])
        rubric_string = "R" + str(rubric)
        rubric_string_student = "Grade" + str(rubric)
        grades_for_rubric = df[rubric_string].tolist()
        total_student_array = np.array([])

        for grade_index in range(1, 23):
            grade = grades_for_rubric[grade_index - 1]
            temp_teacher_arr = [grade] * review_amounts[grade_index -1]
            total_teacher_array = np.concatenate((total_teacher_array, np.array(temp_teacher_arr)))

            dataframe = data_dict["topic" + str(grade_index)]
            student_rubric_list = dataframe[rubric_string_student].tolist()
            total_student_array = np.concatenate((total_student_array, np.array(student_rubric_list)))

        correlation = np.array(scipy.stats.pearsonr(total_teacher_array, total_student_array))
        output[rubric] = correlation

    return output


# could not make use of compute_pearson_per_topic directly, but makes much use of that code
def pearson_per_topic_formatted(with_p=False):
    global r_count
    data_dict = pd.read_excel("data_v2.xlsx", None)
    output = np.zeros(shape=(r_count, 22, 2))
    teacher_grades_per_topic = get_true_grade_sets("data_v2.xlsx")

    for reviewer in range(1, 45):
        for topic in range(1, 23):
            tab_name = 'topic' + str(topic)
            df = data_dict[tab_name]
            student_row = df.loc[df['User'] == reviewer]
            teacher_array = teacher_grades_per_topic[topic - 1]

            # If the student did not write any reviews, the correlation is Nan with p value Nan
            if len(student_row) == 0:
                output[reviewer-1][topic-1][0] = np.nan
                output[reviewer-1][topic-1][1] = np.nan
                continue

            review_array = np.zeros(shape=8)
            for rubric in range(1, 9):
                reviewer_grade = student_row['Grade' + str(rubric)]
                if len(reviewer_grade) != 1:
                    review_array[rubric-1] = np.nan
                else:
                    review_array[rubric-1] = reviewer_grade

            # This might seem odd: but the pearson implementation belong does not work if one of the arrays only
            # has the same value in it. This is why we need to subtract and very small number from the first
            # value of the array to make the pearson computation not just return nan.
            if np.nanstd(teacher_array) == 0:
                teacher_array[0] = teacher_array[0] - 0.0001
            if np.nanstd(review_array) == 0:
                review_array[0] = review_array[0] - 0.0001

            pearson, p = scipy.stats.pearsonr(review_array, teacher_array)
            output[reviewer-1, topic-1, 0] = pearson
            output[reviewer-1, topic-1, 1] = p

    if not with_p:
        return output[:, :, 0]
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
                # This might seem odd: but the pearson implementation belong does not work if one of the arrays only
                # has the same value in it. This is the case for topics 13 and 18, for which the teacher gave the
                # same grades on all the rubrics. This is why we need to subtract and very small number from the first
                # value of the array to make the pearson computation not just return nan.
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
    print(data)

    student_correlation_values = np.array([])
    for student in range(1, 45):
        if student != 18:
            student_correlation_values = np.concatenate((student_correlation_values, np.array([data[student][0]])))
    print(student_correlation_values)

    topics = [i for i in range(1, r_count + 1)]  # 22 = nr of topics, not dynamic yet...
    # We get rid of 18 because this student did not fill in any reviews.
    # This is not done dynamically, but for now, this is fine.
    topics.pop(17)
    x_pos = np.arange(len(topics))
    print(x_pos)

    print("topics", topics)
    print("topic_correlation_values", student_correlation_values)
    plt.bar(x_pos + 1, student_correlation_values)
    plt.title('Bar plot of validity measured with the Pearson Correlation Coefficient')
    plt.xlabel('Student')
    plt.xticks(x_pos + 1, topics)
    plt.ylabel('Pearson Correlation Coefficient')
    plt.show()


# Plot a bar graph with the validity per topic
def plot_rubric_validity() -> None:
    data = compute_pearson_per_rubric()

    rubric_correlation_values = np.array([])
    for rubric in range(1, 9):
        rubric_correlation_values = np.concatenate((rubric_correlation_values, np.array([data[rubric][0]])))

    topics = [i for i in range(1, 8 + 1)]  # 22 = nr of topics, not dynamic yet...
    x_pos = np.arange(len(topics))

    plt.bar(topics, rubric_correlation_values)
    plt.title('Bar plot of validity measured with the Pearson Correlation Coefficient')
    plt.xlabel('Rubric')
    plt.xticks(x_pos + 1, topics)
    plt.ylabel('Pearson Correlation Coefficient')
    plt.show()

# MAIN
# plot_topic_validity()
# plot_student_validity()
# plot_rubric_validity()
