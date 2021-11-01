import config
import pandas as pd
import numpy as np
from numpy import ndarray
from pre_processing import get_review_amount_list


# Compute the reliability for each student (being the std of all grades given by a student)
def compute_student_reliability() -> ndarray:
    data_dict = pd.read_excel("data_v2.xlsx", None)
    result_output = []
    reviews_per_student = get_review_amount_list(data_dict)

    for student in range(1, config.r_count+1):
        total_student_array = []

        for topic in reviews_per_student[student]:
            tab_name = 'topic' + str(topic)
            df = data_dict[tab_name]
            student_row = df.loc[df['User'] == student].iloc[0]

            for rubric in range(1, 9):
                column_name = "Grade" + str(rubric)
                grade = student_row[column_name]
                total_student_array.append(grade)

        std = np.std(total_student_array)
        result_output.append(std)

    return np.array(result_output)


# Compute the reliability for each student, for each topic.
def compute_student_topic_reliability() -> ndarray:
    data_dict = pd.read_excel("data_v2.xlsx", None)
    result_output = []
    reviews_per_student = get_review_amount_list(data_dict)

    for student in range(1, config.r_count+1):
        total_student_array = np.zeros(shape=22)

        for topic in reviews_per_student[student]:
            student_for_topic = []
            tab_name = 'topic' + str(topic)
            df = data_dict[tab_name]
            student_row = df.loc[df['User'] == student].iloc[0]

            for rubric in range(1, 9):
                column_name = "Grade" + str(rubric)
                grade = student_row[column_name]
                student_for_topic.append(grade)

            std = np.std(student_for_topic)
            total_student_array[topic-1] = std

        result_output.append(total_student_array)

    return np.array(result_output)



# MAIN
# compute_student_reliability()
# compute_student_topic_reliability()
