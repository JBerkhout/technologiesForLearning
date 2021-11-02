import pandas as pd
import numpy as np
from numpy import ndarray

from pre_processing import get_review_amount_list


# Compute the average size of the remarks for each students
def compute_student_remarks() -> ndarray:
    data_dict = pd.read_excel("data_v2.xlsx", None)
    result_output = []
    reviews_per_student = get_review_amount_list(data_dict)

    for student in range(1, 45):
        total_student_str = ""

        for topic in reviews_per_student[student]:
            tab_name = 'topic' + str(topic)
            df = data_dict[tab_name]
            student_row = df.loc[df['User'] == student].iloc[0]

            for rubric in range(1, 9):
                column_name = "Explanation" + str(rubric)
                explanation = str(student_row[column_name])
                total_student_str += explanation

        char_amount = len(total_student_str)
        if char_amount != 0:
            av_char_amount = char_amount/(len(reviews_per_student[student])*8)
        else:
            av_char_amount = 0
        result_output.append(av_char_amount)
    return np.array(result_output)


# Compute the average size of the remarks for each students
def compute_student_topic_remarks() -> ndarray:
    data_dict = pd.read_excel("data_v2.xlsx", None)
    result_output = []
    reviews_per_student = get_review_amount_list(data_dict)

    for student in range(1, 45):
        total_student = []

        for topic in reviews_per_student[student]:
            student_per_topic = ""
            tab_name = 'topic' + str(topic)
            df = data_dict[tab_name]
            student_row = df.loc[df['User'] == student].iloc[0]

            for rubric in range(1, 9):
                column_name = "Explanation" + str(rubric)
                explanation = str(student_row[column_name])
                student_per_topic += explanation

            char_amount = len(student_per_topic)

            if char_amount != 0:
                av_char_amount = char_amount / 8
            else:
                av_char_amount = 0

            total_student.append(av_char_amount)

        result_output.append(total_student)

    return np.array(result_output)




# MAIN
# compute_student_remarks()
# compute_student_topic_remarks()
