import config
import numpy as np
import pandas as pd
from scipy.stats.morestats import Std_dev
from pre_processing import get_reviewer_grade_sets
import xlwt
from xlwt import Workbook


def generate_data(reviewer_count):
    prep = gen_preparation()
    means = prep[0]
    stdev = prep[1]

    wb = Workbook()
    for topic in range(0,22):
        sheet1 = wb.add_sheet("topic" + str(topic+1), cell_overwrite_ok=True)
        for rubric in range(0,8):
            sheet1.write(0, 0, "User")
            sheet1.write(0, rubric + 1, "Grade" + str(rubric+1))
            rand_grade = np.random.normal(means[topic,rubric], stdev[topic,rubric], reviewer_count)
            #rand_grade = max(0, min(10, np.round(rand_grade))) 
            i = 0
            for value in rand_grade:
                rand_grade[i] = max(0, min(10, np.round(value))) 
                sheet1.write(i+1, 0, i + 1)
                sheet1.write(i+1, rubric + 1, rand_grade[i])
                i += 1

            

            #print(rand_grade)
    wb.save("data_generated.xls")

def gen_preparation():
    data_dict = get_reviewer_grade_sets("data_v2.xlsx")

    means = np.zeros(shape=(22,8))
    stdev = np.zeros(shape=(22,8))
    for topic in range(0,22):
        for rubric in range(0,8):
            means[topic, rubric] = np.nanmean(data_dict[:, topic, rubric])
            stdev[topic, rubric] = np.nanstd(data_dict[:, topic, rubric])
    return (means, stdev)