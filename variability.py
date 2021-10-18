import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compute_variability():
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
                        "q1": np.percentile(rubric_grades, 25),
                        "q3": np.percentile(rubric_grades, 75)
                        }
            output.append(new_dict)

    return output


def save_excel():
    out = compute_variability()
    df = pd.DataFrame.from_dict(out)
    df.to_excel('statistics.xlsx')

save_excel()