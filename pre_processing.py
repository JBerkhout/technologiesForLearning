import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from typing import Dict, List
#from models import r_count


# Returns an dictionary with the keys being topics and values an array with the students that presented that topic.
def get_topic_presenters_dict() -> Dict[int, List[int]]:
    topic_presenters_dict = {
        1: [8, 41],
        2: [11, 14],
        3: [19, 28],
        4: [21, 26],
        5: [4, 5],
        6: [2, 3],
        7: [32],
        8: [7, 33],
        9: [22],
        10: [12, 36],
        11: [15, 23],
        12: [9, 39],
        13: [6, 31],
        14: [37, 43],
        15: [34, 40],
        16: [29, 38],
        17: [16, 35],
        18: [30],
        19: [1, 13],
        20: [20, 44],
        21: [24, 42],
        22: [10, 25]
    }
    return topic_presenters_dict


# Returns an dictionary with the keys being themes and values a string with the theme name.
def get_theme_names_dict() -> Dict[int, str]:
    theme_names_dict = {
        1: "1 Student Modelling",
        2: "2 Assessment",
        3: "3 Adaptation",
        4: "4 Intelligent Tutoring Systems",
        5: "5 Big Educational Data",
    }
    return theme_names_dict


# Returns an dictionary with the keys being topics and values a string with the topic name.
def get_topics_names_dict() -> Dict[int, str]:
    topics_names_dict = {
        1: "1.1 Modelling affective state",
        2: "1.2 Modelling metacognitive state",
        3: "1.3 Open student modelling",
        4: "1.4 Modelling groups of students",
        5: "1.5 Modelling students in serious games",
        6: "1.6 Modelling context in mobile apps",
        7: "2.1 Psychometrics",
        8: "2.2 Computerised adaptive testing",
        9: "2.3 Adaptive estimation",
        10: "3.1 Tutorial dialog systems",
        11: "3.2 Adaptive educational hypermedia",
        12: "3.3 Adaptive generation and sequencing",
        13: "3.4 Educational recommender systems",
        14: "4.1 Adaptive feedback",
        15: "4.2 Cognitive tutors",
        16: "4.3 Analysis of programming assignments",
        17: "5.1 Detection of important learning events",
        18: "5.2 Optimisation of system behaviour",
        19: "5.3 Layered evaluation of adaptive educational systems",
        20: "5.4 Massive open online courses",
        21: "5.5 Learning dashboards",
        22: "5.6 Early warning systems"
    }
    return topics_names_dict


# Returns an dictionary with the keys being rubric number and values a string with the full rubric descriptions.
def get_long_rubrics_names_dict() -> Dict[int, str]:
    rubrics_names_dict = {
        1: "Overall clarity, good structure and high-level overview",
        2: "Proper aim at the target audience",
        3: "Adequate presentation skills",
        4: "Appropriate, professional language",
        5: "Ability to engage the audience",
        6: "Adequate answers to questions",
        7: "Slides adding value to the presentation",
        8: "Explicit and appealing take-home message",
    }
    return rubrics_names_dict


# Returns an dictionary with the keys being rubric number and values a string with the shorted rubric description.
def get_short_rubrics_names_dict() -> Dict[int, str]:
    rubrics_names_dict = {
        1: "Overall structure",
        2: "Proper aim",
        3: "Presentation skills",
        4: "Language",
        5: "Engaging",
        6: "Giving answers",
        7: "Good slides",
        8: "Take-home message",
    }
    return rubrics_names_dict


# The pre-analysis from the Jupiter Notebook
def pre_analysis() -> None:
    dataDict = pd.read_excel("data_v2.xlsx", None)
    print("Number of topic presentation: ", len(dataDict) - 2)  # not main & topics

    dataDict['main'][dataDict['main'].isna().any(axis=1)]

    no_missing_data_users = []
    for key in dataDict.keys():
        no_missing_data_users = np.concatenate((no_missing_data_users, dataDict[key].iloc[:, 0].values), axis=None)

    unique, frequency = np.unique(no_missing_data_users, return_counts=True)
    arrFormCount = np.asarray((unique, frequency)).T.astype(int)
    dfFormCount = pd.DataFrame(arrFormCount, columns=['User', 'Number of filled forms'])

    # np.average(dfFormCount['Number of filled forms'])

    for key in dataDict.keys():
        # print(key, dataDict[key].shape[0])
        dataDict[key] = dataDict[key][dataDict[key].iloc[:, 0].isin([17, 18, 27]) == False]
        # print(key, dataDict[key].shape[0], "\n")

    dataDict['main'].describe()  # description per grading aspect
    dataDict['main'].iloc[:, 1:].T.describe()  # description per rater
    dataDict['topic1'].describe()

    topics = list(dataDict.keys())[2:]

    # for overview in first scatter plot below:
    topics = topics[:22]  # 5 can be any number from 2 upto 22 (= similar to commenting this line)

    print(topics)

    # try whether it is easy to see outstanding graders
    x = []
    y = []
    nr = 0
    markers = ["o", "v", "^", "<", ">", "s", "p", "P", "*", "X", "D", "d", "+", "x", "1", "2", "3", "4", "|", "_", "8",
               2, 3]

    for topic in topics:
        plt.scatter(dataDict[topic].iloc[:, 0], dataDict[topic].iloc[:, 1], label=topic, marker=markers[nr])
        nr = nr + 1

    plt.rcParams['figure.figsize'] = [10, 10]
    plt.title('Scatter plot of Grade1s for each topic per user')
    plt.xlabel('User')
    plt.ylabel('Grade1')
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.show()

    setup(dataDict)


# Method that calls various sanity checks on the database.
def setup(data_dict: pd.DataFrame) -> None:
    # There are 43 students.
    # Presentation groups are either 1 or 2 students, so there should be either 41 or 42 peer reviews for each
    # presentation. However, some students might not have filled in the review.
    # There should never be more than 42 peer-reviews for a presentation.

    topic_names = get_topics_names_dict()
    topic_presenters = get_topic_presenters_dict()

    # Sanity check: are there self assessors?
    self_assessors = are_there_self_assessors(topic_presenters, data_dict)
    print("Are there any self assessors?", self_assessors)

    # Sanity check: are there duplicate reviews for a presentation?
    are_there_duplicates = are_there_duplicate_reviews(data_dict)
    print("Are there duplicate reviews for any of the presentations?", are_there_duplicates)

    # Sanity check: did all students in the same presentation get grades from the teacher?
    presenters_same_grade = presenters_same_grades(topic_presenters, data_dict)
    print("Do students from the same presentation get the same grades?", presenters_same_grade)

    # Make a list of amount of review per student.
    review_amount_list = get_review_amount_list(data_dict)
    print("Dictionary with keys the students and values lists of what presentations they reviewed:")
    # for i in review_amount_list:
    #    print(str(i), len(review_amount_list[i]))
    print(review_amount_list)


# Checks if all presenters from the same topic get the same grade. Returns True if all presenters from the same topic
# get the same grades.
def presenters_same_grades(topic_presenters: Dict[int, List[int]], data_dict: pd.DataFrame) -> bool:
    for topic in range(1, 23):
        presenters = topic_presenters.get(topic)
        df = data_dict['main']
        str_presenters = []
        for pres in presenters:
            str_presenters.append(str(pres))

        if len(str_presenters) == 2:
            presentation_grades = df.loc[df['User'].isin(str_presenters)]
            for col_num in list(data_dict['main'].keys())[1:9]:
                if presentation_grades[col_num].is_unique:
                    return True
    return False


# Checks whether there are any students who filled in an review for the presentation they presented themselves.
def are_there_self_assessors(topic_presenters: Dict[int, List[int]], data_dict: pd.DataFrame) -> bool:
    for topic in range(1, 23):
        topic_string = 'topic' + str(topic)
        df = data_dict[topic_string]
        self_assessments = df.loc[df['User'].isin(topic_presenters.get(topic))]
        if not self_assessments.empty:
            return True
        # print("No self assessors for topic", str(topic), "? ", self_assessments.empty)
        # print("Self assessments topic ", str(topic), " = ", df.loc[df['User'].isin(topic_presenters.get(topic))])
    return False


# Checks whether there are any students who filled in multiple reviews for the same presentation.
def are_there_duplicate_reviews(data_dict: pd.DataFrame) -> bool:
    for topic in range(1, 23):
        topic_string = 'topic' + str(topic)
        df = data_dict[topic_string]
        if not df['User'].is_unique:
            return True
    return False


# Returns a dict with keys being the students and values an array with all topics they reviewed.
def get_review_amount_list(data_dict: pd.DataFrame) -> Dict[int, List[int]]:
    review_amounts_dict = {}

    for student in range(1, 45):
        review_amounts_dict[student] = []

    for topic in range(1, 23):
        tab_name = 'topic' + str(topic)
        df = data_dict[tab_name]
        for student in range(1, 45):
            if student in df.loc[:, 'User'].values:
                review_amounts_dict[student].append(topic)
    return review_amounts_dict


# Returns the true grades as given by the teacher.
# Returns an array of lists, where each list contains the grades given on the eight rubrics.
def get_true_grade_sets(input_path: str) -> List[List[float]]:
    # Need to include pre-processing before here!!
    data_dict = pd.read_excel(input_path, None)
    df = data_dict['true_grades']

    j = 0
    true_grades = np.zeros(df.__len__(), dtype=list)
    while j < df.__len__():
        grades = []
        h = 0
        localdf = df[df["User"] == j + 1]
        for rubric in range(1, 9):
            grades.append(localdf['R' + str(rubric)].tolist()[0])
            h += 1
        true_grades[j] = grades
        j += 1
    return true_grades


# Returns a list of arrays of lists contain the grades given by each reviewer for each presentation
# Basically for each reviewer contains a structure similar to true grades
def get_reviewer_grade_sets(input_path: str) -> List[List[List[float]]]:
    
    data_dict = pd.read_excel(input_path, None)
    print(data_dict.__len__())
    # global r_count
    reviewer_grade_sets = np.zeros(shape=(44, 22, 8))  # nr of reviewers, topics, rubrics

    for topic in range(1, 23):
        tab_name = 'topic' + str(topic)
        df = data_dict[tab_name]

        reviewer_nr = 1
        while reviewer_nr <= 44:  # df.__len__():
            grades = []
            localdf = df[df["User"] == reviewer_nr]

            for rubric in range(1, 9):
                grade_to_add = localdf['Grade' + str(rubric)].tolist()

                # Add nan if no grade was assigned for this presentation/topic
                if (grade_to_add.__len__() == 0):
                    grades.append(math.nan)
                else:
                    grades.append(grade_to_add[0])

            reviewer_grade_sets[reviewer_nr - 1][topic - 1] = grades  # bye bye .append
            reviewer_nr += 1

    return reviewer_grade_sets


# MAIN
# pre_analysis()