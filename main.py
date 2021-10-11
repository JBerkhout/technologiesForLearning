import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_topic_presenters_dict():
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


def get_topics_names_dict():
    topics_names_dict = {
        1: "1.1 Modelling affective state",
        2: "1.2 Modelling metacognitive state",
        3: "1.3 Open student modelling",
        4: "1.4 Modelling groups of students",
        5: "1.5 Modelling students in serious games",
        6: "1.6 Modelling context in mobile apps",
        7: "2.1 Psychometrics",
        8: "2.2 computerised adaptive testing",
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


def setup():
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

    topic_names = get_topics_names_dict()
    topic_presenters = get_topic_presenters_dict()

    self_assessors = are_there_self_assessors(topic_presenters, dataDict)
    print("Are there any self assessors?", self_assessors)


def are_there_self_assessors(topic_presenters, dataDict):
    for topic in range(1, 23):
        topic_string = 'topic' + str(topic)
        df = dataDict[topic_string]
        self_assessments = df.loc[df['User'].isin(topic_presenters.get(topic))]
        if not self_assessments.empty:
            return True
        #print("No self assessors for topic", str(topic), "? ", self_assessments.empty)
        #print("Self assessments topic ", str(topic), " = ", df.loc[df['User'].isin(topic_presenters.get(topic))])
    return False






setup()
