import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def testprint():
    print("hello world")


def setup():
    dataDict = pd.read_excel("data_v2.xlsx", None)
    dataDict.keys()
    print("Number of topic presentation: ", len(dataDict) - 2)  # not main & topics

    # for key in dataDict.keys():
    #     print(key, dataDict[key].shape[0])
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

setup()