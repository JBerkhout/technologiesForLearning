from numpy.core.numeric import NaN
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

from pre_processing import get_short_rubrics_names_dict, get_topics_names_dict

# Returns a tuple with two arrays containing the high/low bias and the std bias for each reviewer respectively
def compute_systematic_deviation_statistics():
    data_dict = pd.read_excel("data_v2.xlsx", None)

    grades_list = list_grades_per_reviewer(data_dict)
    high_low = sys_high_low(grades_list)
    spread = sys_spread(grades_list)
    
    # print(high_low)
    print(spread)
    return(high_low, spread)

# Calculates the systematic deviation for each reviewer. >0 values means positive bias, <0 value means negative bias
def sys_high_low(grades_list):
    # Calculate the mean grade assigned by each reviewer
    reviewer_grades = grades_list
    reviewer_means = [] # np.zeros(reviewer_grades.__len__())
    # i = 0
    for reviewer in reviewer_grades:
        if(not average(reviewer) == None): # Catch bad guy 18 who never handed in any reviews
            reviewer_means.append(average(reviewer))
    #    i += 1

    # For each reviewer, calculate the difference between their mean and the mean of the other means
    reviewer_bias = np.zeros(reviewer_means.__len__())
    i = 0
    while i < reviewer_bias.__len__():
        # Calculate the individual systematic bias
        reviewer_bias[i] = reviewer_means[i] - average(reviewer_means)
        i += 1
    return(reviewer_bias)

# Calculate the difference in std compared to the average std of the rest of the reviewers
def sys_spread(grades_list):
    # Start by calculating the standard deviation per reviewer
    std_per_reviewer = []
    for reviewer in grades_list:
        if(not reviewer == []): # Catch bad guy 18 who never handed in any reviews
            std_per_reviewer.append(np.std(reviewer, ddof=1))

    # print(std_per_reviewer)

    # For each reviewer, calculate the difference between their standard deviation and the mean of the other standard deviations
    reviewer_range = np.zeros(std_per_reviewer.__len__())
    i = 0
    while i < reviewer_range.__len__():
        # Calculate the systematic range bias
        reviewer_range[i] = std_per_reviewer[i] - average(std_per_reviewer)
        i += 1
    return reviewer_range

def average(list):
    len = list.__len__()
    if(len == 0):
        return None
    return (sum(list) / len)

# Returns an array containing for each reviewer a list with all grades they have given
# If a reviewer never handed in any grades, nan is put in place instead
def list_grades_per_reviewer(data_dict):
    # Create and initialize array of empty lists to store grades in
    reviewer_grades = np.empty(44, dtype=list)
    i = 0
    while i < 44:
        reviewer_grades[i] = []
        i += 1

    # Loop over all sheets
    for topic in range(1, 23):
        tab_name = 'topic' + str(topic)
        df = data_dict[tab_name]

        i = 1

        # Loop over all reviewers
        while i < 45: # 44
            entry = df[df["User"] == i]
            
            # Safeguard to check if the student graded this presentation. If not, move on to the next student
            if(entry.empty):
                i += 1
                continue
            
            # Extract the grades in the form of a list of length 8 contain all grades given by a single user on a single topic
            grades = []
            j = 0
            for rubric in range(1, 9):
                grades.append(entry['Grade' + str(rubric)].tolist()[0])
                j += 1
            reviewer_grades[i - 1] += grades
            i += 1
    # reviewer_grades now contains for each reviewer (index, note that reviewer 1 has index 0) a list of all grades assigned
    return reviewer_grades


def plot_sys_dev_highlow():
    out, _ = compute_systematic_deviation_statistics()
    users = [i for i in range(1, len(out) + 2)]
    bad_user = 18
    users = users[:bad_user - 1] + users[bad_user:]
    plt.bar(users, out)
    plt.title('Bar plot of systematic high/low individual peer bias')
    plt.xlabel('ID of peer reviewer')
    plt.ylabel('Systematic high/low peer bias')
    plt.show()


def plot_sys_dev_broadnarrow():
    _, out = compute_systematic_deviation_statistics()
    users = [i for i in range(1, len(out) + 2)]
    bad_user = 18
    users = users[:bad_user - 1] + users[bad_user:]
    plt.bar(users, out)
    plt.title('Bar plot of systematic broad/narrow individual peer bias')
    plt.xlabel('ID of peer reviewer')
    plt.ylabel('Systematic broad/narrow peer bias')
    plt.show()
