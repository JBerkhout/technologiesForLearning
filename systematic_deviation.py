import config
from typing import Tuple, List
import pandas as pd
import numpy as np
import math
import warnings
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# from models import r_count
from accuracy import get_reviewer_grade_sets
from variability import compute_variability_statistics, read_topic_variability_statistics


# Returns a tuple with two arrays containing the high/low bias and the std bias for each reviewer respectively
def compute_systematic_deviation_statistics(input_path) -> Tuple[List[float], List[float]]:
    data_dict = pd.read_excel(input_path, None)

    grades_list = list_grades_per_reviewer(data_dict)
    high_low = sys_high_low_official(input_path)  # sys_high_low(grades_list)
    spread =  sys_spread_official(input_path) # sys_spread(grades_list)
    
    #print(high_low)
    #print(spread)
    return high_low, spread


# Calculates the systematic deviation for each reviewer. >0 values means positive bias, <0 value means negative bias
def sys_high_low(grades_list):
    # Calculate the mean grade assigned by each reviewer
    reviewer_grades = grades_list
    reviewer_means = [] # np.zeros(reviewer_grades.__len__())
    # i = 0
    for reviewer in reviewer_grades:
        if average(reviewer) is not None: # Catch bad guy 18 who never handed in any reviews
            reviewer_means.append(average(reviewer))
    #    i += 1

    # For each reviewer, calculate the difference between their mean and the mean of the other means
    reviewer_bias = np.zeros(len(reviewer_means))
    i = 0
    while i < len(reviewer_bias):
        # Calculate the individual systematic bias
        reviewer_bias[i] = reviewer_means[i] - average(reviewer_means)
        i += 1
    return reviewer_bias


# exactly as sys_high_low, but in different programming style
def sys_high_low2(grades_list):
    # Calculate the mean grade assigned by each reviewer
    reviewers_grades = grades_list # 2D array with for each reviewer all their grades (both topics and rubrics)
    reviewers_means = []

    for reviewer_grades in reviewers_grades:
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                cur_mean = np.nanmean(reviewer_grades)
            except RuntimeWarning:
                cur_mean = np.NaN
        reviewers_means.append(cur_mean)

    # For each reviewer, calculate the difference between their mean and the mean of the means
    overall_reviewer_mean = np.nanmean(reviewers_means)
    reviewer_bias = reviewers_means - overall_reviewer_mean

    return reviewer_bias


# with correct formula for sys_dev
def sys_high_low_official(input_path, split_topics=False):
    # global r_count
    reviewer_grade_sets = get_reviewer_grade_sets(input_path)  # shape(reviewers, topics, rubrics)
    topics_means_list = read_topic_variability_statistics(False)

    # get reviewers' grades mean for each topic
    topic_means = np.zeros(len(topics_means_list))
    for i_topic in range(0, len(topics_means_list)):
        topic_means[i_topic] = topics_means_list[i_topic]["total_mean"]

    # # calculate reviewers' means for each topic
    means_topics_reviewers = []
    for reviewer_grades in reviewer_grade_sets:
        means_topics = []
        for topic_grades in reviewer_grades:
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    cur_mean = np.nanmean(topic_grades)
                except RuntimeWarning:
                    cur_mean = np.NaN
            means_topics.append(cur_mean)
        means_topics_reviewers.append(means_topics)

    # calculate mean of the difference between earlier calculated means
    total_topic_sys_dev = np.zeros(shape=(config.r_count, 22))
    for reviewer_id in range(0, len(means_topics_reviewers)):
        total_topic_sys_dev[reviewer_id] = means_topics_reviewers[reviewer_id] - topic_means

    if split_topics:
        return np.array(total_topic_sys_dev)

    reviewer_sys_dev = np.zeros(shape=len(total_topic_sys_dev))
    for i in range(len(total_topic_sys_dev)):
        if np.isnan(total_topic_sys_dev[i]).all():
            reviewer_sys_dev[i] = np.nan
        else:
            reviewer_sys_dev[i] = np.nanmean(total_topic_sys_dev[i])
    return reviewer_sys_dev


def sys_dev_ordering(input_path, split_topics=False):
    reviewer_grade_sets = get_reviewer_grade_sets(input_path)  # shape(reviewers, topics, rubrics)
    topics_means_list = read_topic_variability_statistics(False)
    if split_topics:
        return sys_dev_ordering_topics(reviewer_grade_sets)

    # get reviewers' grades mean for each topic
    topic_means = np.zeros(len(topics_means_list))
    for i_topic in range(0, len(topics_means_list)):
        topic_means[i_topic] = topics_means_list[i_topic]["total_mean"]

    # # calculate reviewers' means for each topic
    means_topics_reviewers = []
    for reviewer_grades in reviewer_grade_sets:
        means_topics = []
        for topic_grades in reviewer_grades:
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    cur_mean = np.nanmean(topic_grades)
                except RuntimeWarning:
                    cur_mean = np.NaN
            means_topics.append(cur_mean)
        means_topics_reviewers.append(means_topics)

    # calculate ranking of topics
    avg_topic_ordering = np.argsort(topic_means)
    reviewers_topic_ordering = []
    for reviewer_means in means_topics_reviewers:
        reviewers_topic_ordering.append(np.argsort(reviewer_means))

    # calculate correlation between each reviewer's and average ranking of topics
    total_sys_dev = np.zeros(shape=config.r_count)
    for reviewer_id in range(0, len(reviewers_topic_ordering)):
        if np.isnan(means_topics_reviewers[reviewer_id]).all():
            total_sys_dev[reviewer_id], p = np.nan, np.nan
        else:
            total_sys_dev[reviewer_id], p = spearmanr(reviewers_topic_ordering[reviewer_id], avg_topic_ordering)

    return total_sys_dev


def sys_dev_ordering_topics(reviewer_grade_sets):
    # calculate correlation between each reviewer's and average ranking of topics
    total_topic_sys_dev = np.zeros(shape=(reviewer_grade_sets.shape[0], reviewer_grade_sets.shape[1]))
    for topic_id in range(0, reviewer_grade_sets.shape[1]):
        # calculate average ranking within topics
        rubric_means = np.zeros(reviewer_grade_sets.shape[2])
        for rubric_id in range(0, reviewer_grade_sets.shape[2]):
            rubric_means[rubric_id] = np.nanmean(reviewer_grade_sets[:, topic_id, rubric_id])
        avg_topic_ordering = np.argsort(rubric_means)

        # calculate correlation between each reviewer's and average ranking of topics
        for reviewer_id in range(0, reviewer_grade_sets.shape[0]):
            reviewers_topic_ordering = np.argsort(reviewer_grade_sets[reviewer_id][topic_id])
            if np.isnan(reviewer_grade_sets[reviewer_id][topic_id]).all():
                total_topic_sys_dev[reviewer_id][topic_id], p = np.nan, np.nan
            else:
                total_topic_sys_dev[reviewer_id][topic_id], p = spearmanr(reviewers_topic_ordering, avg_topic_ordering)

    return total_topic_sys_dev


# Calculate the difference in std compared to the average std of the rest of the reviewers
def sys_spread(grades_list):
    # Start by calculating the standard deviation per reviewer
    std_per_reviewer = []
    for reviewer in grades_list:
        if reviewer is not []: # Catch bad guy 18 who never handed in any reviews
            std_per_reviewer.append(np.std(reviewer, ddof=1))

    # For each reviewer, calculate the difference between their standard deviation and the mean of the other standard
    # deviations
    reviewer_range = np.zeros(std_per_reviewer.__len__())
    i = 0
    while i < reviewer_range.__len__():
        # Calculate the systematic range bias
        reviewer_range[i] = std_per_reviewer[i] - average(std_per_reviewer)
        i += 1
    return reviewer_range


# with correct formula for sys_dev
def sys_spread_official(input_path, split_topics=False):
    reviewer_grade_sets = get_reviewer_grade_sets(input_path)  # shape(reviewers, topics, rubrics)
    topics_means_list = read_topic_variability_statistics(False)

    # get reviewers' grades mean std for each topic
    topic_means = np.zeros(len(topics_means_list))
    for i_topic in range(0, len(topics_means_list)):
        topic_means[i_topic] = topics_means_list[i_topic]["mean_std"]

    # # calculate reviewers' means std for each topic
    means_topics_reviewers = []
    for reviewer_grades in reviewer_grade_sets:
        means_topics = []
        for topic_grades in reviewer_grades:
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    cur_mean = np.nanstd(topic_grades)
                except RuntimeWarning:
                    cur_mean = np.NaN
            means_topics.append(cur_mean)
        means_topics_reviewers.append(means_topics)

    # calculate mean of the difference between earlier calculated means
    total_topic_sys_dev = np.zeros(shape=(config.r_count, 22))
    for reviewer_id in range(0, len(means_topics_reviewers)):
        total_topic_sys_dev[reviewer_id] = means_topics_reviewers[reviewer_id] - topic_means

    if split_topics:
        return np.array(total_topic_sys_dev)

    reviewer_sys_dev = np.zeros(shape=len(total_topic_sys_dev))
    for i in range(len(total_topic_sys_dev)):
        if np.isnan(total_topic_sys_dev[i]).all():
            reviewer_sys_dev[i] = np.nan
        else:
            reviewer_sys_dev[i] = np.nanmean(total_topic_sys_dev[i])
    return reviewer_sys_dev


# Does the same as mean(value_list) but separately checks if list is empty
def average(value_list: List[float]) -> None or float:
    if len(value_list) == 0:
        return None
    return sum(value_list) / len(value_list)


# Returns an array containing for each reviewer a list with all grades they have given
# If a reviewer never handed in any grades, nan is put in place instead
def list_grades_per_reviewer(data_dict: pd.DataFrame):
    # Create and initialize array of empty lists to store grades in
    reviewer_grades = np.empty(config.r_count, dtype=list)
    i = 0
    while i < config.r_count:
        reviewer_grades[i] = []
        i += 1

    # Loop over all sheets
    for topic in range(1, 23):
        tab_name = 'topic' + str(topic)
        df = data_dict[tab_name]

        i = 1

        # Loop over all reviewers
        while i < config.r_count + 1: # 44
            entry = df[df["User"] == i]
            
            # Safeguard to check if the student graded this presentation. If not, move on to the next student
            if entry.empty:
                i += 1
                continue
            
            # Extract the grades in the form of a list of length 8 contain all grades given by a single user on a
            # single topic
            grades = []
            j = 0
            for rubric in range(1, 9):
                grades.append(entry['Grade' + str(rubric)].tolist()[0])
                j += 1
            reviewer_grades[i - 1] += grades
            i += 1
    # reviewer_grades now contains for each reviewer (index, note that reviewer 1 has index 0) a list of all grades
    # assigned
    return reviewer_grades


# Plot systematic deviations
def plot_sys_dev_highlow(input_path) -> None:
    out, _ = compute_systematic_deviation_statistics(input_path)
    users = [i for i in range(1, len(out) + 2)]
    bad_user = 18
    users = users[:bad_user - 1] + users[bad_user:]
    plt.bar(users, out)
    plt.title('Bar plot of systematic high/low individual peer bias')
    plt.xlabel('ID of peer reviewer')
    plt.ylabel('Systematic high/low peer bias')
    plt.show()


# Plot systematic deviations
def plot_sys_dev_broadnarrow(input_path) -> None:
    _, out = compute_systematic_deviation_statistics(input_path)
    users = [i for i in range(1, len(out) + 2)]
    bad_user = 18
    users = users[:bad_user - 1] + users[bad_user:]
    plt.bar(users, out)
    plt.title('Bar plot of systematic broad/narrow individual peer bias')
    plt.xlabel('ID of peer reviewer')
    plt.ylabel('Systematic broad/narrow peer bias')
    plt.show()