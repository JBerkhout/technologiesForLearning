import config
import math
import argparse
import matplotlib as mpl
from numpy.core.defchararray import upper
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame

from remarks_size import compute_student_remarks, compute_student_topic_remarks
from variability import read_topic_variability_statistics
from accuracy import get_accuracy, accuracy_per_topic
from reliability import compute_student_reliability, compute_student_topic_reliability
from validity import pearson_per_student_formatted, pearson_per_topic_formatted
from systematic_deviation import compute_systematic_deviation_statistics, sys_dev_ordering, sys_high_low_official, \
    sys_spread_official
import neural_network as nn
# import tensorflow as tf
# import keras
from scipy.stats import pearsonr
import training_data_generation as gen
from pre_processing import get_reviewer_grade_sets

VAR_PER_TOPIC_REVERSE = (np.max(read_topic_variability_statistics(True)) - read_topic_variability_statistics(True))
ACC_OVER_TIME_REVERSE = (np.max(accuracy_per_topic("data_v2.xlsx")) - accuracy_per_topic("data_v2.xlsx"))

# Use: python models.py -t rule -m simple -i data_v2.xlsx
# to run the NN
# Or: python3 models.py -t rule -m simple -i data_v2.xlsx

##############
#### MAIN ####
#### Add your new model here by calling the function you defined below
#### Don't forget to add your model name as an option for the argument way below

def main(args):
    name = args.modelname
    input_path = args.input
    input_path_pred = "data_v2.xlsx"
    # global r_count
    # r_count = get_reviewer_grade_sets(args.input).__len__()

    # print(r_count)
    # Lets see if we want to run a rule based model
    if (args.modeltype == "rule"):
        if (name == "test"):
            # Run model named "test"
            print("Not implemented yet")

        elif (name == "accuracy"):
            print(accuracy_grades(False, input_path))

        elif (name == "remarks size"): 
            # Solo: remarks size
            print(remarks_size_grades(False))

        elif (name == "validity"):
            # Solo: validity
            print(validity_grades(False, input_path))

        elif (name == "reliability"):
            # Solo: reliability
            print(reliability_grades(False, input_path))

        elif (name == "sys_dev_high"):
            # Solo: systematic deviation: high/low
            print(sys_dev_high_grades(False, input_path))

        elif (name == "sys_dev_wide"):
            # Solo: systematic deviation: broad/narrow
            print(sys_dev_wide_grades(False, input_path))

        elif (name == "sys_dev_order"):
            # Solo: systematic deviation: relative ordering
            print(sys_dev_order_grades(False, input_path))

        elif (name == "sys_dev_full"):
            # Combined: all systematic deviation metrics
            print(full_sys_dev_grades(input_path))

        elif (name == "val_rel"):
            # Combined: validity and reliability
            print(val_rel_grades(input_path))

        elif (name == "all"):
            # Combined: # of reviews handed in, length of comments and consistency (variability of grades of a single student)
            print(all_grades(input_path))
        else:
            print("Model name not found")

    # If not, did we want a neural network based one?
    elif (args.modeltype == "nn"):
        if (name == "generator"):
            # Run model named "generator"
            gen.generate_data(1024)
            print("Test data successfully generated and can be found in data_generated.xls")

        elif (name == "accuracy"):
            nn_model([accuracy_grades(True, input_path)], [accuracy_grades(True, input_path_pred)], True)

        elif (name == "remarks size"): 
            # Solo: remarks size
            nn_model([remarks_size_grades(True)], [remarks_size_grades(True)], True)

        elif (name == "validity"):
            # Solo: validity
            nn_model([validity_grades(True, input_path)], [validity_grades(True, input_path_pred)], True)

        elif (name == "reliability"):
            # Solo: reliability
            nn_model([reliability_grades(True, input_path)], [reliability_grades(True, input_path_pred)], True)

        elif (name == "sys_dev_high"):
            # Solo: systematic deviation: high/low
            nn_model([sys_dev_high_grades(True, input_path)], [sys_dev_high_grades(True, input_path_pred)], True)

        elif (name == "sys_dev_wide"):
            # Solo: systematic deviation: broad/narrow
            nn_model([sys_dev_wide_grades(True, input_path)], [sys_dev_wide_grades(True, input_path_pred)], True)

        elif (name == "sys_dev_order"):
            # Solo: systematic deviation: relative ordering
            nn_model([sys_dev_order_grades(True, input_path)], [sys_dev_order_grades(True, input_path)], True)

        elif (name == "sys_dev_full"):
            # Combined: all systematic deviation metrics
            sys_dev_high = compute_systematic_deviation_statistics(input_path)[0]
            normalized_high = normalize(sys_dev_high, -2, 2, 0.5)
            sys_dev_wide = compute_systematic_deviation_statistics(input_path)[1]
            normalized_dev_wide = normalize(sys_dev_wide, -1, 1, .8)
            sys_dev_ord = compute_systematic_deviation_statistics(input_path)[1]
            normalized_dev_ord = normalize(sys_dev_ord, -1, 1, .8)

            total_dev = [normalized_high, normalized_dev_wide, normalized_dev_ord]
            nn_model(total_dev)

        elif (name == "val_rel"):
            # Combined: validity and reliability
            pear = pearson_per_student_formatted()
            normalized_pear = normalize(pear, -1, 1, 0.2)
            rel = pearson_per_student_formatted()  # TODO CHANGE WHEN RELIABILITY IS IMPLEMENTED
            normalized_rel = normalize(rel, -1, 1, 0.2)

            total = [normalized_pear, normalized_rel]
            nn_model(total)

        elif (name == "all"):
            # Combined: # of reviews handed in, length of comments and consistency (variability of grades of a single student)
            rem = compute_student_remarks()
            normalized_rem = normalize(rem, -1, 1, 0.2)
            pear = pearson_per_student_formatted(input_path)
            normalized_pear = normalize(pear, -1, 1, 0.2)
            rel = compute_student_reliability(input_path)
            normalized_rel = normalize(rel, -1, 1, 0.2)
            sys_dev_high = compute_systematic_deviation_statistics(input_path)[0]
            normalized_high = normalize(sys_dev_high, -2, 2, 0.5)
            sys_dev_wide = compute_systematic_deviation_statistics(input_path)[1]
            normalized_dev_wide = normalize(sys_dev_wide, -1, 1, .8)
            sys_dev_ord = compute_systematic_deviation_statistics(input_path)[1]
            normalized_dev_ord = normalize(sys_dev_ord, -1, 1, .8)

            total = [normalized_rem, normalized_pear, normalized_rel, normalized_high, normalized_dev_wide, normalized_dev_ord]
            nn_model(total, np.empty([6, config.r_count]), True)

        else:
            print("Model name not found")
    else:
        print("No such model type")

def full_sys_dev_grades(input_path):
    dev_high = sys_dev_high_grades(False, input_path)
    dev_wide = sys_dev_wide_grades(False, input_path)
    dev_ord = sys_dev_order_grades(False, input_path)

    acc = accuracy_grades(False, input_path)

    corr_high = np.abs(pearsonr(acc, dev_high)[0])
    corr_wide = np.abs(pearsonr(acc, dev_wide)[0])
    corr_ord = np.abs(pearsonr(acc, dev_ord)[0])

    total_dev = (dev_high * corr_high + dev_wide * corr_wide + dev_ord * corr_ord) / (
                corr_high + corr_wide + corr_ord)
    return(total_dev)

def val_rel_grades(input_path):
    validity = validity_grades(False, input_path)
    reliability = reliability_grades(False, input_path)

    acc = accuracy_grades(False, input_path)

    corr_val = np.abs(pearsonr(acc, validity)[0])
    corr_rel = np.abs(pearsonr(acc, reliability)[0])

    total = (validity * corr_val + reliability * corr_rel) / (corr_val + corr_rel)
    return(total)

def all_grades(input_path):
    dev_high = sys_dev_high_grades(False, input_path)
    dev_wide = sys_dev_wide_grades(False, input_path)
    dev_ord = sys_dev_order_grades(False, input_path)
    validity = validity_grades(False, input_path)
    reliability = reliability_grades(False, input_path)
    remarks_size = remarks_size_grades(False)

    acc = accuracy_grades(False, input_path)

    corr_high = np.abs(pearsonr(acc, dev_high)[0])
    corr_wide = np.abs(pearsonr(acc, dev_wide)[0])
    corr_ord = np.abs(pearsonr(acc, dev_ord)[0])
    corr_val = np.abs(pearsonr(acc, validity)[0])
    corr_rel = np.abs(pearsonr(acc, reliability)[0])
    corr_rem = np.abs(pearsonr(acc, remarks_size)[0])

    total = (dev_high * corr_high + dev_wide * corr_wide + dev_ord * corr_ord + validity * corr_val + reliability * corr_rel + remarks_size * corr_rem) / (
                corr_high + corr_wide + corr_ord + corr_val + corr_rel + corr_rem)
    return(total)

# Method that normalizes output values to grades. values represents the array of values, lower and upper bound are the begin and end of the scale on which the values are placed, 
# upper cutoff discounts the value required for a ten (higher value means higher grades all around)
def normalize(values, lower_bound, upper_bound, upper_cutoff):
    # global r_count
    grades = np.zeros(config.r_count)
    i = 0
    range = upper_bound - lower_bound
    for value in values:
        if (math.isnan(value)):
            grades[i] = math.nan
            i += 1
        else:
            # If you only want to normalize,          \/ Comment this \/
            normalized_val = (value + (0 - lower_bound) + upper_cutoff) / range * 10
            bound_val = max(1, min(10, normalized_val))
            grades[i] = bound_val
            i += 1
    return grades


def accuracy_grades(use_nn, input_path, incl=""):
    # try:
    #     args
    # except NameError:
    #     in_acc = get_accuracy("data_v2.xlsx")
    # else:
    #     in_acc = get_accuracy(args.input)

    if incl == "variability":
        in_acc = []
        for student_accs in get_accuracy(input_path, True):
            if np.isnan(student_accs).all():
                in_acc.append(np.nan)
            else:
                in_acc.append(np.nanmean(VAR_PER_TOPIC_REVERSE * student_accs))
    elif incl == "over time":
        in_acc = []
        for student_accs in get_accuracy(input_path, True):
            if np.isnan(student_accs).all():
                in_acc.append(np.nan)
            else:
                in_acc.append(np.nanmean(ACC_OVER_TIME_REVERSE * student_accs))
    else:
        in_acc = get_accuracy(input_path)
        # normalized_acc = normalize(acc, 0, 3, 1)

    max_val = np.nanmax(in_acc)
    acc = max_val - in_acc
    if (use_nn):
        return acc
    mean = np.nanmean(acc)
    stdev = np.nanstd(acc)
    z = (acc - mean) / stdev
    percentage = np.nan_to_num(7.0 + z)

    i = 0
    for value in percentage:
        percentage[i] = max(0, min(10, value))
        i += 1
    return (percentage)


def remarks_size_grades(use_nn, incl=""):
    if incl == "over time":
        remarks_sizes = []
        for student_remarks_sizes in compute_student_topic_remarks():
            if np.isnan(student_remarks_sizes).all():
                remarks_sizes.append(np.nan)
            else:
                remarks_sizes.append(np.nanmean(ACC_OVER_TIME_REVERSE * student_remarks_sizes))
    elif incl == "variability":
        remarks_sizes = []
        for student_remarks_sizes in compute_student_topic_remarks():
            if np.isnan(student_remarks_sizes).all():
                remarks_sizes.append(np.nan)
            else:
                remarks_sizes.append(np.nanmean(VAR_PER_TOPIC_REVERSE * student_remarks_sizes))
    else:
        remarks_sizes = compute_student_remarks()

    if (use_nn):
        return remarks_sizes
    mean = np.nanmean(remarks_sizes)
    stdev = np.nanstd(remarks_sizes)
    z = (remarks_sizes - mean) / stdev
    percentage = np.nan_to_num(7.0 + z)

    i = 0
    for value in percentage:
        percentage[i] = max(0, min(10, value))
        i += 1
    return (percentage)


def reliability_grades(use_nn, input_path, incl=""):
    if incl == "variability":
        rel = []
        for student_rels in compute_student_topic_reliability():
            if np.isnan(student_rels).all():
                rel.append(np.nan)
            else:
                rel.append(np.nanmean(VAR_PER_TOPIC_REVERSE * student_rels))
    elif incl == "over time":
        rel = []
        for student_rels in compute_student_topic_reliability():
            if np.isnan(student_rels).all():
                rel.append(np.nan)
            else:
                rel.append(np.nanmean(ACC_OVER_TIME_REVERSE * student_rels))
    else:
        rel = compute_student_reliability(input_path)

    max_val = np.nanmax(rel)
    rel = max_val - rel
    if use_nn:
        return rel
    mean = np.nanmean(rel)
    stdev = np.nanstd(rel)
    z = (rel - mean) / stdev
    percentage = np.nan_to_num(7.0 + z)

    i = 0
    for value in percentage:
        percentage[i] = max(0, min(10, value))
        i += 1
    return percentage


def validity_grades(use_nn, input_path, incl=""):
    # Pearsons scales from -1 to +1, where higher is better.
    if incl == "variability":
        pear = []
        for student_pears in pearson_per_topic_formatted(input_path):
            if np.isnan(student_pears).all():
                pear.append(np.nan)
            else:
                pear.append(np.nanmean(VAR_PER_TOPIC_REVERSE * student_pears))
    elif incl == "over time":
        pear = []
        for student_pears in pearson_per_topic_formatted(input_path):
            if np.isnan(student_pears).all():
                pear.append(np.nan)
            else:
                pear.append(np.nanmean(ACC_OVER_TIME_REVERSE * student_pears))
    else:
        pear = pearson_per_student_formatted(input_path)

    abs_pear = np.abs(pear)
    if (use_nn):
        return abs_pear
    mean = np.nanmean(abs_pear)
    stdev = np.nanstd(abs_pear)
    z = (abs_pear - mean) / stdev
    percentage = np.nan_to_num(7.5 + z)

    i = 0
    for value in percentage:
        percentage[i] = max(0, min(10, value))
        i += 1
    return (percentage)


def sys_dev_high_grades(use_nn, input_path, incl=""):
    if incl == "variability":
        sys_dev = []
        for student_sys_devs in sys_high_low_official("data_v2.xlsx", True):
            if np.isnan(student_sys_devs).all():
                sys_dev.append(np.nan)
            else:
                sys_dev.append(np.nanmean(VAR_PER_TOPIC_REVERSE * student_sys_devs))
    elif incl == "over time":
        sys_dev = []
        for student_sys_devs in sys_high_low_official("data_v2.xlsx", True):
            if np.isnan(student_sys_devs).all():
                sys_dev.append(np.nan)
            else:
                sys_dev.append(np.nanmean(ACC_OVER_TIME_REVERSE * student_sys_devs))
    else:
        sys_dev = compute_systematic_deviation_statistics(input_path)[0]
        # normalized_dev = normalize(sys_dev, -2, 2, 0.5)

    abs_sys_dev = np.abs(sys_dev)
    max_val = np.nanmax(abs_sys_dev)
    abs_sys_dev_flip = max_val - abs_sys_dev
    if (use_nn):
        return abs_sys_dev_flip
    mean = np.nanmean(abs_sys_dev_flip)
    stdev = np.nanstd(abs_sys_dev_flip)
    z = (abs_sys_dev_flip - mean) / stdev
    percentage = np.nan_to_num(6.5 + z)

    i = 0
    for value in percentage:
        percentage[i] = max(0, min(10, value))
        i += 1
    return (percentage)


def sys_dev_wide_grades(use_nn, input_path, incl=""):
    if incl == "variability":
        sys_dev = []
        for student_sys_devs in sys_spread_official("data_v2.xlsx", True):
            if np.isnan(student_sys_devs).all():
                sys_dev.append(np.nan)
            else:
                sys_dev.append(np.nanmean(VAR_PER_TOPIC_REVERSE * student_sys_devs))
    if incl == "over time":
        sys_dev = []
        for student_sys_devs in sys_spread_official("data_v2.xlsx", True):
            if np.isnan(student_sys_devs).all():
                sys_dev.append(np.nan)
            else:
                sys_dev.append(np.nanmean(ACC_OVER_TIME_REVERSE * student_sys_devs))
    else:
        sys_dev = compute_systematic_deviation_statistics(input_path)[1]
        # normalized_dev = normalize(sys_dev, -2, 2, 0.5)

    max_val = np.nanmax(sys_dev)
    abs_sys_dev_flip = max_val - sys_dev
    if (use_nn):
        return abs_sys_dev_flip
    mean = np.nanmean(abs_sys_dev_flip)
    stdev = np.nanstd(abs_sys_dev_flip)
    z = (abs_sys_dev_flip - mean) / stdev
    percentage = np.nan_to_num(6.5 + z)

    i = 0
    for value in percentage:
        percentage[i] = max(0, min(10, value))
        i += 1
    return (percentage)


def sys_dev_order_grades(use_nn, input_path, incl=""):
    if incl == "variability":
        sys_dev = []
        for student_sys_devs in sys_dev_ordering(input_path, True):
            if np.isnan(student_sys_devs).all():
                sys_dev.append(np.nan)
            else:
                sys_dev.append(np.nanmean(VAR_PER_TOPIC_REVERSE * student_sys_devs))
    elif incl == "over time":
        sys_dev = []
        for student_sys_devs in sys_dev_ordering(input_path, True):
            if np.isnan(student_sys_devs).all():
                sys_dev.append(np.nan)
            else:
                sys_dev.append(np.nanmean(ACC_OVER_TIME_REVERSE * student_sys_devs))
    else:
        sys_dev = sys_dev_ordering(input_path)

    abs_sys_dev = np.abs(sys_dev)
    if (use_nn):
        return abs_sys_dev
    mean = np.nanmean(abs_sys_dev)
    stdev = np.nanstd(abs_sys_dev)
    z = (abs_sys_dev - mean) / stdev
    percentage = np.nan_to_num(7.5 + z)

    i = 0
    for value in percentage:
        percentage[i] = max(0, min(10, value))
        i += 1
    return (percentage)


##############
### MODELS ###
### Write functions for calculating a certain model here. Use methods below for accessing metrics


# The simplest model, only based on accuracy
# def r_simple_model(input_path: str):
#     score = []
#     model = get_accuracy(input_path)
#     for reviewer_score in model:
#         score.append(max(1, min(10, (10 - (reviewer_score - 0.5) * 5))))
#     return score


# Simple neural network model, accuracy as labels, variability and grades(?) as input
def nn_model(train_input_data, predict_input_data, train_network):
    # global r_count

    # training_data = tf.data.Dataset.from_tensor_slices((data[:training_count],labels[:training_count])) 
    # testing_data = tf.data.Dataset.from_tensor_slices((data[training_count:],labels[training_count:])) 

    # DISABLE IF YOU GET ERROR: CORE DUMPED
    network = nn.neural_network_model(args.modelname, train_input_data.__len__())
    if (train_network):
        data = np.zeros(shape=(config.r_train_count, train_input_data.__len__()))
        i = 0
        while (i < train_input_data.__len__()):
            j = 0
            while (j < config.r_train_count):
                # Ugly fix for that annoying reviewer 18
                # if(j == 17):
                #     data[j, i] = train_input_data[i][j-2]
                # else:   
                data[j, i] = train_input_data[i][j]
                j += 1
            i += 1

        # acc = get_accuracy(args.input)
        labels = np.transpose(accuracy_grades(False, "data_generated.xls"))

        training_count = int(config.r_train_count * 0.8)

        training_data = data[:training_count]
        # print(training_data)
        training_labels = labels[:training_count]
        # print(training_labels)

        testing_data = data[training_count:]
        testing_labels = labels[training_count:]

        network.train(training_data, training_labels, testing_data, testing_labels, 1, 1)
        print("Network trained successfully!")
    else:
        predict_data = np.zeros([config.r_count, predict_input_data.__len__()])
        i = 0
        while (i < predict_input_data.__len__()):
            j = 0
            while (j < config.r_count):
                # Ugly fix for that annoying reviewer 18
                if (j == 17):
                    predict_data[j, i] = predict_input_data[i][j - 2]
                else:
                    predict_data[j, i] = predict_input_data[i][j]
                j += 1
            i += 1

        # Data you want to predict a reviewer grade for
        predicted_grade = network.predict_grade(predict_input_data)
        print(predicted_grade)


##############
##############


# change colors and run the line below to change the colors used in the plot
COLORS = ['#ff1969', '#ffbd59', '#00c2cb', '#3788d4', '#044aad', '7317ae', '#000000']
mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color', COLORS)


def plot_grades_per_reviewer_rule_based(incl=""):
    metrics = ["systematic problems in ordering", "systematic broad/narrow peer bias", "systematic high/low peer bias",
               "reliability", "validity", "remarks size", "accuracy"]
    reviewer_ids = [i for i in range(1, config.r_count + 1)]

    for metric in metrics:
        if metric == "accuracy":
            grades = accuracy_grades(False, "data_v2.xlsx", incl)
        elif metric == "remarks size":
            grades = remarks_size_grades(False, incl)
        elif metric == "validity":
            grades = validity_grades(False, "data_v2.xlsx", incl)
        elif metric == "reliability":
            grades = reliability_grades(False, "data_v2.xlsx", incl)
        elif metric == "systematic high/low peer bias":
            grades = sys_dev_high_grades(False, "data_v2.xlsx", incl)
        elif metric == "systematic broad/narrow peer bias":
            grades = sys_dev_wide_grades(False, "data_v2.xlsx", incl)
        elif metric == "systematic problems in ordering":
            grades = sys_dev_order_grades(False, "data_v2.xlsx", incl)
        else:
            print("Metrics " + metric + " was not recognized...")
            return

        assert grades.ndim == 1
        assert len(grades) == config.r_count
        plt.scatter(reviewer_ids, grades, label=metric)

    if incl == "variability":
        plt.title('Bar plot of rule-based model grades for each peer reviewer considering variability of each topic')
    elif incl == "over time":
        plt.title('Bar plot of rule-based model grades for each peer reviewer considering over time performance')
    else:
        plt.title('Bar plot of rule-based model grades for each peer reviewer')
    plt.xlabel('ID of peer reviewer')
    plt.ylabel("Grades according to rule-based model")
    plt.grid()
    plt.legend()
    plt.show()


def plot_grades_per_reviewer_rule_based_combined():
    metrics = ["all individual metrics combined", "systematic deviations", "validity and reliability", "accuracy"]
    reviewer_ids = [i for i in range(1, config.r_count + 1)]

    for metric in metrics:
        if metric == "accuracy":
            grades = accuracy_grades(False, "data_v2.xlsx", "")
        elif metric == "validity and reliability":
            grades = val_rel_grades("data_v2.xlsx")
        elif metric == "systematic deviations":
            grades = full_sys_dev_grades("data_v2.xlsx")
        elif metric == "all individual metrics combined":
            grades = all_grades("data_v2.xlsx")
        else:
            print("Metrics " + metric + " was not recognized...")
            return

        assert grades.ndim == 1
        assert len(grades) == config.r_count
        plt.scatter(reviewer_ids, grades, label=metric)

    plt.title('Bar plot of rule-based combined model grades for each peer reviewer')
    plt.xlabel('ID of peer reviewer')
    plt.ylabel("Grades according to rule-based model")
    plt.grid()
    plt.legend()
    plt.show()


def plot_correlation_grades_with_acc(model="rule-based", incl=""):
    # neural network model not yet added, because not yet implemented
    # (in)accuracy used for 'true' grades for quality
    metrics = ["remarks size", "validity", "reliability", "systematic high/low peer bias", "systematic broad/narrow peer bias",
               "systematic problems in ordering"]
    acc = accuracy_grades(False, "data_v2.xlsx")

    out = []
    p_values = []
    for metric in metrics:
        if model == "rule-based":
            if metric == "remarks size":
                values = remarks_size_grades(False, incl)
            elif metric == "validity":
                values = validity_grades(False, "data_v2.xlsx", incl)
            elif metric == "reliability":
                values = reliability_grades(False, "data_v2.xlsx", incl)
            elif metric == "systematic high/low peer bias":
                values = sys_dev_high_grades(False, "data_v2.xlsx", incl)
            elif metric == "systematic broad/narrow peer bias":
                values = sys_dev_wide_grades(False, "data_v2.xlsx", incl)
            elif metric == "systematic problems in ordering":
                values = sys_dev_order_grades(False, "data_v2.xlsx", incl)
            else:
                print("Metrics " + metric + " was not recognized...")
                return
        elif model == "neural network":
            if metric == "remarks size":
                values = not_implemented_yet()
            elif metric == "validity":
                values = not_implemented_yet()
            elif metric == "reliability":
                values = not_implemented_yet()
            elif metric == "systematic high/low peer bias":
                values = not_implemented_yet()
            elif metric == "systematic broad/narrow peer bias":
                values = not_implemented_yet()
            elif metric == "systematic problems in ordering":
                values = not_implemented_yet()
            else:
                print("Metrics " + metric + " was not recognized...")
                return
        else:
            print("Model " + model + " was not recognized...")
            return

        single_2d_arr = np.vstack([values, acc])
        without_nan = single_2d_arr[:, ~np.any(np.isnan(single_2d_arr), axis=0)]

        corr, p = pearsonr(without_nan[0], without_nan[1])
        out.append(np.abs(corr))
        p_values.append(p)

    # print(out)
    bars = plt.bar([metric.capitalize() for metric in metrics], out)  # , color=COLOR)
    for bar_id in range(len(bars)):
        plt.text(bars[bar_id].get_x(), bars[bar_id].get_height() + .005,
                 "p-value: " + "{:.2e}".format(p_values[bar_id]))
    if incl == "variability":
        plt.title('Bar plot of absolute correlation with rule-based grades for each metric model considering variability of each topic')
    elif incl == "over time":
        plt.title('Bar plot of absolute correlation with rule-based grades for each metric model considering performance over time')
    else:
        plt.title('Bar plot of absolute correlation with rule-based grades for each metric model')
    plt.xlabel('Metric')
    # plt.xticks(rotation=90)
    plt.ylabel('Absolute correlation')
    plt.grid()
    plt.show()


def plot_correlation_grades_with_acc_combined_models(model="rule-based"):
    # neural network model not yet added, because not yet implemented
    # (in)accuracy used for 'true' grades for quality
    metrics = ["validity and reliability", "systematic deviations", "all individual metrics combined"]
    acc = accuracy_grades(False, "data_v2.xlsx")

    out = []
    p_values = []
    for metric in metrics:
        if model == "rule-based":
            if metric == "validity and reliability":
                values = val_rel_grades("data_v2.xlsx")
            elif metric == "systematic deviations":
                values = full_sys_dev_grades("data_v2.xlsx")
            elif metric == "all individual metrics combined":
                values = all_grades("data_v2.xlsx")
            else:
                print("Metrics " + metric + " was not recognized...")
                return
        elif model == "neural network":
            if metric == "validity and reliability":
                values = not_implemented_yet()
            elif metric == "systematic deviations":
                values = not_implemented_yet()
            elif metric == "all individual metrics combined":
                values = not_implemented_yet()
            else:
                print("Metrics " + metric + " was not recognized...")
                return
        else:
            print("Model " + model + " was not recognized...")
            return

        single_2d_arr = np.vstack([values, acc])
        without_nan = single_2d_arr[:, ~np.any(np.isnan(single_2d_arr), axis=0)]

        corr, p = pearsonr(without_nan[0], without_nan[1])
        out.append(np.abs(corr))
        p_values.append(p)

    # print(out)
    bars = plt.bar([metric.capitalize() for metric in metrics], out)  # , color=COLOR)
    for bar_id in range(len(bars)):
        plt.text(bars[bar_id].get_x(), bars[bar_id].get_height() + .005,
                 "p-value: " + "{:.2e}".format(p_values[bar_id]))
    plt.title('Bar plot of absolute correlation with rule-based grades for each combined metric model')
    plt.xlabel('Metric')
    # plt.xticks(rotation=90)
    plt.ylabel('Absolute correlation')
    plt.grid()
    plt.show()


##############
### Inputs ###
### Write functions here for accessing metric values (or auxilliary calculations)

#######################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--modeltype', choices=["rule", "nn"],
                        help='The type of model you want to use')

    parser.add_argument('-m', '--modelname',
                        choices=["test", "generator", "accuracy", "validity", "reliability", "sys_dev_high",
                                 "sys_dev_wide", "sys_dev_order", "sys_dev_full", "val_rel", "all"], default="accuracy",
                        help='The name of the model you want to use')

    parser.add_argument('-i', '--input',
                        help='Specify a path to the data file')

    args = parser.parse_args()
    main(args)
