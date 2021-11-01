import math
import argparse
import matplotlib as mpl
from numpy.core.defchararray import upper
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
import variability
from accuracy import get_accuracy
from validity import pearson_per_student_formatted
from systematic_deviation import compute_systematic_deviation_statistics, sys_dev_ordering
import neural_network as nn
# import tensorflow as tf
# import keras
from scipy.stats import pearsonr
import training_data_generation as gen
from pre_processing import get_reviewer_grade_sets

# Use: python models.py -t rule -m simple -i data_v2.xlsx
# to run the NN
# Or: python3 models.py -t rule -m simple -i data_v2.xlsx

##############
#### MAIN ####
#### Add your new model here by calling the function you defined below
#### Don't forget to add your model name as an option for the argument way below

def main(args):
    name = args.modelname
    global r_count
    r_count = get_reviewer_grade_sets(args.input).__len__()

    print(r_count)
    # Lets see if we want to run a rule based model
    if(args.modeltype == "rule"):
        if (name == "test"):
            # Run model named "test"
            print("Not implemented yet")

        elif (name == "accuracy"):
            print(accuracy_grades())

        elif (name == "validity"):
            # Solo: validity
            print(validity_grades())

        elif (name == "reliability"):
            # Solo: reliability
            print("Not implemented")

        elif (name == "sys_dev_high"):
            # Solo: systematic deviation: high/low
            print(sys_dev_high_grades())

        elif (name == "sys_dev_wide"):
            # Solo: systematic deviation: broad/narrow
            print(sys_dev_wide_grades())

        elif (name == "sys_dev_order"):
            # Solo: systematic deviation: relative ordering
            print(sys_dev_order_grades())

        elif (name == "sys_dev_full"):
            # Combined: all systematic deviation metrics
            dev_high = sys_dev_high_grades()
            dev_wide = sys_dev_wide_grades()
            dev_ord  = sys_dev_order_grades()

            acc = accuracy_grades()

            corr_high = np.abs(pearsonr(acc, dev_high)[0])
            corr_wide = np.abs(pearsonr(acc, dev_wide)[0])
            corr_ord  = np.abs(pearsonr(acc, dev_ord)[0])

            total_dev = (dev_high * corr_high + dev_wide * corr_wide + dev_ord * corr_ord) / (corr_high + corr_wide + corr_ord)
            print(total_dev)

        elif (name == "val_rel"):
            # Combined: validity and reliability
            pear = pearson_per_student_formatted()
            normalized_pear = normalize(pear, -1, 1, 0.2)
            rel = pearson_per_student_formatted() # TODO CHANGE WHEN RELIABILITY IS IMPLEMENTED
            normalized_rel = normalize(rel, -1, 1, 0.2)

            total = (normalized_pear + normalized_rel) / 2
            print(total)

        elif (name == "all"):
            # Combined: # of reviews handed in, length of comments and consistency (variability of grades of a single student)
            # pear = pearson_per_student_formatted()
            # normalized_pear = normalize(pear, -1, 1, 0.2)
            # rel = pearson_per_student_formatted() # TODO CHANGE WHEN RELIABILITY IS IMPLEMENTED
            # normalized_rel = normalize(rel, -1, 1, 0.2)
            # sys_dev_high = compute_systematic_deviation_statistics()[0]
            # normalized_high = normalize(sys_dev_high, -2, 2, 0.5)
            # sys_dev_wide = compute_systematic_deviation_statistics()[1]
            # normalized_dev_wide = normalize(sys_dev_wide, -1, 1, .8)
            # sys_dev_ord = compute_systematic_deviation_statistics()[1]
            # normalized_dev_ord = normalize(sys_dev_ord, -1, 1, .8)

            dev_high = sys_dev_high_grades()
            dev_wide = sys_dev_wide_grades()
            dev_ord  = sys_dev_order_grades()
            validity = validity_grades()
            
            #total = (normalized_pear + normalized_rel + dev_high + dev_wide + dev_ord) / 5
            #print(total)
        
        else:
            print("Model name not found")

    # If not, did we want a neural network based one?
    elif(args.modeltype == "nn"):
        if (name == "generator"):
            # Run model named "generator"
            print(gen.generate_data(1024))
            print("Test data successfully generated and can be found in data_generated.xls")

        elif (name == "accuracy"):
            acc = get_accuracy(args.input)
            normalized_acc = [normalize(acc, 0, 3, 1)]
            print(normalized_acc)

        elif (name == "validity"):
            # Solo: validity
            pear = pearson_per_student_formatted()
            nn_model(validity_grades())

        elif (name == "reliability"):
            # Solo: reliability
            print("Not implemented")

        elif (name == "sys_dev_high"):
            # Solo: systematic deviation: high/low
            sys_dev = compute_systematic_deviation_statistics()[0]
            normalized_dev = [normalize(sys_dev, -2, 2, 0.5)]
            nn_model(normalized_dev)

        elif (name == "sys_dev_wide"):
            # Solo: systematic deviation: broad/narrow
            sys_dev = compute_systematic_deviation_statistics()[1]
            normalized_dev = [normalize(sys_dev, -1, 1, .8)]
            nn_model(normalized_dev)

        elif (name == "sys_dev_order"):
            # Solo: systematic deviation: relative ordering
            sys_dev = compute_systematic_deviation_statistics()[1]
            normalized_dev = [normalize(sys_dev, -1, 1, .8)]
            nn_model(normalized_dev)

        elif (name == "sys_dev_full"):
            # Combined: all systematic deviation metrics
            sys_dev_high = compute_systematic_deviation_statistics()[0]
            normalized_high = normalize(sys_dev_high, -2, 2, 0.5)
            sys_dev_wide = compute_systematic_deviation_statistics()[1]
            normalized_dev_wide = normalize(sys_dev_wide, -1, 1, .8)
            sys_dev_ord = compute_systematic_deviation_statistics()[1]
            normalized_dev_ord = normalize(sys_dev_ord, -1, 1, .8)

            total_dev = [normalized_high, normalized_dev_wide, normalized_dev_ord]
            nn_model(total_dev)

        elif (name == "val_rel"):
            # Combined: validity and reliability
            pear = pearson_per_student_formatted()
            normalized_pear = normalize(pear, -1, 1, 0.2)
            rel = pearson_per_student_formatted() # TODO CHANGE WHEN RELIABILITY IS IMPLEMENTED
            normalized_rel = normalize(rel, -1, 1, 0.2)

            total = [normalized_pear, normalized_rel]
            nn_model(total)

        elif (name == "all"):
            # Combined: # of reviews handed in, length of comments and consistency (variability of grades of a single student)
            pear = pearson_per_student_formatted()
            normalized_pear = normalize(pear, -1, 1, 0.2)
            rel = pearson_per_student_formatted() # TODO CHANGE WHEN RELIABILITY IS IMPLEMENTED
            normalized_rel = normalize(rel, -1, 1, 0.2)
            sys_dev_high = compute_systematic_deviation_statistics()[0]
            normalized_high = normalize(sys_dev_high, -2, 2, 0.5)
            sys_dev_wide = compute_systematic_deviation_statistics()[1]
            normalized_dev_wide = normalize(sys_dev_wide, -1, 1, .8)
            sys_dev_ord = compute_systematic_deviation_statistics()[1]
            normalized_dev_ord = normalize(sys_dev_ord, -1, 1, .8)

            total = [normalized_pear, normalized_rel, normalized_high, normalized_dev_wide, normalized_dev_ord]
            nn_model(total)

        else:
            print("Model name not found")
    else:
        print("No such model type")


# Method that normalizes output values to grades. values represents the array of values, lower and upper bound are the begin and end of the scale on which the values are placed, 
# upper cutoff discounts the value required for a ten (higher value means higher grades all around)
def normalize(values, lower_bound, upper_bound, upper_cutoff):
    global r_count
    grades = np.zeros(r_count)
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

def accuracy_grades():
    try:
        args
    except NameError:
        in_acc = get_accuracy("data_v2.xlsx")
    else:
        in_acc = get_accuracy(args.input)
    # normalized_acc = normalize(acc, 0, 3, 1)
    max_val = np.nanmax(in_acc)
    acc = max_val - in_acc
    mean = np.nanmean(acc)
    stdev = np.nanstd(acc)
    z = (acc - mean) / stdev
    percentage = np.nan_to_num(7.0 + z)

    i = 0
    for value in percentage:
        percentage[i] = max(0, min(10, value))
        i += 1
    return(percentage)

def validity_grades():
    pear = pearson_per_student_formatted()
    # Pearsons scales from -1 to +1, where higher is better. 

    abs_pear = np.abs(pear)
    mean = np.nanmean(abs_pear)
    stdev = np.nanstd(abs_pear)
    z = (abs_pear - mean) / stdev
    percentage = np.nan_to_num(7.5 + z)
    
    i = 0
    for value in percentage:
        percentage[i] = max(0, min(10, value))
        i += 1
    return(percentage)

def sys_dev_high_grades():
    sys_dev = compute_systematic_deviation_statistics()[0]
    # normalized_dev = normalize(sys_dev, -2, 2, 0.5)

    abs_sys_dev = np.abs(sys_dev)
    max_val = np.nanmax(abs_sys_dev)
    abs_sys_dev_flip = max_val - abs_sys_dev
    mean = np.nanmean(abs_sys_dev_flip)
    stdev = np.nanstd(abs_sys_dev_flip)
    z = (abs_sys_dev_flip - mean) / stdev
    percentage = np.nan_to_num(6.5 + z)

    i = 0
    for value in percentage:
        percentage[i] = max(0, min(10, value))
        i += 1
    return(percentage)

def sys_dev_wide_grades():
    sys_dev = compute_systematic_deviation_statistics()[1]

    max_val = np.nanmax(sys_dev)
    abs_sys_dev_flip = max_val - sys_dev
    mean = np.nanmean(abs_sys_dev_flip)
    stdev = np.nanstd(abs_sys_dev_flip)
    z = (abs_sys_dev_flip - mean) / stdev
    percentage = np.nan_to_num(6.5 + z)

    i = 0
    for value in percentage:
        percentage[i] = max(0, min(10, value))
        i += 1
    return(percentage)

def sys_dev_order_grades():
    sys_dev = sys_dev_ordering()

    abs_sys_dev = np.abs(sys_dev)
    mean = np.nanmean(abs_sys_dev)
    stdev = np.nanstd(abs_sys_dev)
    z = (abs_sys_dev - mean) / stdev
    percentage = np.nan_to_num(7.5 + z)

    i = 0
    for value in percentage:
        percentage[i] = max(0, min(10, value))
        i += 1
    return(percentage)

##############
### MODELS ###
### Write functions for calculating a certain model here. Use methods below for accessing metrics


# The simplest model, only based on accuracy
def r_simple_model(input_path: str):
    score = []
    model = get_accuracy(input_path)
    for reviewer_score in model:
        score.append(max(1, min(10, (10 - (reviewer_score - 0.5) * 5))))
    return score


# Simple neural network model, accuracy as labels, variability and grades(?) as input
def nn_model(input_data):
    global r_count
    data = np.zeros([r_count, input_data.__len__()])
    i = 0
    while (i < input_data.__len__()):
        j = 0
        while (j < r_count):
            # Ugly fix for that annoying reviewer 18
            if(j == 17):
                data[j, i] = input_data[i][j-2]
            else:
                data[j, i] = input_data[i][j]
            j += 1
        i += 1

    acc = get_accuracy(args.input)
    labels = np.transpose([normalize(acc, 0, 3, 1)])
    
    training_count = int(r_count * 0.8)

    training_data = data[:training_count]
    training_labels = labels[:training_count]

    testing_data = data[training_count:]
    testing_labels = labels[training_count:]

    # training_data = tf.data.Dataset.from_tensor_slices((data[:training_count],labels[:training_count])) 
    # testing_data = tf.data.Dataset.from_tensor_slices((data[training_count:],labels[training_count:])) 

# Data you want to predict a reviewer grade for
    data = [8]

# DISABLE IF YOU GET ERROR: CORE DUMPED
    network = nn.neural_network_model(args.modelname, input_data.__len__())
    network.train(training_data, training_labels, testing_data, testing_labels, 1, 1)
    predicted_grade = network.predict_grade(data)

    print(predicted_grade)

##############
##############


# change colors and run the line below to change the colors used in the plot
COLORS = ['#ff1969', '#ffbd59', '#00c2cb', '#3788d4', '#044aad', '#000000']
mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color', COLORS)


def plot_grades_per_reviewer_rule_based():
    metrics = ["systematic problems in ordering", "systematic broad/narrow peer bias", "systematic high/low peer bias", "validity", "accuracy"]
    reviewer_ids = [i for i in range(1, 44 + 1)]

    for metric in metrics:
        if metric == "accuracy":
            grades = accuracy_grades()
        elif metric == "validity":
            grades = validity_grades()
        elif metric == "reliability":
            print("Metrics " + metric + " has not been implemented yet...")
            return
        elif metric == "systematic high/low peer bias":
            grades = sys_dev_high_grades()
        elif metric == "systematic broad/narrow peer bias":
            grades = sys_dev_wide_grades()
        elif metric == "systematic problems in ordering":
            grades = sys_dev_order_grades()
        else:
            print("Metrics " + metric + " was not recognized...")
            return

        assert grades.ndim == 1
        assert len(grades) == 44
        plt.scatter(reviewer_ids, grades, label=metric)

    plt.title('Bar plot of rule-based model grades for each peer reviewer')
    plt.xlabel('ID of peer reviewer')
    plt.ylabel("Grades according to rule-based model")
    plt.grid()
    plt.legend()
    plt.show()


def plot_correlation_grades_with_acc(model):
    # reliability not yet added, because not yet implemented
    # neural network model not yet added, because not yet implemented
    # (in)accuracy used for 'true' grades for quality
    metrics = ["validity", "systematic high/low peer bias", "systematic broad/narrow peer bias", "systematic problems in ordering"]
    acc = accuracy_grades()

    out = []
    p_values = []
    for metric in metrics:
        if model == "rule-based":
            if metric == "validity":
                values = validity_grades()
            elif metric == "reliability":
                print("Metrics " + metric + " has not been implemented yet...")
                return
            elif metric == "systematic high/low peer bias":
                values = sys_dev_high_grades()
            elif metric == "systematic broad/narrow peer bias":
                values = sys_dev_wide_grades()
            elif metric == "systematic problems in ordering":
                values = sys_dev_order_grades()
            else:
                print("Metrics " + metric + " was not recognized...")
                return
        elif model == "neural network":
            if metric == "validity":
                values = not_implemented_yet()
            elif metric == "reliability":
                print("Metrics " + metric + " has not been implemented yet...")
                return
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
        plt.text(bars[bar_id].get_x(), bars[bar_id].get_height() + .005, "p-value: " + "{:.2e}".format(p_values[bar_id]))
    plt.title('Bar plot of absolute correlation with grades for each metric model per reviewer')
    plt.xlabel('Metric')
    plt.xticks(rotation=90)
    plt.ylabel('Absolute correlation')
    plt.grid()
    plt.show()


##############
### Inputs ###
### Write functions here for accessing metric values (or auxilliary calculations)

#######################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--modeltype', choices=["rule","nn"],
                        help='The type of model you want to use')

    parser.add_argument('-m', '--modelname', choices=["test","generator","accuracy","validity","reliability","sys_dev_high","sys_dev_wide","sys_dev_order","sys_dev_full","val_rel","all"], default="accuracy",
                        help='The name of the model you want to use')

    parser.add_argument('-i', '--input',
                        help='Specify a path to the data file')

    args = parser.parse_args()
    main(args)