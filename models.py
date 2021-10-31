import math
import argparse
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
import tensorflow as tf
import keras

# Use: python models.py -t rule -m simple -i data_v2.xlsx
# to run the NN
# Or: python3 models.py -t rule -m simple -i data_v2.xlsx

##############
#### MAIN ####
#### Add your new model here by calling the function you defined below
#### Don't forget to add your model name as an option for the argument way below

def main(args):
    name = args.modelname
    # Lets see if we want to run a rule based model
    if(args.modeltype == "rule"):
        if (name == "test"):
            # Run model named "test"
            print("Not implemented yet")

        elif (name == "accuracy"):
            acc = get_accuracy(args.input)
            normalized_acc = normalize(acc, 0, 3, 1)
            print(normalized_acc)

        elif (name == "validity"):
            # Solo: validity
            pear = pearson_per_student_formatted()
            # Pearsons scales from -1 to +1, where higher is better. 
            normalized_pear = normalize(pear, -1, 1, 0.2)
            print(normalized_pear)

        elif (name == "reliability"):
            # Solo: reliability
            print("Not implemented")
            
        elif (name == "sys_dev_high"):
            # Solo: systematic deviation: high/low
            sys_dev = compute_systematic_deviation_statistics()[0]
            normalized_dev = normalize(sys_dev, -2, 2, 0.5)
            print(normalized_dev)
        
        elif (name == "sys_dev_wide"):
            # Solo: systematic deviation: broad/narrow
            sys_dev = compute_systematic_deviation_statistics()[1]
            normalized_dev = normalize(sys_dev, -1, 1, .8)
            print(normalized_dev)
            
        elif (name == "sys_dev_order"):
            # Solo: systematic deviation: relative ordering
            sys_dev = compute_systematic_deviation_statistics()[1]
            normalized_dev = normalize(sys_dev, -1, 1, .8)
            print(normalized_dev)
        
        elif (name == "sys_dev_full"):
            # Combined: all systematic deviation metrics
            sys_dev_high = compute_systematic_deviation_statistics()[0]
            normalized_high = normalize(sys_dev_high, -2, 2, 0.5)
            sys_dev_wide = compute_systematic_deviation_statistics()[1]
            normalized_dev_wide = normalize(sys_dev_wide, -1, 1, .8)
            sys_dev_ord = compute_systematic_deviation_statistics()[1]
            normalized_dev_ord = normalize(sys_dev_ord, -1, 1, .8)

            total_dev = (normalized_high + normalized_dev_wide + normalized_dev_ord) / 3
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
            
            total = (normalized_pear + normalized_rel + normalized_high + normalized_dev_wide + normalized_dev_ord) / 5
            print(total)
        
        else:
            print("Model name not found")

    # If not, did we want a neural network based one?
    elif(args.modeltype == "nn"):
        if (name == "test"):
            # Run model named "test"
            print("Not implemented yet")

        elif (name == "accuracy"):
            acc = get_accuracy(args.input)
            normalized_acc = [normalize(acc, 0, 3, 1)]
            print(normalized_acc)

        elif (name == "validity"):
            # Solo: validity
            pear = pearson_per_student_formatted()
            # Pearsons scales from -1 to +1, where higher is better. 
            normalized_pear = [normalize(pear, -1, 1, 0.2)]
            nn_model(normalized_pear)

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
    grades = np.zeros(44)
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
    data = np.zeros([44, input_data.__len__()])
    i = 0
    while (i < input_data.__len__()):
        j = 0
        while (j < 44):
            # Ugly fix for that annoying reviewer 18
            if(j == 17):
                data[j, i] = input_data[i][j-2]
            else:
                data[j, i] = input_data[i][j]
            j += 1
        i += 1

    acc = get_accuracy(args.input)
    labels = np.transpose([normalize(acc, 0, 3, 1)])
    
    training_count = int(44 * 0.8)

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

##############
### Inputs ###
### Write functions here for accessing metric values (or auxilliary calculations)

#######################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--modeltype', choices=["rule","nn"],
                        help='The type of model you want to use')

    parser.add_argument('-m', '--modelname', choices=["test","accuracy","validity","reliability","sys_dev_high","sys_dev_wide","sys_dev_order","sys_dev_full","val_rel","all"], default="accuracy",
                        help='The name of the model you want to use')

    parser.add_argument('-i', '--input',
                        help='Specify a path to the data file')

    args = parser.parse_args()
    main(args)