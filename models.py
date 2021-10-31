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
            print(np.sort(normalized_acc))

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

        elif (name == "validity"):
            # Solo: validity
            print()

        elif (name == "reliability"):
            # Solo: reliability
            print()
            
        elif (name == "sys_dev_high"):
            # Solo: systematic deviation: high/low
            print()
        
        elif (name == "sys_dev_wide"):
            # Solo: systematic deviation: broad/narrow
            print()
            
        elif (name == "sys_dev_order"):
            # Solo: systematic deviation: relative ordering
            print()
        
        elif (name == "sys_dev_full"):
            # Combined: all systematic deviation metrics
            print()
        
        elif (name == "val_rel"):
            # Combined: validity and reliability
            print()
        
        elif (name == "all"):
            # Combined: # of reviews handed in, length of comments and consistency (variability of grades of a single student)
            print()
            
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
def nn_variability_model(input_path: str):
    # Input: 
    # - Variability statistics
    # - All grades?
    # Labels: Accuracy?
    var_dict = variability.compute_variability_statistics()
    print(var_dict)
    # TODO: process input data
    training_data = 0
    training_labels = 0

    test_data = 0
    test_labels = 0

    data = 0

# DISABLE IF YOU GET ERROR: CORE DUMPED
    #model = neural_network_model(args.model, 4)
    #model.train(training_data, training_labels, test_data, test_labels, 64, 8)
    #predicted_grade = model.predict(data)

    #print(predicted_grade)

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