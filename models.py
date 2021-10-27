import math
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
import variability
from accuracy import get_accuracy

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
            print(r_simple_model(args.input))

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
        
        elif (name == "geyser"):
            # Combined: # of reviews handed in, length of comments and consistency (variability of grades of a single student)
            print()
        
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
        
        elif (name == "geyser"):
            # Combined: # of reviews handed in, length of comments and consistency (variability of grades of a single student)
            print()
            
        else:
            print("Model name not found")
    else:
        print("No such model type")


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

    parser.add_argument('-m', '--modelname', choices=["test","accuracy","validity","reliability","sys_dev_high","sys_dev_wide","sys_dev_order","sys_dev_full","val_rel","geyser"], default="accuracy",
                        help='The name of the model you want to use')

    parser.add_argument('-i', '--input',
                        help='Specify a path to the data file')

    args = parser.parse_args()
    main(args)