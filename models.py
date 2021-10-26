import math
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
import variability
from scipy.stats import trim_mean

#from neural_network import neural_network_model

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
##############

##############
### MODELS ###
### Write functions for calculating a certain model here. Use methods below for accessing metrics

# The simplest model, only based on accuracy
def r_simple_model(input_path):
    score = []
    model = accuracy(input_path)
    for reviewer_score in model:
        score.append(max(1, min(10, (10 - (reviewer_score - 0.5) * 5))))
    return score

# Simple neural network model, accuracy as labels, variability and grades(?) as input
def nn_variability_model(input_path):
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

# Takes two arguments: array of reviewer grades per presentation
# And an array of true trades
# Returns an array with a simple accuracy measure
def accuracy(input_path, split_topics=False):
    reviewer_grade_sets = get_reviewer_grade_sets(input_path) # currently NUMPY array with shape(nr_reviewers, nr_topics, nr_rubrics)
    true_grade_sets = get_true_grade_sets(input_path)
    nr_topics = 22  # should be more dynamic
    nr_rubrics = 8  # should be more dynamic
    nr_reviewers = reviewer_grade_sets.__len__()

    total_grades_counted = 0
    total_grades_difference = np.zeros(nr_rubrics)
    total_topic_accuracy = np.zeros(shape=(nr_reviewers, nr_topics))
    total_accuracy = np.zeros(nr_reviewers)

    i = 0
    while (i < nr_reviewers):
        # Loop over all reviewers
        j = 0
        while (j < nr_topics):
            # Loop over all presentations
            h = 0
            while (h < nr_rubrics):
                if (math.isnan(reviewer_grade_sets[i][j][h])):
                    # Loop over all rubrics
                    h += 1
                    continue
                # TODO: The next part calculates the accuracy in a very simple way, might need to update this later
                total_grades_difference[h] = np.abs(true_grade_sets[j][h] - reviewer_grade_sets[i][j][h])
                total_grades_counted += 1
                h += 1

            if total_grades_counted == 0:
                total_topic_accuracy[i][j] = math.nan
            else:
                total_topic_accuracy[i][j] = np.nanmean(total_grades_difference)  # 8 = nr of rubrics
            print(total_topic_accuracy[i])

            total_grades_difference = np.zeros(nr_rubrics)
            total_grades_counted = 0
            j += 1

        total_accuracy[i] = np.nanmean(total_topic_accuracy[i])
        i += 1

    if split_topics:
        return total_topic_accuracy
    return total_accuracy

# Returns the true grades as given by the teacher.
# Returns an array of lists, where each list contains the grades given on the eight rubrics. 
def get_true_grade_sets(input_path):
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
    # print(true_grades)
    return true_grades

# Returns a list of arrays of lists contain the grades given by each reviewer for each presentation
# Basically for each reviewer contains a structure similar to true grades
def get_reviewer_grade_sets(input_path):
    data_dict = pd.read_excel(input_path, None)
    
    reviewer_grade_sets = np.zeros(shape=(44, 22, 8)) # nr of reviewers, topics, rubrics

    for topic in range(1, 23):
        tab_name = 'topic' + str(topic)
        df = data_dict[tab_name]

        reviewer_nr = 1
        while reviewer_nr <=  44:  #df.__len__():
            grades = []
            localdf = df[df["User"] == reviewer_nr]

            for rubric in range(1, 9):
                grade_to_add = localdf['Grade' + str(rubric)].tolist()

                # Add nan if no grade was assigned for this presentation/topic
                if (grade_to_add.__len__() == 0):
                    grades.append(math.nan)
                else:
                    grades.append(grade_to_add[0])

            reviewer_grade_sets[reviewer_nr-1][topic-1] = grades # bye bye .append
            reviewer_nr += 1

    return reviewer_grade_sets


def plot_accuracy(input_path):
    out = accuracy(input_path)
    users = [i for i in range(1, len(out) + 1)]
    plt.bar(users, out)
    plt.title('Bar plot of inaccuracy for each peer reviewer')  # TODO: atm inaccuracy function
    plt.xlabel('ID of peer reviewer')
    plt.ylabel('Inaccuracy')  # see the to do above
    plt.show()


def plot_accuracy_topics(input_path):
    # to change to accuracy over time [simply by looking at topic IDs becoming higher]
    out = accuracy(input_path, split_topics=True)
    topics = [i for i in range(1, 22+1)]  # 22 = nr of topics, not dynamic yet...
    for reviewer in range(0, len(out)):
        plt.plot(topics, out[reviewer], 'o-')
    plt.title('Bar plot of inaccuracy for each peer reviewer')  # see the to do above
    plt.xlabel('ID of topic')
    plt.ylabel('Inaccuracy')  # see the to do above
    plt.grid(True)
    plt.show()


def plot_accuracy_per_topic(input_path):
    # to change to accuracy over time [simply by looking at topic IDs becoming higher]
    out = [trim_mean(revs_accs[~np.isnan(revs_accs)], 0.1) for revs_accs in np.transpose(accuracy(input_path, split_topics=True))]
    topics = [i for i in range(1, 22+1)]  # 22 = nr of topics, not dynamic yet...
    plt.plot(topics, out, 'o-')
    plt.title('Bar plot of inaccuracy for trimmed average of peer reviewers')  # see the to do above
    plt.xlabel('ID of topic')
    plt.ylabel('Inaccuracy')  # see the to do above
    plt.grid(True)
    plt.show()


def plot_accuracy_nr_topics(input_path):
    # to change to accuracy over time [by looking at increasing number of topics reviewed]
    out = accuracy(input_path, split_topics=True)
    topics = [i for i in range(1, 22+1)]  # 22 = nr of topics, not dynamic yet...
    for reviewer in range(0, len(out)):
        reviewed_topics = [acc for acc in out[reviewer] if not math.isnan(acc)]
        accuracies = np.empty(len(topics))
        accuracies[:] = np.nan
        accuracies[:len(reviewed_topics)] = reviewed_topics
        print(accuracies)
        plt.plot(topics, accuracies, 'o-')
    plt.title('Bar plot of inaccuracy for each peer reviewer')  # see the to do above
    plt.xlabel('Number of presentations reviewed')
    plt.ylabel('Inaccuracy')  # see the to do above
    plt.grid(True)
    plt.show()

##############
##############

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