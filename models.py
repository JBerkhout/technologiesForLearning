import math
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
        elif (name == "simple"):
            print(simple_model(args.input))
        else:
            print("Model name not found")

    # If not, did we want a neural network based one?
    elif(args.modeltype == "nn"):
        if (name == "test"):
            # Run model named "test"
            print("Not implemented yet")
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
def simple_model(input_path):
    score = []
    model = accuracy(input_path)
    for reviewer_score in model:
        score.append(max(1, min(10, (10 - (reviewer_score - 0.5) * 5))))
    return score


def run_nn(args):
    # TODO: process input data
    training_data = 0
    training_labels = 0

    test_data = 0
    test_labels = 0

    data = 0

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
def accuracy(input_path):
    reviewer_grade_sets = get_reviewer_grade_sets(input_path) # currently NUMPY array with shape(nr_reviewers, nr_topics, nr_rubrics)
    true_grade_sets = get_true_grade_sets(input_path)

    total_grades_counted = np.zeros(22)
    total_grades_difference = np.zeros(22)
    total_accuracy = np.zeros(22)

    i = 0
    while (i < reviewer_grade_sets.__len__()):
        # Loop over all reviewers
        j = 0
        while (j < true_grade_sets.__len__()):
            # Loop over all presentations
            h = 0
            while (h <= 7):
                if (math.isnan(reviewer_grade_sets[i][j][h])):
                    # Loop over all rubrics
                    h += 1
                    continue
                # TODO: The next part calculates the accuracy in a very simple way, might need to update this later
                total_grades_difference[j] += np.abs(true_grade_sets[j][h] - reviewer_grade_sets[i][j][h])
                total_grades_counted[j] += 1
                h += 1
            if(total_grades_counted[j] == 0):
                total_accuracy[j] = math.nan
            else:
                total_accuracy[j] = total_grades_difference[j] / total_grades_counted[j]
            j += 1
        i += 1
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
    plt.title('Bar plot of accuracy for each peer reviewer')
    plt.xlabel('ID of peer reviewer')
    plt.ylabel('Accuracy')
    plt.show()


##############
##############

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--modeltype', choices=["rule","nn"],
                        help='The type of model you want to use')

    parser.add_argument('-m', '--modelname', choices=["test","simple"], default="test",
                        help='The name of the model you want to use')

    parser.add_argument('-i', '--input',
                        help='Specify a path to the data file')

    args = parser.parse_args()
    main(args)