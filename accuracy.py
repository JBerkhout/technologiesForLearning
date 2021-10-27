from scipy.stats import trim_mean
import math
import numpy as np
import matplotlib.pyplot as plt
from pre_processing import get_reviewer_grade_sets, get_true_grade_sets


# Takes two arguments: array of reviewer grades per presentation
# And an array of true trades
# Returns an array with a simple accuracy measure
def get_accuracy(input_path: str, split_topics=False) -> [float]:
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
    while i < nr_reviewers:
        # Loop over all reviewers
        j = 0
        while j < nr_topics:
            # Loop over all presentations
            h = 0
            while h < nr_rubrics:
                if math.isnan(reviewer_grade_sets[i][j][h]):
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
            #print(total_topic_accuracy[i])

            total_grades_difference = np.zeros(nr_rubrics)
            total_grades_counted = 0
            j += 1

        total_accuracy[i] = np.nanmean(total_topic_accuracy[i])
        i += 1

    if split_topics:
        return total_topic_accuracy
    return total_accuracy



def accuracy_per_topic(input_path: str):
    return [trim_mean(revs_accs[~np.isnan(revs_accs)], 0.1) for revs_accs in np.transpose(get_accuracy(input_path, split_topics=True))]



# Plots the general accuracy for each peer reviewer
def plot_accuracy(input_path: str) -> None:
    out = get_accuracy(input_path)
    users = [i for i in range(1, len(out) + 1)]
    plt.bar(users, out)
    plt.title('Bar plot of inaccuracy for each peer reviewer')  # TODO: atm inaccuracy function
    plt.xlabel('ID of peer reviewer')
    plt.ylabel('Inaccuracy')  # see the to do above
    plt.show()


# Plots the accuracy topic over time
def plot_accuracy_topics(input_path: str) -> None:
    # to change to accuracy over time [simply by looking at topic IDs becoming higher]
    out = get_accuracy(input_path, split_topics=True)
    topics = [i for i in range(1, 22+1)]  # 22 = nr of topics, not dynamic yet...
    for reviewer in range(0, len(out)):
        plt.plot(topics, out[reviewer], 'o-')
    plt.title('Bar plot of inaccuracy for each peer reviewer')  # see the to do above
    plt.xlabel('ID of topic')
    plt.ylabel('Inaccuracy')  # see the to do above
    plt.grid(True)
    plt.show()


# Plots the accuracy per topic [by looking at the mean accuracy of different topics]
def plot_accuracy_per_topic(input_path: str) -> None:
    # to change to accuracy over time [simply by looking at topic IDs becoming higher]
    out = [trim_mean(revs_accs[~np.isnan(revs_accs)], 0.1) for revs_accs in np.transpose(get_accuracy(input_path, split_topics=True))]
    topics = [i for i in range(1, 22+1)]  # 22 = nr of topics, not dynamic yet...
    plt.plot(topics, out, 'o-')
    plt.title('Bar plot of inaccuracy for trimmed average of peer reviewers')  # see the to do above
    plt.xlabel('ID of topic')
    plt.ylabel('Inaccuracy')  # see the to do above
    plt.grid(True)
    plt.show()


# Plot the accuracy per reviewer dependent on amount of reviews already done
def plot_accuracy_nr_topics(input_path):
    # to change to accuracy over time [by looking at increasing number of topics reviewed]
    out = get_accuracy(input_path, split_topics=True)
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