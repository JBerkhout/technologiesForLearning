import matplotlib.pyplot as plt

from accuracy import get_accuracy
from systematic_deviation import sys_high_low_official, sys_spread_official


INPUT_PATH = "data_v2.xlsx"
COLOR = 'lightseagreen'


def plot_per_reviewer(metric):
    if metric == "inaccuracy":
        out = get_accuracy(INPUT_PATH)
    elif metric == "validity":
        print("Metrics " + metric + " has not been implemented yet...")
        return
    elif metric == "reliability":
        print("Metrics " + metric + " has not been implemented yet...")
        return
    elif metric == "systematic high/low peer bias":
        out = sys_high_low_official()
    elif metric == "systematic broad/narrow peer bias":
        out = sys_spread_official()
    elif metric == "systematic problems in ordering":
        print("Metrics " + metric + " has not been implemented yet...")
        return
    else:
        print("Metrics " + metric + " was not recognized...")
        return

    assert out.ndim == 1
    assert len(out) == 44

    users = [i for i in range(1, len(out) + 1)]
    plt.bar(users, out, color=COLOR)
    plt.title('Bar plot of ' + metric + ' for each peer reviewer')
    plt.xlabel('ID of peer reviewer')
    plt.ylabel(metric.capitalize())
    plt.grid()
    plt.show()

