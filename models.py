import numpy, argparse
from neural_network import neural_network_model

# Use: python models.py -m test
# to run the NN

def main(args):
    # TODO: process input data
    training_data = 0
    training_labels = 0

    test_data = 0
    test_labels = 0

    data = 0

    model = neural_network_model(args.model, 4)
    model.train(training_data, training_labels, test_data, test_labels, 64, 8)
    predicted_grade = model.predict(data)

    print(predicted_grade)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=["test"], default="test",
                        help='The name of the model you want to use')

    args = parser.parse_args()
    main(args)