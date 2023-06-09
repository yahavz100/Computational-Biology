import json
import numpy as np


def sigmoid(x):
    # Sigmoid activation function
    return 1 / (1 + np.exp(-x))


def runnet(wnet_file, data_file, output_file):
    # Load the network parameters from the wnet file
    with open(wnet_file, 'r') as f:
        network_params = json.load(f)
    W1 = np.array(network_params['W1'])
    b1 = np.array(network_params['b1'])
    W2 = np.array(network_params['W2'])
    b2 = np.array(network_params['b2'])

    # Load the data from the data file
    with open(data_file, 'r') as file:
        data = file.read().splitlines()

    # Perform classification for each string in the data file
    classifications = []
    for string in data:
        # Perform feedforward propagation to classify the string
        input_layer = np.array([int(char) for char in string])  # Convert the string to a binary input array
        hidden_layer = sigmoid(np.dot(input_layer, W1) + b1)
        output_layer = sigmoid(np.dot(hidden_layer, W2) + b2)

        # Determine the predicted class based on the output layer
        classification = 1 if output_layer > 0.5 else 0

        classifications.append(classification)

    # Save the classifications to the output file
    with open(output_file, 'w') as file:
        file.write('\n'.join(str(classification) for classification in classifications))


if __name__ == '__main__':
    wnet_file = "wnet0.txt"
    data_file = "testnet0.txt"
    output_file = "output0.txt"
    runnet(wnet_file, data_file, output_file)
