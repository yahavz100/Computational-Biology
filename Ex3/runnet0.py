import numpy as np


def runnet(wnet_file, data_file, output_file):
    # Load the network parameters from the wnet file
    network_params = np.load(wnet_file)
    network_structure = dict(network_params)

    # Load the data from the data file
    with open(data_file, 'r') as file:
        data = file.read().splitlines()

    # Perform classification for each string in the data file
    classifications = []
    for string in data:
        # Perform feedforward propagation to classify the string
        input_layer = np.array([int(char) for char in string])  # Convert the string to a binary input array
        hidden_layer = input_layer
        for layer_num in range(1, len(network_structure) // 2):
            W = network_structure['W' + str(layer_num)]
            b = network_structure['b' + str(layer_num)]
            hidden_layer = np.maximum(0, np.dot(W, hidden_layer) + b)
        output_layer = np.dot(network_structure['W' + str(len(network_structure) // 2)],
                              hidden_layer) + network_structure['b' + str(len(network_structure) // 2)]

        # Determine the predicted class based on the output layer
        classification = np.argmax(output_layer)

        classifications.append(classification)

    # Save the classifications to the output file
    with open(output_file, 'w') as file:
        file.write('\n'.join(str(classification) for classification in classifications))


if __name__ == '__main__':
    wnet_file = "wnet0.npz"
    data_file = "testnet0.txt"
    output_file = "output.txt"
    runnet(wnet_file, data_file, output_file)
