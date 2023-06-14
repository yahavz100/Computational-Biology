
def run_feedforward(wnet_file, data_file):
    # return classification
    pass


def save_classifications(classification, output_file):
    with open(output_file, 'w') as file:
        for c in classification:
            file.write(str(c) + '\n')


if __name__ == '__main__':
    wnet_file, data_file, output_file = "wnet0.txt", "testnet0.txt", "output.txt"
    classification = run_feedforward(wnet_file, data_file)
    save_classifications(classification, output_file)
