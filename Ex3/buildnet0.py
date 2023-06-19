import numpy as np


# Step 1: Data Preparation
def load_data(file_path):
    data_pairs = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line and len(line) >= 17:
                features = line[:16]
                label = int(line[16:].strip())
                data_pairs.append((features, label))

    return data_pairs


def sigmoid(x):
    # Apply sigmoid activation function to the input x
    # Limit the input values to avoid overflow error
    x = np.clip(x, -500, 500)  # Adjust the range as needed
    return 1 / (1 + np.exp(-x))


def predict(network, X):
    W1, b1, W2, b2 = network  # Unpack the network values
    # Convert input data to float data type
    X = X.astype(float)
    # Perform the prediction using the weights and biases
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    return A2


# Step 3: Fitness Function
def calculate_fitness(network, train_set):
    correct_predictions = 0

    for input_data, target_output in train_set:
        input_data = np.array(input_data, dtype=float)  # Convert input_data to numeric array
        predicted_output = predict(network, input_data)
        predicted_class = np.argmax(predicted_output)
        target_class = np.argmax(target_output)

        if predicted_class == target_class:
            correct_predictions += 1

    accuracy = correct_predictions / len(train_set)
    fitness = accuracy  # Use accuracy as the fitness metric

    return fitness


def select_parents(population, fitness_scores):
    population_array = np.array(population, dtype=object)
    fitness_scores_array = np.array(fitness_scores)

    probabilities = fitness_scores_array / np.sum(fitness_scores_array)

    if np.isnan(probabilities).any() or np.sum(fitness_scores_array) == 0:
        probabilities = np.ones(len(population)) / len(population)

    parent_indices = np.random.choice(len(population), size=2, replace=False, p=probabilities)

    parent1 = population_array[parent_indices[0]].tolist()
    parent2 = population_array[parent_indices[1]].tolist()

    return parent1, parent2


# Step 5: Crossover
def perform_crossover(parent1, parent2, crossover_rate):
    offspring = []
    for gene1, gene2 in zip(parent1, parent2):
        gene1 = np.array(gene1)
        gene2 = np.array(gene2)

        if np.random.rand() < crossover_rate:
            # Perform crossover between gene1 and gene2
            crossover_point = np.random.randint(1, len(gene1) + 1)
            new_gene = np.concatenate((gene1[:crossover_point], gene2[crossover_point:]))
        else:
            new_gene = gene1.copy()

        offspring.append(new_gene)

    return offspring


# Step 6: Mutation
def perform_mutation(network, mutation_rate):
    # Extract the network weights
    W1, b1, W2, b2 = network

    # Mutate the network structure
    if np.random.rand() < mutation_rate:
        # Perform structure mutation (e.g., add/remove a hidden layer)

        # Determine the new hidden layer size
        new_hidden_size = np.random.randint(10, 100)

        # Transpose W1
        W1 = W1.T

        # Add a new hidden layer with random weights
        W_new = np.random.randn(new_hidden_size, W1.shape[1])
        b_new = np.zeros(new_hidden_size)

        # Concatenate the new hidden layer with the existing weights
        W1 = np.concatenate((W1, W_new.T), axis=1)
        b1 = np.concatenate((b1, b_new))

    # Mutate the network weights
    if np.random.rand() < mutation_rate:
        # Perform weight mutation (e.g., add small random noise to the weights)
        noise_scale = 0.01  # Adjust the noise scale as needed

        # Add small random noise to the network weights
        W1 += np.random.randn(*W1.shape) * noise_scale
        b1 += np.random.randn(*b1.shape) * noise_scale
        W2 += np.random.randn(*W2.shape) * noise_scale
        b2 += np.random.randn(*b2.shape) * noise_scale

    # Construct the mutated network
    mutated_network = [W1, b1, W2, b2]

    return mutated_network


# Step 7: Repeat Steps 3-6
def evolve_population(population, train_set, mutation_rate, crossover_rate):
    fitness_scores = []
    for network in population:
        fitness = calculate_fitness(network, train_set)
        if fitness is not None:
            fitness_scores.append(fitness)

    next_generation = []
    for _ in range(len(population)):
        parent1, parent2 = select_parents(population, fitness_scores)
        offspring = perform_crossover(parent1, parent2, crossover_rate)
        offspring = perform_mutation(offspring, mutation_rate)
        next_generation.append(offspring)

    return next_generation


def initialize_population(size_of_population):
    population = []
    for _ in range(size_of_population):
        # Define the network architecture
        input_size = 16
        hidden_size = 10  # one layer
        output_size = 1

        # Initialize weights and biases
        W1 = np.random.randn(input_size, hidden_size)
        b1 = np.zeros(hidden_size)
        W2 = np.random.randn(hidden_size, output_size)
        b2 = np.zeros(output_size)

        # Create the network variable
        network = [np.array(W1), np.array(b1), np.array(W2), np.array(b2)]

        # Add the network to the population
        population.append(network)

    return population


# Step 8: Termination (Using a fixed number of generations)
def genetic_algorithm(data_train, size_of_population, mutation_rate, crossover_rate, num_generations):
    population = initialize_population(size_of_population)

    for generation in range(num_generations):
        print("Generation:", generation + 1)
        population = evolve_population(population, data_train, mutation_rate, crossover_rate)

    return population


# Step 9: Retrieve the Best Individual
def select_best_network(individuals, train_set):
    best_network = None
    best_fitness = -1

    for network in individuals:
        fitness = calculate_fitness(network, train_set)
        if fitness > best_fitness:
            best_fitness = fitness
            best_network = network

    return best_network


# Step 11: Save the Trained Network
def save_network(network, file_path):
    # Save the network structure and weights to a file

    # Unpack the network weights
    W1, b1, W2, b2 = network

    # Create a dictionary to store the network parameters
    network_params = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }

    # Save the network parameters to the file using NumPy's saves function
    np.savez(file_path, **network_params)


if __name__ == '__main__':
    train_file = "nn0.txt"
    data = load_data(train_file)

    # Split the data into training and test sets
    train_size = 15000
    X_train, X_test = data[:train_size], data[train_size:]

    # Step 2: Define the Genetic Algorithm
    population_size = 100
    mutation_rate = 0.01
    crossover_rate = 0.8
    num_generations = 50

    # Define the maximum number of layers and connections for the neural network
    # max_layers = 5
    # max_connections = 50

    best_individuals = genetic_algorithm(X_train, population_size, mutation_rate, crossover_rate, num_generations)
    best_network = select_best_network(best_individuals, X_train)

    save_network(best_network, "wnet0")

# 1. Data Preparation:
#    - Load the data from the files nn0.txt and nn1.txt.
#    - Each file contains 20,000 binary strings followed by a digit indicating legality.
#    - Split the data into a training set and a test set according to your desired ratio.
# 2. Genetic Algorithm:
#    - Define the structure of the neural network: the number of layers, number of neurons in each layer,
#    and the connections between them.
#    - Initialize a population of neural networks with random weights and structure.
#    - Evaluate the fitness of each network in the population by training and testing them on the provided data.
#    - Select the best-performing networks based on their fitness for reproduction.
#    - Apply genetic operators such as crossover and mutation to generate a new population of networks.
#    - Repeat the evaluation, selection, and reproduction steps for a certain number of generations or
#    until convergence criteria are met.
# 3. Neural Network Training:
#    - Implement a training algorithm for the neural network using a supervised learning method such as backpropagation.
#    - Use the training set to update the weights of the network iteratively.
#    - Validate the network's performance on the test set during training to monitor overfitting and generalization.
# 4. Save the Best Network:
#    - After the genetic algorithm completes, select the best-performing network from the final population based
#    on its fitness.
#    - Save the network's structure and weights to a file (e.g., wnet) for future use.
# It's important to note that implementing a genetic algorithm for neural network training can be computationally
# expensive and time-consuming. You may need to fine-tune the parameters of the genetic algorithm and experiment with
# different network structures to achieve optimal results.