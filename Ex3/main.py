import numpy as np


# Step 1: Data Preparation
def load_data(file_path):
    # Load the data
    data0 = np.loadtxt(file_path, dtype=int)
    return data0[:15000], data0[15000:]


train_file = "nn0.txt"
test_file = "nn1.txt"

X_train, y_train = load_data(train_file)
X_test, y_test = load_data(test_file)


# Step 2: Define the Genetic Algorithm
population_size = 100
mutation_rate = 0.01
crossover_rate = 0.8
num_generations = 50

# Define the maximum number of layers and connections for the neural network
max_layers = 5
max_connections = 50


def sigmoid(x):
    # Apply sigmoid activation function to the input x
    # Limit the input values to avoid overflow error
    x = np.clip(x, -500, 500)  # Adjust the range as needed
    return 1 / (1 + np.exp(-x))


def initialize_weights(input_size, hidden_size, output_size):
    # Initialize random weights for the neural network
    W1 = np.random.randn(input_size, hidden_size)
    b1 = np.zeros(hidden_size)
    W2 = np.random.randn(hidden_size, output_size)
    b2 = np.zeros(output_size)
    return W1, b1, W2, b2


def forward_propagation(X, W1, b1, W2, b2):
    # Perform forward propagation through the neural network
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    return A2


def predict(network, X):
    W1, b1, W2, b2 = network

    # Forward propagation
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    predictions = sigmoid(z2)

    # Convert predictions to binary values (0 or 1)
    binary_predictions = np.where(predictions >= 0.5, 1, 0)

    return binary_predictions


# Step 3: Fitness Function
def evaluate_fitness(network, X, y):
    # Implement your neural network model here
    # Load the trained network from the best_individuals or wnet file

    # Make predictions on the training data
    predictions = predict(network, X)

    # Calculate the accuracy
    accuracy = np.count_nonzero(predictions == y) / len(y)

    return accuracy


def select_parents(population, fitness_scores):
    # Convert fitness scores to probabilities
    probabilities = fitness_scores / np.sum(fitness_scores)

    # Select parents based on their probabilities
    parents = np.random.choice(population, size=2, replace=False, p=probabilities)

    return parents


# Step 5: Crossover
def perform_crossover(parent1, parent2):
    # Select a crossover point
    crossover_point = np.random.randint(1, len(parent1))

    # Perform crossover to create offspring
    offspring = (
        np.concatenate((parent1[:crossover_point], parent2[crossover_point:])),
        np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    )

    return offspring


# Step 6: Mutation
def perform_mutation(network):
    # Mutation probability (adjust as needed)
    mutation_prob = 0.01

    # Extract the network weights
    W1, b1, W2, b2 = network

    # Mutate the network structure
    if np.random.rand() < mutation_prob:
        # Perform structure mutation (e.g., add/remove a hidden layer)

        # Example: Add a hidden layer
        # Determine the new hidden layer size
        new_hidden_size = np.random.randint(10, 100)

        # Add a new hidden layer with random weights
        W_new = np.random.randn(W1.shape[1], new_hidden_size)
        b_new = np.zeros(new_hidden_size)

        # Concatenate the new hidden layer with the existing weights
        W1 = np.concatenate((W1, W_new), axis=1)
        b1 = np.concatenate((b1, b_new))

    # Mutate the network weights
    if np.random.rand() < mutation_prob:
        # Perform weight mutation (e.g., add small random noise to the weights)
        noise_scale = 0.01  # Adjust the noise scale as needed

        # Add small random noise to the network weights
        W1 += np.random.randn(*W1.shape) * noise_scale
        b1 += np.random.randn(*b1.shape) * noise_scale
        W2 += np.random.randn(*W2.shape) * noise_scale
        b2 += np.random.randn(*b2.shape) * noise_scale

    # Construct the mutated network
    mutated_network = (W1, b1, W2, b2)

    return mutated_network


# Step 7: Repeat Steps 3-6
def evolve_population(population, X_train, y_train):
    fitness_scores = []
    for network in population:
        fitness = evaluate_fitness(network, X_train, y_train)
        fitness_scores.append(fitness)

    next_generation = []
    for _ in range(len(population)):
        parent1, parent2 = select_parents(population, fitness_scores)
        offspring = perform_crossover(parent1, parent2)
        offspring = perform_mutation(offspring)
        next_generation.append(offspring)

    return next_generation


def initialize_population(population_size, network_shape):
    population = []
    for _ in range(population_size):
        # Initialize network weights randomly
        W1 = np.random.randn(network_shape[0], network_shape[1])
        b1 = np.zeros(network_shape[1])
        W2 = np.random.randn(network_shape[1], network_shape[2])
        b2 = np.zeros(network_shape[2])

        # Create a network tuple
        network = (W1, b1, W2, b2)

        # Add the network to the population
        population.append(network)

    return population


# Step 8: Termination (Using a fixed number of generations)
def genetic_algorithm(X, y, population_size, mutation_rate, crossover_rate, num_generations):
    network_shape = (X.shape[1], 10, 1)  # Adjust the hidden layer size as needed
    population = initialize_population(population_size, network_shape)

    for generation in range(num_generations):
        print("Generation:", generation + 1)
        population = evolve_population(population, X_train, y_train)

    return population


# Step 9: Retrieve the Best Individual
def select_best_network(individuals, X_train, y_train):
    best_network = None
    best_fitness = -1

    for network in individuals:
        fitness = evaluate_fitness(network, X_train, y_train)
        if fitness > best_fitness:
            best_fitness = fitness
            best_network = network

    return best_network


# best_network = select_best_network(best_individuals, X_train, y_train)
best_individuals = genetic_algorithm(X_train, y_train, population_size, mutation_rate, crossover_rate, num_generations)
best_network = select_best_network(best_individuals, X_train, y_train)


# Step 10: Evaluate Performance on the Test Set
def evaluate_performance(network, X, y):
    # Implement your neural network model here
    # Load the trained network from the best_individuals or wnet file

    # Unpack the network weights
    W1, b1, W2, b2 = network

    # Implement the forward propagation step to make predictions on the test set
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    # Apply a threshold of 0.5 to convert predictions to binary values
    binary_predictions = (A2 >= 0.5).astype(int)

    # Calculate and return performance metrics (e.g., accuracy, precision, recall, F1 score)
    accuracy = np.mean(binary_predictions == y)
    precision = np.sum(binary_predictions * y) / np.sum(binary_predictions)
    recall = np.sum(binary_predictions * y) / np.sum(y)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return accuracy, precision, recall, f1_score

test_performance = evaluate_performance(best_network, X_test, y_test)

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

    # Save the network parameters to the file using NumPy's savez function
    np.savez(file_path, **network_params)


save_network(best_network, "wnet")

