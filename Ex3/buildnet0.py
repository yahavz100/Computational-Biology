import numpy as np

NUM_GENERATIONS = 10
POPULATION_SIZE = 20
MUTATION_RATE = 0.05


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
    return 1 / (1 + np.exp(-x))


def predict(network, X):
    W1, b1, W2, b2 = network  # Unpack the network values

    # Convert X to a NumPy array
    X_array = np.array(list(X), dtype=np.float32)

    # Reshape X_array if necessary
    X_array = X_array.reshape(1, -1)  # Assuming X is a single sample

    # Perform the prediction using the weights and biases
    Z1 = np.dot(X_array, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    return A2


# Step 3: Fitness Function
def calculate_fitness(network, train_set):
    correct_predictions = 0

    for input_data, target_output in train_set:
        output = predict(network, input_data)
        if output > 0.5:
            predicted_class = 1
        else:
            predicted_class = 0
        if predicted_class == target_output:
            correct_predictions += 1

    accuracy = correct_predictions / len(train_set)
    # print(accuracy)
    fitness = accuracy  # Use accuracy as the fitness metric

    return fitness


def select_parents(population, fitness_scores):
    population_array = np.array(population, dtype=object)
    fitness_scores_array = np.array(fitness_scores)

    # Calculate selection probabilities based on fitness scores
    probabilities = fitness_scores_array / np.sum(fitness_scores_array)

    if np.isnan(probabilities).any() or np.sum(fitness_scores_array) == 0:
        probabilities = np.ones(len(population)) / len(population)

    # Select parents using Roulette Wheel Selection
    parent_indices = np.random.choice(len(population), size=2, replace=True, p=probabilities)

    parent1 = population_array[parent_indices[0]].tolist()
    parent2 = population_array[parent_indices[1]].tolist()

    return parent1, parent2


# Step 5: Crossover
def perform_crossover(parent1, parent2):
    offspring = []
    for gene1, gene2 in zip(parent1, parent2):
        gene1 = np.array(gene1)
        gene2 = np.array(gene2)

        # Perform crossover between gene1 and gene2
        crossover_point = np.random.randint(1, len(gene1) + 1)
        new_gene = np.concatenate((gene1[:crossover_point], gene2[crossover_point:]))

        offspring.append(new_gene)

    return offspring


# Step 6: Mutation
def perform_mutation(network):
    # Extract the network weights
    W1, b1, W2, b2 = network
    # print(network)
    for i in network:
        # Mutate the network weights
        if np.random.rand() < MUTATION_RATE:
            # Perform weight mutation
            i += np.random.randn(*i.shape)
            # b1 += np.random.randn(*b1.shape)
            # W2 += np.random.randn(*W2.shape)
            # b2 += np.random.randn(*b2.shape)

    # Construct the mutated network
    mutated_network = [W1, b1, W2, b2]

    return mutated_network


# Step 7: Repeat Steps 3-6
def evolve_population(train_set):
    best_network = None
    best_fitness = -1
    population = initialize_population()

    fitness_scores = [calculate_fitness(network, train_set) for network in population]
    for i in range(NUM_GENERATIONS):
        print("Generation:", i + 1)
        next_generation = []
        next_generation_fitness = []

        for _ in range(POPULATION_SIZE):
            parent1, parent2 = select_parents(population, fitness_scores)
            offspring = perform_crossover(parent1, parent2)
            offspring = perform_mutation(offspring)

            # Calculate fitness for each offspring
            offspring_fitness = calculate_fitness(offspring, train_set)
            next_generation.append(offspring)
            next_generation_fitness.append(offspring_fitness)

            # Save the best network
            if offspring_fitness > best_fitness:
                best_fitness = offspring_fitness
                print(best_fitness)
                best_network = offspring

        # Combine the current population, next generation, and their fitness scores
        combined_population = population + next_generation
        combined_fitness_scores = fitness_scores + next_generation_fitness

        # Sort the combined population based on fitness scores in descending order
        combined_sorted = sorted(zip(combined_population, combined_fitness_scores), key=lambda x: x[1], reverse=True)
        combined_sorted = combined_sorted[:POPULATION_SIZE]

        # Separate the sorted population and fitness scores
        sorted_population, sorted_fitness_scores = zip(*combined_sorted)

        population = list(sorted_population)
        fitness_scores = list(sorted_fitness_scores)
    print(best_fitness)
    return best_network



def initialize_population():
    population = []
    for _ in range(POPULATION_SIZE):
        # Define the network architecture
        input_size = 16
        hidden_size = 10
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
# def genetic_algorithm(data_train):
#     population = initialize_population()
#
#     for generation in range(NUM_GENERATIONS):
#         print("Generation:", generation + 1)
#         population = evolve_population(population, data_train)
#
#     return population


# Step 9: Retrieve the Best Individual
# def select_best_network(individuals, train_set):
#     best_network = None
#     best_fitness = -1
#
#     for network in individuals:
#         fitness = calculate_fitness(network, train_set)
#         if fitness > best_fitness:
#             best_fitness = fitness
#             best_network = network
#
#     return best_network


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

    best_network = evolve_population(X_train)

    save_network(best_network, "wnet0")
    print("done")

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
