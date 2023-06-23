import numpy as np

NUM_GENERATIONS = 150
POPULATION_SIZE = 200
MUTATION_RATE = 0.16
EARLY_CONVERGE = 0.25
TRAIN_SIZE = 15000


# Data Preparation
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


# Fitness Function
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


# Crossover
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


# Mutation
def perform_mutation(network):
    # Extract the network weights
    W1, b1, W2, b2 = network

    for i in network:
        # Mutate the network weights
        if np.random.rand() < MUTATION_RATE:
            # Perform weight mutation
            i += np.random.randn(*i.shape)

    # Construct the mutated network
    mutated_network = [W1, b1, W2, b2]

    return mutated_network


def evolve_population(train_set):
    # Initialize the population
    population = initialize_population()

    # Calculate fitness scores for each network in the population
    fitness_scores = [calculate_fitness(network, train_set) for network in population]

    # Sort the population based on fitness scores in descending order
    initialized_pop = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)

    # Select the best network as the initial best network
    best_network = initialized_pop[0][0]
    best_fitness = initialized_pop[0][1]

    # Counter to keep track of generations without improvement
    generations_without_improvement = 0

    # Best fitness threshold to track improvement
    best_fitness_threshold = best_fitness

    # Iterate over the specified number of generations
    for i in range(NUM_GENERATIONS):
        print("Generation:", i + 1)

        # Create lists for the next generation
        next_generation = []
        next_generation_fitness = []

        # Generate offspring for the next generation
        for _ in range(POPULATION_SIZE):
            # Select parents for crossover
            parent1, parent2 = select_parents(population, fitness_scores)

            # Perform crossover to create offspring
            offspring = perform_crossover(parent1, parent2)

            # Perform mutation on the offspring
            offspring = perform_mutation(offspring)

            # Calculate fitness for each offspring
            offspring_fitness = calculate_fitness(offspring, train_set)

            # Add offspring and its fitness to the next generation
            next_generation.append(offspring)
            next_generation_fitness.append(offspring_fitness)

            # Update the best network if the offspring has higher fitness
            if offspring_fitness > best_fitness:
                best_fitness = offspring_fitness
                best_network = offspring

        # Check for early convergence
        if best_fitness > best_fitness_threshold:
            generations_without_improvement = 0
            best_fitness_threshold = best_fitness
        else:
            generations_without_improvement += 1

        # Check if 25% of the generations have passed without improvement
        if generations_without_improvement > (NUM_GENERATIONS // 4):
            print("Early convergence")
            return best_network

        # Combine the current population, next generation, and their fitness scores
        combined_population = population + next_generation
        combined_fitness_scores = fitness_scores + next_generation_fitness

        # Sort the combined population based on fitness scores in descending order
        combined_sorted = sorted(zip(combined_population, combined_fitness_scores), key=lambda x: x[1], reverse=True)

        # Keep only the top population size individuals for the next generation
        combined_sorted = combined_sorted[:POPULATION_SIZE]

        # Separate the sorted population and fitness scores
        sorted_population, sorted_fitness_scores = zip(*combined_sorted)

        # Convert the sorted population and fitness scores back to lists
        population = list(sorted_population)
        fitness_scores = list(sorted_fitness_scores)

    # Return the best network found
    print("best network fitness:", best_fitness)
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


# Save the Trained Network
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
    X_train, X_test = data[:TRAIN_SIZE], data[TRAIN_SIZE:]

    best_network = evolve_population(X_train)

    save_network(best_network, "wnet0")