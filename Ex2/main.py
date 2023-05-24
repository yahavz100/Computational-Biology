import time
from ypstruct import structure
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

fitness_counter = 0


def load_text(filename: str) -> str:
    """
    Load text from a file and return it as a string.
    """
    with open(filename, 'r') as file:
        text = file.read()
    return text


def load_frequencies(filename: str, is_pair: bool) -> dict:
    """
    Load letter or letter pair frequencies from a file and return them as a dictionary.
    """
    freq_dict = defaultdict(float)
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                if line == '#REF!':
                    continue
                freq, item = line.split()
                dict_key = item.lower() if is_pair else item.lower()
                freq_dict[dict_key] = float(freq)

    return freq_dict


def load_english_words(filename: str) -> set:
    """
    Load a set of English words from a file and return it.
    """
    english_words = set()
    with open(filename, 'r') as file:
        for line in file:
            word = line.strip().lower()
            english_words.add(word)
    return english_words


def optimize_key_fitness(encrypted_text, given_letter_freq, given_letter_pairs_freq, words,
                         mutation_rate=0.05, num_generations=100, population_size=50, converge_limit=47,
                         num_local_oppositions=10, mode="regular"):
    def calculate_fitness(given_letter_freq, given_letter_pairs_freq, english_words, plain_text):
        """
        Calculate the fitness score of a decryption key.
        :param given_letter_freq: Frequency of letters in the decrypted text.
        :param given_letter_pairs_freq: Frequency of letter pairs in the decrypted text.
        :param english_words: Set of English words.
        :param plain_text: Decrypted text.
        :return: Fitness score of the decryption key.
        """
        global fitness_counter
        fitness_counter += 1
        fitness = 0.0
        fitness += sum(1.0 for word in plain_text.lower().split() if word in english_words)
        fitness += sum(given_letter_freq[char] for char in plain_text)
        fitness += sum(given_letter_pairs_freq[plain_text[i:i + 2]] * 10 for i in range(len(plain_text) - 1))
        return fitness

    def mapping_pmx(key1, key2, a, b, child_b_sequence, crossover_point1, crossover_point2):
        """
        Perform Partially Mapped Crossover (PMX) for key sequences.
        :param key1: Parent key 1.
        :param key2: Parent key 2.
        :param a: Key sequence of parent 1.
        :param b: Key sequence of parent 2.
        :param child_b_sequence: Child key sequence.
        :param crossover_point1: First crossover point.
        :param crossover_point2: Second crossover point.
        :return: Child key sequences after PMX.
        """
        child_a_seq = a
        mapping_a = {key1.sequence[i]: b[i] for i in range(crossover_point1, crossover_point2 + 1)}
        mapping_b = {key2.sequence[i]: child_a_seq[i] for i in range(crossover_point1, crossover_point2 + 1)}

        for i in range(len(key2.sequence)):
            if i < crossover_point1 or i > crossover_point2:
                value = key2.sequence[i]
                while value in mapping_a:
                    value = mapping_a[value]
                child_a_seq[i] = value

        for i in range(len(key1.sequence)):
            if i < crossover_point1 or i > crossover_point2:
                value = key1.sequence[i]
                while value in mapping_b:
                    value = mapping_b[value]
                child_b_sequence[i] = value
        return child_a_seq, child_b_sequence

    def crossover(key1, key2):
        """
        Perform crossover between two decryption keys.
        :param key1: First parent key.
        :param key2: Second parent key.
        :return: Two child keys generated through crossover.
        """

        # Select random crossover points
        crossover_point1 = np.random.randint(1, len(key1.sequence))
        crossover_point2 = np.random.randint(crossover_point1, len(key1.sequence))

        # Create copies of key1 and key2
        child_a, child_b = key1.deepcopy(), key2.deepcopy()

        # Create temporary arrays to hold the crossover segments
        a, b = np.empty_like(key1.sequence), np.empty_like(key2.sequence)

        # Perform crossover by exchanging the segments between the parents
        a[crossover_point1:crossover_point2 + 1], b[crossover_point1:crossover_point2 + 1] = \
            key1.sequence[crossover_point1:crossover_point2 + 1], key2.sequence[crossover_point1:crossover_point2 + 1]

        # Apply position-based mapping crossover (pmx) to generate child sequences
        child_a.sequence, child_b.sequence = mapping_pmx(key1, key2, a, b, child_b.sequence, crossover_point1,
                                                         crossover_point2)

        return child_a, child_b

    def mutation_function(child_1, child_2, mutation_rate):
        """
        Perform mutation on two child keys.
        :param child_1: First child key.
        :param child_2: Second child key.
        :param mutation_rate: Rate of mutation.
        :return: Mutated child keys.
        """

        # Create copies of the children
        new_key_child_1, new_key_child_2 = child_1.deepcopy(), child_2.deepcopy()

        # Select random letters for mutation
        letter1, letter2 = np.random.randint(len(child_1.sequence)), np.random.randint(len(child_2.sequence))

        # Mutate child 1
        if np.argwhere(np.random.rand(*child_1.sequence.shape) <= mutation_rate).size:
            for i in np.argwhere(np.random.rand(*child_1.sequence.shape) <= mutation_rate):
                # Swap the letters at the mutation positions
                new_key_child_1.sequence[i.item()], new_key_child_1.sequence[letter1] = \
                    new_key_child_1.sequence[letter1], new_key_child_1.sequence[i.item()]

        # Mutate child 2
        if np.argwhere(np.random.rand(*child_2.sequence.shape) <= mutation_rate).size:
            for i in np.argwhere(np.random.rand(*child_2.sequence.shape) <= mutation_rate):
                # Swap the letters at the mutation positions
                new_key_child_2.sequence[i.item()], new_key_child_2.sequence[letter2] = \
                    new_key_child_2.sequence[letter2], new_key_child_2.sequence[i.item()]

        return new_key_child_1, new_key_child_2

    def roulette_wheel_selection(p):
        """
        Perform roulette wheel selection to get the index of an individual based on the given probabilities.

        :param p: A list of probabilities representing the individuals' fitness scores.
        :return: The index of the selected individual.
        """
        cumulative_probs = np.cumsum(p)
        total_sum = cumulative_probs[-1]
        r = np.random.uniform(0, total_sum)
        selected_index = np.argmax(r <= cumulative_probs)

        return selected_index

    def decrypt_text(code):
        """
        Decrypt the encrypted text using a given code.

        :param code: List representing the decryption code.
        :return: Decrypted text.
        """

        # Create a dictionary mapping English letters to the code letters
        code_dict = dict(zip(list('abcdefghijklmnopqrstuvwxyz'), code))

        # Create a mapping dictionary with default values as the original letters
        mapp = {letter: code_dict.get(letter, letter) for letter in list('abcdefghijklmnopqrstuvwxyz')}

        # Translate the encrypted text using the mapping dictionary
        decrypted_text = encrypted_text.translate(str.maketrans(mapp))

        return decrypted_text

    def darwin_optimization(individual, num_swaps):
        # Make a copy of the individual's sequence
        sequence = individual.sequence.copy()

        # Perform num swaps
        for _ in range(num_swaps):
            # Select two random indices to swap
            i, j = np.random.choice(len(sequence), size=2, replace=False)

            # Swap the values at the selected indices
            sequence[i], sequence[j] = sequence[j], sequence[i]

            # Calculate the fitness of the new sequence
            plain_text = decrypt_text(sequence)
            new_fitness = calculate_fitness(given_letter_freq, given_letter_pairs_freq, words, plain_text)

            # Accept the swap if it improves the fitness
            if new_fitness > individual.fitness:
                individual.sequence = sequence
                individual.fitness = new_fitness

    empty_individual = structure()
    empty_individual.sequence = None
    empty_individual.fitness = None

    # Initialize the best_key with an empty individual
    best_key = empty_individual.deepcopy()
    best_key.fitness = -np.inf

    # Create a breeding population of empty individuals
    breeding_population = empty_individual.repeat(population_size)

    # Generate initial population and calculate their fitness scores
    for i in range(population_size):
        # Generate a random permutation sequence as the individual's sequence
        breeding_population[i].sequence = np.random.permutation(np.array(list('abcdefghijklmnopqrstuvwxyz')).copy())

        # Decrypt the text using the individual's sequence
        plain_text = decrypt_text(breeding_population[i].sequence)

        # Calculate the fitness of the individual
        breeding_population[i].fitness = calculate_fitness(given_letter_freq, given_letter_pairs_freq,
                                                           words, plain_text)

        # Update the best_key if the individual has higher fitness
        if breeding_population[i].fitness > best_key.fitness:
            best_key = breeding_population[i].deepcopy()

    # Initialize arrays to store fitness scores and average costs for each generation
    fitness_scores = np.empty(num_generations)
    avg = np.empty(num_generations)

    # Initialize variable for convergence check
    consecutive_generations_without_improvement = 0
    previous_best_fitness = best_key.fitness

    # Perform the evolution process for the specified number of generations
    for Generation in range(num_generations):
        costs = np.array([x.fitness for x in breeding_population])
        average_cost = np.mean(costs)

        # Normalize costs to calculate selection probabilities
        normalized_costs = costs / average_cost if average_cost != 0 else costs
        probs = np.exp(-normalized_costs)
        probs /= np.sum(probs)

        avg[Generation] = average_cost

        print(f"Generation : {Generation}")

        population = []
        range_population_size = int(np.round(population_size) * 2) // 2

        # Generate new individuals through crossover and mutation
        for _ in range(range_population_size):
            # Perform roulette wheel selection to choose parents
            parent1, parent2 = breeding_population[roulette_wheel_selection(probs)], breeding_population[
                roulette_wheel_selection(probs)]

            # Perform crossover to create two children
            child1, child2 = crossover(parent1, parent2)

            # Perform mutation on the children
            child1, child2 = mutation_function(child1, child2, mutation_rate)

            if mode == "darwinian":
                # Perform Darwinian optimization mode
                # Perform local optimization on the children
                darwin_optimization(child1, num_local_oppositions)
                darwin_optimization(child2, num_local_oppositions)

            elif mode == "american":
                # Perform American optimization mode
                pass

            # Regular optimization mode

            # Decrypt the text using the children's sequences and calculate their fitness
            plain_text = decrypt_text(child1.sequence)
            child1.fitness = calculate_fitness(given_letter_freq, given_letter_pairs_freq, words,
                                               plain_text)

            plain_text = decrypt_text(child2.sequence)
            child2.fitness = calculate_fitness(given_letter_freq, given_letter_pairs_freq, words,
                                               plain_text)

            population.append(child1)
            population.append(child2)

            # Update the best_key if any of the children have higher fitness
            if child1.fitness > best_key.fitness:
                best_key = child1.deepcopy()
            elif child2.fitness > best_key.fitness:
                best_key = child2.deepcopy()

        # Add the new population to the breeding population and select the fittest individuals
        breeding_population += population
        breeding_population = sorted(breeding_population, key=lambda x: x.fitness, reverse=True)
        breeding_population = breeding_population[:population_size]

        # Check for convergence
        if best_key.fitness <= previous_best_fitness:
            consecutive_generations_without_improvement += 1
        else:
            consecutive_generations_without_improvement = 0

        if consecutive_generations_without_improvement >= converge_limit:
            break

        previous_best_fitness = best_key.fitness

        # Store the fitness score of the best individual in each generation
        fitness_scores[Generation] = best_key.fitness

    # Return the best decryption key, fitness scores, and average costs
    return best_key.sequence, fitness_scores, avg


def create_plain_and_perm_files(key, encrypted_text, eng_alph):
    """
    Create 'plain.txt' and 'perm.txt' files based on the decryption key and encrypted text.

    :param key: Decryption key.
    :param encrypted_text: Encrypted text.
    :param eng_alph: English alphabet set.
    """

    with open("plain.txt", 'w') as file:
        file.write('\n'.join([f"{eng_alph[i]} {key[i]}"
                              for i in range(len(eng_alph))]))

    mapping = dict(zip(eng_alph, key))
    decoded_text = encrypted_text.translate(str.maketrans(mapping))
    with open("perm.txt", 'w') as file:
        file.write(decoded_text)


if __name__ == '__main__':
    words = load_english_words('dict.txt')
    given_letter_freq = load_frequencies('Letter_Freq.txt', False)
    given_letter_pairs_freq = load_frequencies('Letter2_Freq.txt', True)
    encrypted_text = load_text('enc.txt')
    english_letters_alph = np.array(list('abcdefghijklmnopqrstuvwxyz'))
    key, fitness_scores, avg_fitness = optimize_key_fitness(encrypted_text, given_letter_freq,
                                                            given_letter_pairs_freq, words, 0.05,
                                                            100, 50, 47, 10, "regular")
    create_plain_and_perm_files(key, encrypted_text, english_letters_alph)
    print('Fitness counter:', fitness_counter)
    plt.plot(fitness_scores, avg_fitness, marker='o')
    plt.xlabel('Fitness Scores')
    plt.ylabel('Average')
    plt.title('Average in Relation to Fitness Scores')
    plt.grid(True)
    plt.show()
