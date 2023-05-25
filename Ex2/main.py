import tkinter as tk
from ypstruct import structure
from collections import defaultdict
import numpy as np
import string

FITNESS_COUNTER = 0
ENGLISH_ALPHABET = list(string.ascii_lowercase)
NUM_GENERATIONS = 100
POPULATION_SIZE = 50
MUTATION_RATE = 0.05
CONVERGE_LIMIT = 47
LOCAL_SEARCH_ITER = 10
MODE = "Regular"


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
                         mutation_rate=MUTATION_RATE, num_generations=NUM_GENERATIONS, population_size=POPULATION_SIZE, converge_limit=CONVERGE_LIMIT,
                         num_local_oppositions=LOCAL_SEARCH_ITER, mode=MODE):
    def calculate_fitness(given_letter_freq, given_letter_pairs_freq, english_words, plain_text):
        """
        Calculate the fitness score of a decryption key.
        :param given_letter_freq: Frequency of letters in the decrypted text.
        :param given_letter_pairs_freq: Frequency of letter pairs in the decrypted text.
        :param english_words: Set of English words.
        :param plain_text: Decrypted text.
        :return: Fitness score of the decryption key.
        """
        global FITNESS_COUNTER
        FITNESS_COUNTER += 1
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
        if np.argwhere(np.random.rand(*child_1.sequence.shape) <= MUTATION_RATE).size:
            for i in np.argwhere(np.random.rand(*child_1.sequence.shape) <= MUTATION_RATE):
                # Swap the letters at the mutation positions
                new_key_child_1.sequence[i.item()], new_key_child_1.sequence[letter1] = \
                    new_key_child_1.sequence[letter1], new_key_child_1.sequence[i.item()]

        # Mutate child 2
        if np.argwhere(np.random.rand(*child_2.sequence.shape) <= MUTATION_RATE).size:
            for i in np.argwhere(np.random.rand(*child_2.sequence.shape) <= MUTATION_RATE):
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
        code_dict = dict(zip(ENGLISH_ALPHABET, code))

        # Create a mapping dictionary with default values as the original letters
        mapp = {letter: code_dict.get(letter, letter) for letter in ENGLISH_ALPHABET}

        # Translate the encrypted text using the mapping dictionary
        decrypted_text = encrypted_text.translate(str.maketrans(mapp))

        return decrypted_text

    def local_search(individual, num_swaps, ga_mode):
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
                individual.fitness = new_fitness

                if MODE == "lamarck":
                    individual.sequence = sequence

    empty_individual = structure()
    empty_individual.sequence = None
    empty_individual.fitness = None

    # Initialize the best_key with an empty individual
    best_key = empty_individual.deepcopy()
    best_key.fitness = -np.inf

    # Create a breeding population of empty individuals
    breeding_population = empty_individual.repeat(POPULATION_SIZE)

    # Generate initial population and calculate their fitness scores
    for i in range(POPULATION_SIZE):
        # Generate a random permutation sequence as the individual's sequence
        breeding_population[i].sequence = np.random.permutation(np.array(ENGLISH_ALPHABET).copy())

        # Decrypt the text using the individual's sequence
        plain_text = decrypt_text(breeding_population[i].sequence)

        # Calculate the fitness of the individual
        breeding_population[i].fitness = calculate_fitness(given_letter_freq, given_letter_pairs_freq,
                                                           words, plain_text)

        # Update the best_key if the individual has higher fitness
        if breeding_population[i].fitness > best_key.fitness:
            best_key = breeding_population[i].deepcopy()

    # Initialize arrays to store fitness scores and average costs for each generation
    fitness_scores = np.empty(NUM_GENERATIONS)
    avg = np.empty(NUM_GENERATIONS)

    # Initialize variable for convergence check
    consecutive_generations_without_improvement = 0
    previous_best_fitness = best_key.fitness

    # Perform the evolution process for the specified number of generations
    for Generation in range(NUM_GENERATIONS):
        costs = np.array([x.fitness for x in breeding_population])
        average_cost = np.mean(costs)

        # Normalize costs to calculate selection probabilities
        normalized_costs = costs / average_cost if average_cost != 0 else costs
        probs = np.exp(-normalized_costs)
        probs /= np.sum(probs)

        avg[Generation] = average_cost

        print(f"Generation : {Generation}")

        population = []
        range_population_size = int(np.round(POPULATION_SIZE) * 2) // 2

        # Generate new individuals through crossover and mutation
        for _ in range(range_population_size):
            # Perform roulette wheel selection to choose parents
            parent1, parent2 = breeding_population[roulette_wheel_selection(probs)], breeding_population[
                roulette_wheel_selection(probs)]

            # Perform crossover to create two children
            child1, child2 = crossover(parent1, parent2)

            # Perform mutation on the children
            child1, child2 = mutation_function(child1, child2, MUTATION_RATE)

            if MODE == "Darwin":
                # Perform Darwinian optimization mode
                # Perform local optimization on the children
                local_search(child1, LOCAL_SEARCH_ITER, MODE)
                local_search(child2, LOCAL_SEARCH_ITER, MODE)

            elif MODE == "Lamarck":
                # Perform Lamarck optimization mode
                # Update the children's sequences using their fitness scores
                local_search(child1, LOCAL_SEARCH_ITER, MODE)
                local_search(child2, LOCAL_SEARCH_ITER, MODE)

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
        breeding_population = breeding_population[:POPULATION_SIZE]

        # Check for convergence
        if best_key.fitness <= previous_best_fitness:
            consecutive_generations_without_improvement += 1
        else:
            consecutive_generations_without_improvement = 0

        if consecutive_generations_without_improvement >= CONVERGE_LIMIT:
            break

        previous_best_fitness = best_key.fitness

        # Store the fitness score of the best individual in each generation
        fitness_scores[Generation] = best_key.fitness

    # Return the best decryption key, fitness scores, and average costs
    return best_key.sequence, fitness_scores, avg


def create_plain_and_perm_files(key, encrypted_text):
    """
    Create 'plain.txt' and 'perm.txt' files based on the decryption key and encrypted text.

    :param key: Decryption key.
    :param encrypted_text: Encrypted text.
    """

    with open("plain.txt", 'w') as file:
        file.write('\n'.join([f"{ENGLISH_ALPHABET[i]} {key[i]}"
                              for i in range(len(ENGLISH_ALPHABET))]))

    mapping = dict(zip(ENGLISH_ALPHABET, key))
    decoded_text = encrypted_text.translate(str.maketrans(mapping))
    with open("perm.txt", 'w') as file:
        file.write(decoded_text)


class UpdateValuesScreen(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

        # Create a label for the title text
        self.title_label = tk.Label(self, text="Genetic Algorithm", font=("Helvetica", 24, "bold"), pady=20)

        # Create labels and entry fields for each input value
        labels = ["Generations", "Population size", "Mutation rate", "Convergence limit", "Local search"]
        default_vals = [NUM_GENERATIONS, POPULATION_SIZE, MUTATION_RATE, CONVERGE_LIMIT, LOCAL_SEARCH_ITER]

        self.entries = []
        for i, label in enumerate(labels):
            label_text = f"Enter {label} value:"
            label = tk.Label(self, text=label_text, font=("Helvetica", 14), padx=20, pady=10)
            label.grid(row=i + 1, column=0, sticky="w")
            entry = tk.Entry(self, font=("Helvetica", 14), width=10)
            entry.grid(row=i + 1, column=1, padx=20, pady=10)
            entry.insert(0, str(default_vals[i]))
            self.entries.append(entry)

        # Create a button to update the values and display the plot
        self.regular_button = tk.Button(self, text="Regular", font=("Helvetica", 14), command=lambda: self.update_values("Regular"))
        self.darwin_button = tk.Button(self, text="Darwin", font=("Helvetica", 14), command=lambda: self.update_values("Darwin"))
        self.lamarck_button = tk.Button(self, text="Lamarck", font=("Helvetica", 14), command=lambda: self.update_values("Lamarck"))

        # Layout the widgets using grid
        self.title_label.grid(row=0, column=0, columnspan=2)
        self.regular_button.grid(row=len(labels) + 2, column=0, columnspan=1, pady=20, sticky="s")
        self.darwin_button.grid(row=len(labels) + 2, column=1, columnspan=2, pady=20, padx=20, sticky="s")
        self.lamarck_button.grid(row=len(labels) + 2, column=2, columnspan=3, pady=20, padx=20, sticky="s")

        # Set the last row and column to have a weight of 1
        self.grid_rowconfigure(len(labels) + 2, weight=1)
        self.grid_columnconfigure(2, weight=1)

    def update_values(self, button_name):
        global NUM_GENERATIONS, POPULATION_SIZE, MUTATION_RATE, CONVERGE_LIMIT, LOCAL_SEARCH_ITER, MODE
        # Get the values entered by the user
        NUM_GENERATIONS = int(self.entries[0].get())
        POPULATION_SIZE = int(self.entries[1].get())
        MUTATION_RATE = float(self.entries[3].get())
        CONVERGE_LIMIT = int(self.entries[4].get())
        LOCAL_SEARCH_ITER = int(self.entries[5].get())

        if button_name == "Darwin":
            MODE = "Darwin"
        elif button_name == "Lamarck":
            MODE = "Lamarck"
        else:
            MODE = "Regular"

        self.parent.destroy()


if __name__ == '__main__':
    eng_words = load_english_words('dict.txt')
    given_letter_freqs = load_frequencies('Letter_Freq.txt', False)
    given_letter_pairs_freqs = load_frequencies('Letter2_Freq.txt', True)
    enc_text = load_text('enc.txt')
    english_letters_alph = np.array(ENGLISH_ALPHABET)

    root = tk.Tk()
    root.geometry("800x800")
    root.resizable(True, True)
    update_screen = UpdateValuesScreen(root)
    update_screen.pack()

    # Update the window to show the contents
    root.update()

    # Start the main event loop
    root.mainloop()

    # sol_key, fitness_scores, avg_fitness = optimize_key_fitness(enc_text, given_letter_freqs,
    #                                                             given_letter_pairs_freqs, eng_words)
    # create_plain_and_perm_files(sol_key, enc_text)
    # print('Fitness counter:', FITNESS_COUNTER)
    # plt.plot(fitness_scores, avg_fitness, marker='o')
    # plt.xlabel('Fitness Scores')
    # plt.ylabel('Average')
    # plt.title('Average in Relation to Fitness Scores')
    # plt.grid(True)
    # plt.show()
