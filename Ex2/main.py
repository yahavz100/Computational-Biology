
import time

from ypstruct import structure
from collections import defaultdict
import numpy as np


def load_text(filename: str) -> str:
    """
    Load text from a file and return it as a string.
    """
    with open(filename, 'r') as file:
        text = file.read()
    return text



def load_letter_frequencies(file_path):
    with open(file_path, 'r') as file:
        freqs = defaultdict(float)
        for line in file:
            values = line.strip().split('\t')
            if len(values) == 2:
                freq, char = values
                freqs[char.lower()] = float(freq)
    return freqs



def load_letter_pair_frequencies(file_path):
    with open(file_path, 'r') as file:
        freqs = defaultdict(float)
        for line in file:

            values = line.strip().split('\t')
            if len(values) == 2:
                freq, pair = values
                freqs[pair.lower()] = float(freq)
    return freqs


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


def optimize_key_fitness(encrypted_text, given_letter_freq, given_letter_pairs_freq, words, fitness_counter,
                         mutation_rate=0.05, num_generations=100, population_size=50):
    def calculate_fitness(fitness_counter, given_letter_freq, given_letter_pairs_freq, english_words, plain_text):
        fitness_counter += 1
        fitness = 0.0
        fitness += sum(1.0 for word in plain_text.lower().split() if word in english_words)
        fitness += sum(given_letter_freq[char] for char in plain_text)
        fitness += sum(given_letter_pairs_freq[plain_text[i:i + 2]] * 10 for i in range(len(plain_text) - 1))
        return fitness

    def Mapping_PMC(key1, key2, a, b, child_b_sequence, crossover_point1, crossover_point2):
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
        crossover_point1 = np.random.randint(1, len(key1.sequence))
        crossover_point2 = np.random.randint(crossover_point1, len(key1.sequence))
        child_a, child_b = key1.deepcopy(), key2.deepcopy()
        a, b = np.empty_like(key1.sequence), np.empty_like(key2.sequence)
        a[crossover_point1:crossover_point2 + 1], b[crossover_point1:crossover_point2 + 1] = \
             key1.sequence[crossover_point1:crossover_point2 + 1], key2.sequence[crossover_point1:crossover_point2 + 1]
        child_a.sequence, child_b.sequence = Mapping_PMC(key1, key2, a, b, child_b.sequence, crossover_point1, crossover_point2)

        return child_a, child_b

    def mutation_function(child_1, child_2, mutation_rate):
        new_key_child_1, new_key_child_2 = child_1.deepcopy(), child_2.deepcopy()
        letter1, letter2 = np.random.randint(len(child_1.sequence)), np.random.randint(len(child_2.sequence))
        if np.argwhere(np.random.rand(*child_1.sequence.shape) <= mutation_rate).size:
            for i in np.argwhere(np.random.rand(*child_1.sequence.shape) <= mutation_rate):
                new_key_child_1.sequence[i.item()], new_key_child_1.sequence[letter1] = \
                    new_key_child_1.sequence[letter1], new_key_child_1.sequence[
                        i.item()]
        if np.argwhere(np.random.rand(*child_2.sequence.shape) <= mutation_rate).size:
            for i in np.argwhere(np.random.rand(*child_2.sequence.shape) <= mutation_rate):
                new_key_child_2.sequence[i.item()], new_key_child_2.sequence[letter2] = \
                    new_key_child_2.sequence[letter2], new_key_child_2.sequence[
                        i.item()]
        return new_key_child_1, new_key_child_2

    def roulette(p):
        return np.argwhere((sum(p) * np.random.rand()) <= np.cumsum(p))[0][0]

    def decrypt_text(code):
        code_dict = dict(zip(np.array(list('abcdefghijklmnopqrstuvwxyz')), code))
        mapp = {letter: code_dict.get(letter, letter) for letter in list('abcdefghijklmnopqrstuvwxyz')}
        return encrypted_text.translate(str.maketrans(mapp))

    empty_individual = structure()
    empty_individual.sequence = None
    empty_individual.fitness = None
    best_key = empty_individual.deepcopy()
    best_key.fitness = -np.inf
    breeding_population = empty_individual.repeat(population_size)
    for i in range(population_size):
        breeding_population[i].sequence = np.random.permutation(np.array(list('abcdefghijklmnopqrstuvwxyz')).copy())
        plain_text = decrypt_text(breeding_population[i].sequence)
        breeding_population[i].fitness = calculate_fitness(fitness_counter, given_letter_freq, given_letter_pairs_freq, words,
                                           plain_text)
        if breeding_population[i].fitness > best_key.fitness:
            best_key = breeding_population[i].deepcopy()
    fitness_scores = np.empty(num_generations)
    avg = np.empty(num_generations)

    for Generation in range(num_generations):
        costs = np.array([x.fitness for x in breeding_population])
        average_cost = np.mean(costs)
        normalized_costs = costs / average_cost if average_cost != 0 else costs
        probs = np.exp(-normalized_costs)
        probs /= np.sum(probs)
        avg[Generation] = average_cost     # assuming 'Generation' is a variable storing the current generation number
        print(f"Generation : {Generation}")

        population = []
        range_population_size = int(np.round(population_size) * 2) // 2
        for _ in range(range_population_size):
            parent1, parent2 = breeding_population[roulette(probs)], breeding_population[roulette(probs)]
            child1, child2 = crossover(parent1, parent2)
            child1, child2 = mutation_function(child1, child2, mutation_rate)
            plain_text = decrypt_text(child1.sequence)
            child1.fitness = calculate_fitness(fitness_counter, given_letter_freq, given_letter_pairs_freq, words,
                                               plain_text)
            plain_text = decrypt_text(child2.sequence)
            child2.fitness = calculate_fitness(fitness_counter, given_letter_freq, given_letter_pairs_freq, words,
                                               plain_text)
            population.append(child1)
            population.append(child2)
            if child1.fitness > best_key.fitness:
                best_key = child1.deepcopy()
            elif child2.fitness > best_key.fitness:
                best_key = child2.deepcopy()
        breeding_population += population
        breeding_population = sorted(breeding_population, key=lambda x: x.fitness, reverse=True)
        breeding_population = breeding_population[:population_size]
        fitness_scores[Generation] = best_key.fitness
    # print(best_key.sequence)
    return best_key.sequence, fitness_scores, avg


if __name__ == '__main__':
    fitness_counter = 0
    words = load_english_words('dict.txt')
    given_letter_freq = load_letter_frequencies('Letter_Freq.txt')
    given_letter_pairs_freq = load_letter_pair_frequencies('Letter2_Freq.txt')
    encrypted_text = load_text('enc.txt')
    avg = 0
    english_letters_alph = np.array(list('abcdefghijklmnopqrstuvwxyz'))
    for i in range(1):
        # Start the timer
        start_time = time.time()
        key, score, avg_fitness = optimize_key_fitness(encrypted_text, given_letter_freq,
                                                   given_letter_pairs_freq, words, fitness_counter, 0.05, 100, 50)


        # Calculate the elapsed time
        elapsed_time = time.time() - start_time

        # Convert elapsed time to MM:SS format
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        time_formatted = "{:02d}:{:02d}".format(minutes, seconds)

        avg += score[99]
        print("i:", i)
        # Print the elapsed time in MM:SS format
        print("Elapsed time: " + time_formatted)
        with open("plain.txt", 'w') as file:
            file.write('\n'.join([f"{english_letters_alph[i]} {key[i]}"
                                  for i in range(len(english_letters_alph))]))

        mapping = dict(zip(english_letters_alph, key))
        decoded_text = encrypted_text.translate(str.maketrans(mapping))
        with open("perm.txt", 'w') as file:
            file.write(decoded_text)

    print("avg score:", avg / 10)
