import string
import random
import time

NUM_LETTERS_ENGLISH_ALPHABET = 26


def load_text(filename: str) -> str:
    """
    Load text from a file and return it as a string.
    """
    with open(filename, 'r') as file:
        text = file.read()
    return text


def calculate_letter_frequencies(text: str) -> dict:
    """
    Calculate the frequency of each letter in the given text and return the result as a dictionary.
    """
    freq = {}
    for char in text:
        if char.isalpha():
            char = char.lower()
            freq[char] = freq.get(char, 0) + 1
    total_chars = sum(freq.values())
    for char in freq:
        freq[char] /= total_chars

    freq = dict(sorted(freq.items(), key=lambda x: x[1], reverse=True))
    return freq


def load_letter_frequencies(filename: str) -> dict:
    """
    Load letter frequencies from a file and return them as a dictionary.
    """
    with open(filename, 'r') as file:
        freq_dict = {}
        for line in file:
            freq, letter = line.strip().split()
            freq_dict[letter] = float(freq)

    lower_freq_dict = {k.lower(): v for k, v in freq_dict.items()}
    lower_freq_dict = dict(sorted(lower_freq_dict.items(), key=lambda x: x[1], reverse=True))
    return lower_freq_dict


def load_letter_pair_frequencies(filename: str) -> dict:
    """
    Load letter pair frequencies from a file and return them as a dictionary.
    """
    freq_dict = {}
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                if line == '#REF!':
                    continue
                freq, pair = line.split()
                freq_dict[pair] = float(freq)

    lower_freq_dict = {k.lower(): v for k, v in freq_dict.items()}
    lower_freq_dict = dict(sorted(lower_freq_dict.items(), key=lambda x: x[1], reverse=True))
    return lower_freq_dict


def calculate_letter_pair_frequencies(text: str) -> dict:
    """
    Calculate the frequency of each pair of letters in the given text and return the result as a dictionary.
    """
    frequencies = {}
    for i in range(len(text) - 1):
        pair = text[i:i + 2].lower()
        if pair.isalpha():
            frequencies[pair] = frequencies.get(pair, 0) + 1
    total_pairs = sum(frequencies.values())
    for pair, count in frequencies.items():
        frequencies[pair] = count / total_pairs

    frequencies = dict(sorted(frequencies.items(), key=lambda x: x[1], reverse=True))
    return frequencies


def guess_key_letters(encrypted_letter: dict, encrypted_letter_pair: dict, given_letter: dict, given_letter_pair: dict
                      , words_file: str, encrypt_text: str):
    # Convert dictionary_file into a set of words
    with open(words_file, 'r') as f:
        words = set(f.read().splitlines())

    conversion_dict = dict()
    for letter in encrypted_letter:
        conversion_dict[letter] = list(given_letter.keys())[list(encrypted_letter.keys()).index(letter)]
    print(conversion_dict)
    return 1


def decrypt_text(ciphertext: str, key: dict) -> str:
    """
    Decrypt the ciphertext using the provided key and return the result as a string.
    """
    plaintext = ""
    for i in range(len(ciphertext)):
        if ciphertext[i] in key:
            # If the current character is a key letter, replace it with its corresponding plaintext letter
            plaintext += key[ciphertext[i]]
        else:
            # Otherwise, leave the character unchanged
            plaintext += ciphertext[i]
    return plaintext


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


def calculate_fitness(decrypted_text: str, given_letter_frequencies: dict, given_letter_pair_frequencies: dict,
                      english_words: set, decryption_key: dict) -> float:
    """
    Calculate the fitness of a given key based on the letter frequencies in the decrypted text.
    """
    # Calculate the frequency of each letter and letter pair in the decrypted text
    letter_freqs = calculate_letter_frequencies(decrypted_text)
    letter_pair_freqs = calculate_letter_pair_frequencies(decrypted_text)

    # Compute the fitness based on the letter frequencies and the frequency of English words in the decrypted text
    fitness = 0
    for letter, freq in letter_freqs.items():
        expected_freq = given_letter_frequencies.get(letter, 0)
        fitness -= abs(expected_freq - freq)
    for pair, freq in letter_pair_freqs.items():
        expected_freq = given_letter_pair_frequencies.get(pair, 0)
        fitness -= abs(expected_freq - freq)
    words = decrypted_text.split()

    num_english_words = sum(1 for word in words if word in english_words)
    fitness += num_english_words * 100  # Arbitrary weighting for English words
    if check_unique_values(decryption_key):
        fitness += 5
    for word in decryption_key:
        if check_len_values(decrypted_text, word):
            fitness += 10


    # print(fitness)

    return fitness


def check_unique_values(dictionary):
    values = list(dictionary.values())
    if len(set(values)) == len(values):
        # print("All keys have different values.")
        return True
    # print("Some keys have the same value.")
    return False


def check_len_values(dictionary, word):
    if word in dictionary and len(word) > 3:
        return True
    return False


def optimize_key_fitness(ciphertext, given_letter_freq, given_letter_pair_freq, english_word_set,
                         num_generations=1000, population_size=100, mutation_rate=0.1):
    """
    Use a genetic algorithm to optimize the fitness of a decryption key for the given ciphertext.

    :param ciphertext: The encrypted text to decrypt.
    :param encrypted_letter_freq: The frequency of each letter in the encrypted text.
    :param encrypted_letter_pair_freq: The frequency of each pair of letters in the encrypted text.
    :param english_word_set: A set of English words.
    :param num_generations: The number of generations to run the genetic algorithm for.
    :param population_size: The number of decryption keys in each generation.
    :param mutation_rate: The probability that a random mutation occurs in each decryption key.

    :return: The decryption key with the highest fitness score found by the genetic algorithm.
    """

    # Define the fitness function
    def fitness_function(decryption_key):
        plaintext = decrypt_text(ciphertext, decryption_key)
        return calculate_fitness(plaintext, given_letter_freq, given_letter_pair_freq, english_word_set, decryption_key)

    # Define the crossover function
    import random

    def crossover_function(key1, key2):
        new_key = {}
        crossover_point = random.randint(1, len(key1) - 1)

        # Combine the first part of parent1 with the shuffled second part of parent2
        for i in range(crossover_point):
            letter = list(key1.keys())[i]
            new_key[letter] = key1[letter]

        available_values = list(key2.values())
        random.shuffle(available_values)

        for i, letter in enumerate(list(key2.keys())[crossover_point:]):
            while available_values[i] in new_key.values():
                random.shuffle(available_values)
            new_key[letter] = available_values[i]
        # print(new_key)
        return new_key

    # Define the mutation function
    def mutation_function(key):
        new_key = dict(key)
        letters = list(new_key.keys())

        for _ in range(random.randint(1, 5)):
            letter1, letter2 = random.sample(letters, 2)

            # Check if the letters are already mapped to each other
            if new_key[letter1] != letter2 and new_key[letter2] != letter1:
                new_key[letter1], new_key[letter2] = new_key[letter2], new_key[letter1]
        #print(new_key)
        return new_key

    # Initialize the population with random decryption keys
    population = []
    available_letters = list(string.ascii_lowercase)
    random.shuffle(available_letters)

    for i in range(population_size):
        key = {}
        for j, letter in enumerate(encrypted_letter_freq):
            key[letter] = available_letters[j]
        population.append(key)

    convergence_limit = 25  # Maximum number of generations without improvement
    generations_without_improvement = 0
    best_key = None
    best_fitness = -float('inf')

    # Run the genetic algorithm for the specified number of generations
    for generation in range(num_generations):

        # Evaluate the fitness of each decryption key in the population
        # Add a small positive value to all fitness scores to ensure they are strictly positive
        fitness_scores = [fitness_function(key) + 1e-10 for key in population]

        # Update the best key and its fitness score
        current_best_fitness = max(fitness_scores)
        current_best_index = fitness_scores.index(current_best_fitness)
        current_best_key = population[current_best_index]
        current_best_fitness = fitness_scores[current_best_index]

        if current_best_fitness > best_fitness:
            best_key = current_best_key
            best_fitness = current_best_fitness
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1

        # Check for convergence
        if generations_without_improvement >= convergence_limit:
            print("Reached converge limit, breaking")
            break

        # Select the best decryption keys to breed the next generation
        breeding_population = []
        for i in range(population_size // 2):
            parent1 = random.choices(population, weights=fitness_scores, k=1)[0]
            parent2 = random.choices(population, weights=fitness_scores, k=1)[0]
            breeding_population.append((parent1, parent2))

        # Breed the next generation of decryption keys
        next_population = []
        for parent1, parent2 in breeding_population:
            child = crossover_function(parent1, parent2)
            if random.random() < mutation_rate:
                child = mutation_function(child)
            next_population.append(child)
        population = next_population

        # Replace the worst key with the best key from the previous generation (elitism)
        if best_key is not None:
            worst_index = fitness_scores.index(min(fitness_scores))
            population[worst_index] = best_key
        print("Generation:", generation)
        check_unique_values(population[fitness_scores.index(max(fitness_scores))])

    # Return the decryption key with the highest fitness score
    fitness_scores = [fitness_function(key) for key in population]
    best_index = fitness_scores.index(max(fitness_scores))
    best_key = population[best_index]
    best_fitness_score = fitness_scores[best_index]

    print("Best decryption key:", best_key)
    print("Best fitness score:", best_fitness_score)
    return best_key, best_fitness_score


def calculate_eligible_words_percentage(decrypted_text: str, english_words: set):
    words = decrypted_text.split()
    num_total_words = len(words)
    num_eligible_words = 0

    print("Eligible words:")
    for word in words:
        if word in english_words and word == word in english_words:
            print(word)
            num_eligible_words += 1

    percentage = (num_eligible_words / num_total_words) * 100
    return percentage, num_eligible_words


if __name__ == '__main__':
    encrypted_text = load_text('enc.txt')
    encrypted_letter_freq = calculate_letter_frequencies(encrypted_text)
    given_letter_freq = load_letter_frequencies('Letter_Freq.txt')
    given_letter_pairs_freq = load_letter_pair_frequencies('Letter2_Freq.txt')
    encrypted_letter_pair_freq = calculate_letter_pair_frequencies(encrypted_text)
    print(encrypted_letter_freq)
    print(given_letter_freq)
    print(given_letter_pairs_freq)
    print(encrypted_letter_pair_freq)
    words = load_english_words('dict.txt')
    avg = 0
    for i in range(10):
        # Start the timer
        start_time = time.time()
        key, score = optimize_key_fitness(encrypted_text, encrypted_letter_freq, encrypted_letter_pair_freq, words, 100,
                                          100, 0.01)

        # Calculate the elapsed time
        elapsed_time = time.time() - start_time

        # Convert elapsed time to MM:SS format
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        time_formatted = "{:02d}:{:02d}".format(minutes, seconds)
        avg += score
        print(key)
        print(decrypt_text(encrypted_text, key))
        print("Percentage of eligible words, num of words:",
              calculate_eligible_words_percentage(decrypt_text(encrypted_text, key), words))
        print("i:", i)
        # Print the elapsed time in MM:SS format
        print("Elapsed time: " + time_formatted)
    print("avg score:", avg / 10)