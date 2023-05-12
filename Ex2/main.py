import string
import random

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

    # print(fitness_function(conversion_dict, encrypt_text, given_letter, given_letter_pairs_freq))

    # plain_text = ""
    # for letter in encrypted_text:
    #     if letter in conversion_dict:
    #         plain_text += conversion_dict[letter]
    #     else:
    #         plain_text += letter

    # print(plain_text)
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
                      english_words: set) -> float:
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
        fitness += abs(expected_freq - freq)
    for pair, freq in letter_pair_freqs.items():
        expected_freq = given_letter_pair_frequencies.get(pair, 0)
        fitness += abs(expected_freq - freq)
    words = decrypted_text.split()
    num_english_words = sum(1 for word in words if word in english_words)
    fitness += num_english_words * 10  # Arbitrary weighting for English words
    # print(fitness)

    return fitness


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
        return calculate_fitness(plaintext, given_letter_freq, given_letter_pair_freq, english_word_set)

    # Define the crossover function
    def crossover_function(key1, key2):
        new_key = {}
        for letter in key1:
            if random.random() < 0.5:
                new_key[letter] = key1[letter]
            else:
                new_key[letter] = key2[letter]
        return new_key

    # Define the mutation function
    def mutation_function(key):
        new_key = dict(key)
        for _ in range(random.randint(1, 5)):
            letter1, letter2 = random.sample(new_key.keys(), 2)
            new_key[letter1], new_key[letter2] = new_key[letter2], new_key[letter1]
        return new_key

    # Initialize the population with random decryption keys
    population = []
    for i in range(population_size):
        key = {}
        for letter in encrypted_letter_freq:
            key[letter] = letter
        random.shuffle(list(key.values()))
        population.append(key)

    # Run the genetic algorithm for the specified number of generations
    for generation in range(num_generations):
        # Evaluate the fitness of each decryption key in the population
        fitness_scores = [fitness_function(key) for key in population]

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
        print(generation)

    # Return the decryption key with the highest fitness score
    fitness_scores = [fitness_function(key) for key in population]
    print(fitness_scores)
    best_key = population[fitness_scores.index(max(fitness_scores))]
    return best_key


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
    key = optimize_key_fitness(encrypted_text, encrypted_letter_freq, encrypted_letter_pair_freq, words)
    print(key)
    print(decrypt_text(encrypted_text, key))
    # guessed_letter = guess_key_letter(encrypted_letter_freq, given_letter_freq)
    # print(guessed_letter)
    # words = load_english_words('dict.txt')
    # result = guess_key_letters(encrypted_letter_freq, encrypted_letter_pair_freq, given_letter_freq,
    #                            given_letter_pairs_freq, 'dict.txt', encrypted_text)
    # print(result)

# def fitness_function(decryption_key: dict, text: str, letter_freq: dict, letter_pair_freq: dict) -> float:
#     """
#     Evaluate the fitness of the given decryption key by comparing its letter and letter pair frequencies to those of the English language.
#
#     Returns a float value between 0 and 1, where 1 is a perfect match with English letter and letter pair frequencies.
#     """
#     plaintext = decrypt_text(text, decryption_key)
#
#     # Calculate the frequency of each letter pair in the plaintext
#     pair_freq = calculate_letter_pair_frequencies(plaintext)
#
#     # Calculate the fitness as the sum of the absolute differences between the frequencies of each letter and letter
#     # pair in the plaintext and those of English
#     fitness = 0.0
#     for letter, freq in letter_freq.items():
#         fitness += abs(plaintext.count(letter) / len(plaintext) - freq)
#     for pair, freq in letter_pair_freq.items():
#         fitness += abs(pair_freq.get(pair, 0) - freq)
#
#     # Normalize the fitness to a value between 0 and 1
#     fitness /= (len(letter_freq) + len(letter_pair_freq))
#
#     return 1 - fitness


# def guess_key_letter(letter_frequency: dict, target_frequency: dict) -> str:
#     """
#     Guess the letter that maps to the highest frequency letter in the target frequency, based on the provided letter
#     frequency.
#     """
#     max_target_freq = max(target_frequency.values())
#     target_letter = [letter for letter, freq in target_frequency.items() if freq == max_target_freq][0]
#     max_letter_freq = max(letter_frequency.values())
#     letter = [letter for letter, freq in letter_frequency.items() if freq == max_letter_freq][0]
#     key_letter = chr((ord(target_letter) - ord(letter)) % NUM_LETTERS_ENGLISH_ALPHABET + ord('a'))
#     return key_letter


# def guess_key_letters(encrypted_letter_pair, given_letter_pair, dictionary_file, encrypt_text):
#     # Convert dictionary_file into a set of words
#     with open(dictionary_file, 'r') as f:
#         words = set(f.read().splitlines())
#
#     key_letters = {}
#
#     # Iterate over all pairs of letters in the encrypted text
#     for pair in encrypted_letter_pair:
#         # If the pair is not in the given_letter_pair dictionary, skip it
#         if pair not in given_letter_pair:
#             continue
#
#         # Get the list of possible letters that could represent this pair
#         possible_letters = []
#         for letter in given_letter_pair.keys():
#             if letter in encrypted_letter_pair:
#                 possible_letters.append(letter)
#
#         # If there is only one possible letter, assign it to the pair
#         if len(possible_letters) == 1:
#             key_letters[pair] = possible_letters[0]
#
#         # If there are multiple possible letters, use the dictionary file to choose the most likely one
#         elif len(possible_letters) > 1:
#             max_count = 0
#             max_letter = ''
#             for letter in possible_letters:
#                 # Replace the pair in the encrypted text with the letter and count the number of words that appear
#                 decrypted_text = encrypt_text.replace(pair, letter)
#                 count = sum([1 for word in decrypted_text.split() if word.lower() in words])
#
#                 # If this letter produces more valid words than the previous best, update max_count and max_letter
#                 if count > max_count:
#                     max_count = count
#                     max_letter = letter
#
#             # Assign the most likely letter to the pair
#             key_letters[pair] = max_letter
#
#     return key_letters