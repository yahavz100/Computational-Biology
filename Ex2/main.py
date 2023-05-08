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
    return freq_dict


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
    return freq_dict


def calculate_letter_pair_frequencies(text: str) -> dict:
    """
    Calculate the frequency of each pair of letters in the given text and return the result as a dictionary.
    """
    frequencies = {}
    for i in range(len(text) - 1):
        pair = text[i:i+2].lower()
        if pair.isalpha():
            frequencies[pair] = frequencies.get(pair, 0) + 1
    total_pairs = sum(frequencies.values())
    for pair, count in frequencies.items():
        frequencies[pair] = count / total_pairs
    return frequencies


def guess_key_letter(letter_frequency: dict, target_frequency: dict) -> str:
    """
    Guess the letter that maps to the highest frequency letter in the target frequency, based on the provided letter
    frequency.
    """
    max_target_freq = max(target_frequency.values())
    target_letter = [letter for letter, freq in target_frequency.items() if freq == max_target_freq][0]
    max_letter_freq = max(letter_frequency.values())
    letter = [letter for letter, freq in letter_frequency.items() if freq == max_letter_freq][0]
    key_letter = chr((ord(target_letter) - ord(letter)) % NUM_LETTERS_ENGLISH_ALPHABET + ord('a'))
    return key_letter


def guess_key_letters(pair_frequency: dict, target_frequency: dict, dict_file: str) -> dict:
    """
    Guess the key letters that map to the most frequent letter pairs in the target frequency, based on the provided pair frequency and a dictionary file.
    """
    pass


def decrypt_text(ciphertext: str, key: dict) -> str:
    """
    Decrypt the ciphertext using the provided key and return the result as a string.
    """
    pass


if __name__ == '__main__':
    encrypted_text = load_text('enc.txt')
    encrypted_letter_freq = calculate_letter_frequencies(encrypted_text)
    print(encrypted_letter_freq)
    given_letter_freq = load_letter_frequencies('Letter_Freq.txt')
    print(given_letter_freq)
    given_letter_pairs_freq = load_letter_pair_frequencies('Letter2_Freq.txt')
    print(given_letter_pairs_freq)
    encrypted_letter_pair_freq = calculate_letter_pair_frequencies(encrypted_text)
    print(encrypted_letter_pair_freq)
    result = guess_key_letter(encrypted_letter_freq, given_letter_freq)
    print(result)

