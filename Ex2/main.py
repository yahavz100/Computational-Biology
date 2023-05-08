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
    pass


def calculate_letter_pair_frequencies(text: str) -> dict:
    """
    Calculate the frequency of each pair of letters in the given text and return the result as a dictionary.
    """
    pass


def guess_key_letter(letter_frequency: dict, target_frequency: dict) -> str:
    """
    Guess the letter that maps to the highest frequency letter in the target frequency, based on the provided letter frequency.
    """
    pass


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
    text_str = load_text('enc.txt')
    letter_freq = calculate_letter_frequencies(text_str)
    print(letter_freq)
    given_letter_freq = load_letter_frequencies('Letter_Freq.txt')
    print(given_letter_freq)
