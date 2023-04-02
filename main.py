# Yahav Zarfati
# Noa Miara Levi
from typing import List

import numpy as np

L = 10


class Person:
    def __init__(self, x, y, level_of_skepticism=0, generations_left=L):
        self.x = x
        self.y = y
        self.level_of_skepticism = level_of_skepticism
        self.generations_left = generations_left


"""
Person decide if to accept rumor based on probability,
"""


def accept_rumor(p: float) -> bool:
    pass


"""
Draw cached matrix to client
"""


def draw_to_client(matrix: np.ndarray, people: List[Person]):
    pass


"""
Function receive a person and a list of its neighbors, 
decides according to his neighbors the person temp level of skepticism
"""


def take_decision(person: Person, neighbors: List[Person]):
    pass


if __name__ == '__main__':
    print("hello world")
