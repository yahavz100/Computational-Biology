# Yahav Zarfati
# Noa Miara Levi
import random
from typing import List

import numpy as np

L = 10


class Person:
    def __init__(self, x, y, level_of_skepticism):
        self.x = x
        self.y = y
        self.level_of_skepticism = level_of_skepticism
        self.generations_left = 0

    def pass_rumor(self):
        # Set the number of generations left for the rumor to be passed to L
        self.generations_left = L

    """
    Person decides whether to accept a rumor based on a probability.
    Returns True if the person accepts the rumor, False otherwise.
    """

    def accept_rumor(self, p: float) -> bool:
        if random.random() < p:
            return True
        return False

    """
    Receives a person and a list of its neighbors, 
    and decides according to the neighbors the person's temporary level of skepticism.
    """

    def take_decision(self, neighbors: List['Person']):
        rumor_received = False
        for neighbor in neighbors:
            if neighbor.generations_left == 0:
                # Neighbor can pass on the rumor
                if self.level_of_skepticism == "S1":
                    neighbor.pass_rumor()
                    rumor_received = True
                elif self.level_of_skepticism == "S2":
                    # If S2, accept rumor with probability of 2/3
                    if self.accept_rumor(2 / 3):
                        neighbor.pass_rumor()
                        rumor_received = True
                elif self.level_of_skepticism == "S3":
                    # If S3, accept rumor with probability of 1/3
                    if self.accept_rumor(1 / 3):
                        neighbor.pass_rumor()
                        rumor_received = True
                elif self.level_of_skepticism == "S4":
                    # If S4, never accept the rumor
                    if self.accept_rumor(0):
                        neighbor.pass_rumor()
                        rumor_received = True

        if rumor_received:
            # Temporarily decrease confidence level if rumor received from at least two neighbors
            if sum([neighbor.generations_left == 0 for neighbor in neighbors]) >= 2:
                if self.level_of_skepticism == "S3":
                    self.level_of_skepticism = "S2"
                elif self.level_of_skepticism == "S2":
                    self.level_of_skepticism = "S1"
            # Set the number of generations left for the rumor to be passed to L
            self.generations_left = L


"""
Draw the cached matrix to the client.
"""


def draw_to_client(matrix: np.ndarray, people: List[Person]):
    pass


if __name__ == '__main__':
    print("hello world")
