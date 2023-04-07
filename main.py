# Yahav Zarfati
# Noa Miara Levi
import random

from matplotlib import pyplot as plt
from matplotlib import colors as c
import numpy as np
from typing import List

P = 0.5
SIZE = 100
L = 10
NUM_OF_RUNS = 10

# Define the probability of passing on the rumor for each level of skepticism
P_S1 = 1.0
P_S2 = 0.67
P_S3 = 0.33
P_S4 = 0.0

s2_ratio = 0.1
s3_ratio = 0.3
s4_ratio = 0.4


def init_persons():
    # Define skepticism levels and their corresponding ratios
    skepticism_levels = [1, 2, 3, 4]
    skepticism_ratios = [1 - s2_ratio - s3_ratio - s4_ratio, s2_ratio, s3_ratio, s4_ratio]

    # Create a grid of None values with the given size
    grid = np.empty((SIZE, SIZE), dtype=object)
    grid[:] = None

    # Loop through each cell in the grid and randomly assign a skepticism level to each person
    persons = []
    for i in range(SIZE):
        for j in range(SIZE):
            if np.random.rand() < P:
                level_of_skepticism = np.random.choice(skepticism_levels, p=skepticism_ratios)
                person = Person(i, j, level_of_skepticism)
                grid[i][j] = person
                persons.append(person)

    return persons, grid


def do_step(persons, grid):
    random_person = random.choice(persons)
    neighbors = random_person.scan_neighbors(grid)
    random_person.take_decision(neighbors)
    return grid


# def find_person(i, j, people):
#     for person in people:
#         if person.x == i and person.y == j:
#             return person
#     return None


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
            if neighbor:
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
                else:
                    neighbor.generations_left = neighbor.generations_left - 1

        if rumor_received:
            # Temporarily decrease confidence level if rumor received from at least two neighbors
            if sum([neighbor.generations_left == 0 for neighbor in neighbors]) >= 2:
                if self.level_of_skepticism == "S4":
                    self.level_of_skepticism = "S3"
                elif self.level_of_skepticism == "S3":
                    self.level_of_skepticism = "S2"
                elif self.level_of_skepticism == "S2":
                    self.level_of_skepticism = "S1"
            # Set the number of generations left for the rumor to be passed to L
            self.generations_left = L

    """
    init the list of persons and the grid accordingly
    """

    def scan_neighbors(self, grid):
        neighbors = []
        x = self.x
        y = self.y
        for i in range(x - 1, x + 2):
            for j in range(y - 1, y + 2):
                if i == x and j == y:
                    continue
                if 0 <= i < SIZE and 0 <= j < SIZE:
                    neighbors.append(grid[i][j])
        return neighbors


"""
Draw the cached matrix to the client.
"""


def draw_to_client(grid: np.ndarray, people: List[Person]):
    print(grid)
    for i in range(NUM_OF_RUNS):
        do_step(people, grid)
        print(grid)


if __name__ == '__main__':
    n_persons, n_grid = init_persons()
    print(n_grid)
    draw_to_client(n_grid, n_persons)
    print(n_persons, n_grid)
