# Yahav Zarfati
# Noa Miara Levi
import random

from matplotlib import pyplot as plt
from matplotlib import colors as c
import numpy as np
from typing import List
import matplotlib.colors as mcolors


P = 0.5
SIZE = 100
L = 10
NUM_OF_RUNS = 10

# Define the probability of passing on the rumor for each level of skepticism
P_S1 = 1.0
P_S2 = 0.67
P_S3 = 0.33
P_S4 = 0.0

S1 = 1
S2 = 2
S3 = 3
S4 = 4

s2_ratio = 0.4
s3_ratio = 0.3
s4_ratio = 0.1


def init_persons():
    # Define skepticism levels and their corresponding ratios
    skepticism_levels = [1, 2, 3, 4]
    skepticism_ratios = [0.02, 0.03, 0.9, 0.05]

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
                # print(person.level_of_skepticism)
    return persons, grid


def do_step(persons, grid):
    random_person = random.choice(persons)
    neighbors = random_person.scan_neighbors(grid)
    random_person.take_decision(neighbors)
    for neighbor in neighbors:
        new_nei = neighbor.scan_neighbors(grid)
        neighbor.take_decision(new_nei)
    for person in persons:
        new_nei = person.scan_neighbors(grid)
        person.if_received(new_nei)
    return grid


def copy_grid_with_skepticism_levels(grid):
    """
    Returns a copy of the input grid with the skepticism level of each person in each cell.
    """
    copied_grid = []
    for row in grid:
        copied_row = []
        for person in row:
            if person is None:
                copied_row.append(0)
            else:
                copied_row.append(person.level_of_skepticism)
        copied_grid.append(copied_row)
    if np.array_equal(grid, copied_grid):
        print("DIFF")
    return copied_grid


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
        self.rumor_received = False

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
        self.rumor_received = False
        for neighbor in neighbors:
            if neighbor is not None:
                if neighbor.generations_left == 0:
                    # Neighbor can pass on the rumor
                    if self.level_of_skepticism == S1:
                        neighbor.pass_rumor()
                        self.rumor_received = True
                    elif self.level_of_skepticism == S2:
                        # If S2, accept rumor with probability of 2/3
                        if self.accept_rumor(2 / 3):
                            neighbor.pass_rumor()
                            self.rumor_received = True
                    elif self.level_of_skepticism == S3:
                        # If S3, accept rumor with probability of 1/3
                        if self.accept_rumor(1 / 3):
                            neighbor.pass_rumor()
                            self.rumor_received = True
                    elif self.level_of_skepticism == S4:
                        # If S4, never accept the rumor
                        if self.accept_rumor(0):
                            neighbor.pass_rumor()
                            self.rumor_received = True
                else:
                    neighbor.generations_left = neighbor.generations_left - 1

    def if_received(self, neighbors):
        if self.rumor_received:
            # Temporarily decrease confidence level if rumor received from at least two neighbors
            if sum([neighbor is not None and neighbor.generations_left == 0 for neighbor in neighbors]) >= 2:
                if self.level_of_skepticism is None and all(
                        [neighbor.level_of_skepticism is None for neighbor in neighbors]):
                    self.level_of_skepticism = S4
                elif self.level_of_skepticism == S4:
                    self.level_of_skepticism = S3
                elif self.level_of_skepticism == S3:
                    self.level_of_skepticism = S2
                elif self.level_of_skepticism == S2:
                    self.level_of_skepticism = S1

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
                if 0 <= i < SIZE and 0 <= j < SIZE and grid[i][j] is not None:
                    neighbors.append(grid[i][j])
        return neighbors


"""
Draw the cached matrix to the client.
"""


def draw_to_client(grid: np.ndarray, people: List[Person]):
    # Define a dictionary to map values to colors
    color_dict = {0: 'white', 1: 'cyan', 2: 'yellow', 3: 'orange', 4: 'red'}
    grid_colors = np.vectorize(color_dict.get)(grid)

    for i in range(400):
        # fig, ax = plt.subplots()
        do_step(people, grid)
        grid_to_show = copy_grid_with_skepticism_levels(grid)
        # print(grid_to_show)
        plt.cla()
        cmap = c.ListedColormap(['white', 'cyan', 'yellow', 'orange', 'red'])      #todo fix colors
        bounds = [0, S1, S2, S3, S4, 5]
        norm = c.BoundaryNorm(bounds, cmap.N)
        plt.pcolormesh(grid_to_show, cmap=cmap, norm=norm)
        # plt.colorbar()  # Add a colorbar to the plot
        plt.pause(0.001)

    plt.cla()


if __name__ == '__main__':
    n_persons, n_grid = init_persons()
    draw_to_client(n_grid, n_persons)
