# Yahav Zarfati
# Noa Miara Levi
import random

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as c

P = 0.5
SIZE = 5
L = 10

# Define the probability of passing on the rumor for each level of skepticism
P_S2 = 0.67
P_S3 = 0.33

S1 = 1
S2 = 2
S3 = 3
S4 = 4

s2_ratio = 0.2
s3_ratio = 0.1
s4_ratio = 0.1


def init_grid():
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

    return grid, persons


def copy_grid_by_rumors_recieved(grid):
    return [[0 if person is None else person.rumor_received for person in row] for row in grid]


def display_grid(grid: np.ndarray):
    fig, ax = plt.subplots()
    cmap = c.ListedColormap(['white', 'yellow', 'red'])
    bounds = [0, 1, 2, 3]
    norm = c.BoundaryNorm(bounds, cmap.N)
    grid_to_show = copy_grid_by_rumors_recieved(grid)
    ax.imshow(grid_to_show, cmap=cmap, norm=norm)
    plt.show()
    # plt.draw()
    plt.pause(0.001)

    # plt.cla()


def main_loop(grid, persons):
    random_person: Person = random.choice(persons)
    queue = [random_person]
    is_first_person = True

    while queue:

        current_person = queue.pop(0)
        current_person.decide_if_to_accept_rumor()

        if is_first_person:
            is_first_person = False
            current_person.rumor_received = True
            current_person.generations_left = L

        if current_person.rumor_received:
            neighbors_list = current_person.scan_neighbors(grid)

            for neighbor in neighbors_list:
                if neighbor.generations_left == 0:
                    queue.append(neighbor)

        display_grid(grid)


class Person:
    def __init__(self, x, y, level_of_skepticism):
        self.x = x
        self.y = y
        self.level_of_skepticism = level_of_skepticism
        self.generations_left = 0
        self.rumor_received = False

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

    def decide_if_to_accept_rumor(self):
        if self.level_of_skepticism == S1:
            self.rumor_received = True
            self.generations_left = L
        elif self.level_of_skepticism == S2:
            if random.random() < P_S2:
                self.rumor_received = True
                self.generations_left = L
        elif self.level_of_skepticism == S3:
            # If S3, accept rumor with probability of 1/3
            if random.random() < P_S3:
                self.rumor_received = True
                self.generations_left = L
        elif self.level_of_skepticism == S4:
            pass

        else:
            self.generations_left -= 1


if __name__ == '__main__':
    initialized_grid, list_persons = init_grid()
    main_loop(initialized_grid, list_persons)
