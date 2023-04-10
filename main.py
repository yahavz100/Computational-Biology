# Yahav Zarfati
# Noa Miara Levi

import random
from typing import List, Tuple
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as c
import tkinter as tk
import matplotlib.pyplot as plt

SIZE: int = 5
# global P, L, s1_ratio, s2_ratio, s3_ratio, s4_ratio
P: float = 0.5
L: int = 10
s2_ratio: float = 0.2
s3_ratio: float = 0.1
s4_ratio: float = 0.1

# Define the probability of passing on the rumor for each level of skepticism
P_S2: float = 0.67
P_S3: float = 0.33

S1: int = 1
S2: int = 2
S3: int = 3
S4: int = 4


def init_grid() -> Tuple[np.ndarray, List['Person']]:
    """
    Initialize a grid with persons randomly distributed across it and return the grid and persons.
    Returns:
    A tuple containing the initialized grid and a list of Person instances.
    """
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


def copy_grid_by_rumors_received(grid: np.ndarray):
    """
    Return a copy of the input grid where each person is represented by the rumor_received attribute.
    """
    return [[-1 if person is None else person.rumor_received for person in row] for row in grid]


def display_grid(grid: np.ndarray) -> None:
    """
    Display the grid as an image.
    """
    fig, ax = plt.subplots()
    cmap = c.ListedColormap(['white', 'black', 'red'])
    bounds = [-1, 0, 1, 2]
    norm = c.BoundaryNorm(bounds, cmap.N)
    grid_to_show = copy_grid_by_rumors_received(grid)
    ax.imshow(grid_to_show, cmap=cmap, norm=norm)
    plt.show()
    # plt.draw()
    plt.pause(0.001)

    # plt.cla()


class UpdateValuesScreen(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

        # Create labels for each input field
        self.p_label = tk.Label(self, text="Enter P value:")
        self.l_label = tk.Label(self, text="Enter L value:")
        self.s1_label = tk.Label(self, text="Enter S1 value:")
        self.s2_label = tk.Label(self, text="Enter S2 value:")
        self.s3_label = tk.Label(self, text="Enter S3 value:")
        self.s4_label = tk.Label(self, text="Enter S4 value:")

        # Create entry fields for each input value
        self.p_entry = tk.Entry(self)
        self.l_entry = tk.Entry(self)
        self.s1_entry = tk.Entry(self)
        self.s2_entry = tk.Entry(self)
        self.s3_entry = tk.Entry(self)
        self.s4_entry = tk.Entry(self)

        # Create a button to update the values and display the plot
        self.update_button = tk.Button(self, text="Update Values", command=self.update_values)

        # Layout the widgets using grid
        self.p_label.grid(row=0, column=0)
        self.p_entry.grid(row=0, column=1)
        self.l_label.grid(row=1, column=0)
        self.l_entry.grid(row=1, column=1)
        self.s1_label.grid(row=2, column=0)
        self.s1_entry.grid(row=2, column=1)
        self.s2_label.grid(row=3, column=0)
        self.s2_entry.grid(row=3, column=1)
        self.s3_label.grid(row=4, column=0)
        self.s3_entry.grid(row=4, column=1)
        self.s4_label.grid(row=5, column=0)
        self.s4_entry.grid(row=5, column=1)
        self.update_button.grid(row=6, column=0, columnspan=2)

    def update_values(self):
        global P, L, s1_ratio, s2_ratio, s3_ratio, s4_ratio
        # Get the values entered by the user
        P = float(self.p_entry.get())
        L = float(self.l_entry.get())
        s1_ratio = float(self.s1_entry.get())
        s2_ratio = float(self.s2_entry.get())
        s3_ratio = float(self.s3_entry.get())
        s4_ratio = float(self.s4_entry.get())

        self.parent.destroy()


def main_loop(grid: np.ndarray, persons: list) -> None:
    """
    Simulate the spreading of a rumor throughout the grid.
    """
    random_person: Person = random.choice(persons)
    queue = [random_person]
    is_first_person = True
    # Create the main window and add the UpdateValuesScreen to it
    root = tk.Tk()
    update_screen = UpdateValuesScreen(root)
    update_screen.pack()

    # Start the main event loop
    root.mainloop()
    # print(P, L, s1_ratio, s2_ratio, s3_ratio, s4_ratio)

    while queue:

        current_person = queue.pop(0)
        current_person.decide_if_to_accept_rumor(grid)

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


def check_neighbors_rumor(neighbors_list):
    """
    Check if there are at least two neighbors that have received the rumor.
    """
    rumor_count = 0
    for neighbor in neighbors_list:
        if neighbor.rumor_received:
            rumor_count += 1
        if rumor_count >= 2:
            return True
    return False


class Person:
    def __init__(self, x, y, level_of_skepticism):
        self.x = x
        self.y = y
        self.level_of_skepticism = level_of_skepticism
        self.generations_left = 0
        self.rumor_received = False

    def scan_neighbors(self, grid: np.ndarray) -> list:
        """
        Scans the neighbors of the current person and returns a list of neighbors.
        """
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

    def define_level_of_skepticism(self, grid: np.ndarray) -> None:
        """
        Check if at least two neighbors of the person has received the rumor and change the level of skepticism accordingly.
        """
        neighbors = self.scan_neighbors(grid)

        # Check if at least two neighbors has received the rumor
        if check_neighbors_rumor(neighbors):
            # Decrease skepticism level based on current level
            if self.level_of_skepticism == S4:
                self.level_of_skepticism = S3
            elif self.level_of_skepticism == S3:
                self.level_of_skepticism = S2
            elif self.level_of_skepticism == S2:
                self.level_of_skepticism = S1

    def decide_if_to_accept_rumor(self, grid):
        """
        Decides whether the person should accept the rumor or not.
        """
        self.define_level_of_skepticism(grid)

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
