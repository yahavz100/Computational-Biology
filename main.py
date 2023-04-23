# Yahav Zarfati
# Noa Miara Levi

import random
from typing import List, Tuple
import numpy as np
from matplotlib import colors as c
import tkinter as tk
import matplotlib.pyplot as plt

SIZE: int = 100
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


def round_small_values(arr: List[float], threshold: float = 1e-12) -> List[float]:
    """
    Round values in the input array that are very close to zero to zero.

    Args:
        arr (List[float]): The input array of floats.
        threshold (float): The threshold value used to determine if a value is close to zero.

    Returns:
        List[float]: The input array with small values rounded to zero.
    """
    return [0 if abs(x) < threshold else x for x in arr]


def init_grid() -> Tuple[np.ndarray, List['Person']]:
    """
    Initialize a grid with persons randomly distributed across it and return the grid and persons.
    Returns:
    A tuple containing the initialized grid and a list of Person instances.
    """
    # Define skepticism levels and their corresponding ratios
    skepticism_levels = [1, 2, 3, 4]
    skepticism_ratios = round_small_values([1 - s2_ratio - s3_ratio - s4_ratio, s2_ratio, s3_ratio, s4_ratio])

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


class UpdateValuesScreen(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

        # Create a label for the title text
        self.title_label = tk.Label(self, text="Welcome to Spreading Rumours\n Enter the following values:",
                                    justify="center")

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
        self.title_label.grid(row=0, column=0, columnspan=3, pady=(10, 20))
        self.p_label.grid(row=1, column=0, padx=20, pady=10)
        self.p_entry.grid(row=1, column=1, padx=20, pady=10)
        self.l_label.grid(row=2, column=0, padx=20, pady=10)
        self.l_entry.grid(row=2, column=1, padx=20, pady=10)
        self.s1_label.grid(row=3, column=0, padx=20, pady=10)
        self.s1_entry.grid(row=3, column=1, padx=20, pady=10)
        self.s2_label.grid(row=4, column=0, padx=20, pady=10)
        self.s2_entry.grid(row=4, column=1, padx=20, pady=10)
        self.s3_label.grid(row=5, column=0, padx=20, pady=10)
        self.s3_entry.grid(row=5, column=1, padx=20, pady=10)
        self.s4_label.grid(row=6, column=0, padx=20, pady=10)
        self.s4_entry.grid(row=6, column=1, padx=20, pady=10)
        self.update_button.grid(row=7, column=0, columnspan=3, pady=20)

        # Add an empty label to fill the bottom right cell of the grid
        self.bottom_label = tk.Label(self, text="", width=20)
        self.bottom_label.grid(row=8, column=2, sticky="nsew")

        # Set the last row and column to have a weight of 1
        self.grid_rowconfigure(8, weight=1)
        self.grid_columnconfigure(2, weight=1)

    def update_values(self):
        global P, L, s2_ratio, s3_ratio, s4_ratio
        # Get the values entered by the user
        P = float(self.p_entry.get())
        L = float(self.l_entry.get())
        s2_ratio = float(self.s2_entry.get())
        s3_ratio = float(self.s3_entry.get())
        s4_ratio = float(self.s4_entry.get())

        self.parent.destroy()


def main_loop(grid: np.ndarray, persons: list) -> None:
    """
    Simulate the spreading of a rumor throughout the grid.
    """
    # Create a new figure and axis for plotting
    fig, ax = plt.subplots(figsize=(8, 8))

    # Define the colors for the grid (white, black, red)
    cmap = c.ListedColormap(['white', 'black', 'red'])

    # Define the boundaries between the different colors
    bounds = [-1, 0, 1, 2]

    # Create a norm object that maps values to colors
    norm = c.BoundaryNorm(bounds, cmap.N)

    # Initialize variables for tracking the number of people who have received the rumor
    random_person: Person = random.choice(persons)
    random_person.rumor_received = True
    num_rumor_received = 1
    random_person.generations_left = L
    queue = [random_person]
    visited = set()
    temp_queue = []

    # Loop until the rumor has spread to everyone in the network
    while queue:

        # Move all the people in the queue to a temporary queue
        while queue:
            person = queue.pop(0)
            if person not in visited:
                visited.add(person)
                temp_queue.append(person)

        for person in visited:
            if person.generations_left == 0:
                person.rumor_received = False
            else:
                person.generations_left -= 1

        # Process each person in the temporary queue
        while temp_queue:
            current_person = temp_queue.pop(0)
            if current_person.rumor_received is False and current_person in visited:
                visited.remove(current_person)

            neighbors_list = current_person.scan_neighbors(grid)
            # Check each neighbor to see if they can receive the rumor
            for neighbor in neighbors_list:
                if neighbor not in visited and neighbor.generations_left == 0:
                    # visited.add(neighbor)
                    neighbor.decide_if_to_accept_rumor(grid)
                    if neighbor.rumor_received:
                        num_rumor_received += 1
                        queue.append(neighbor)
                        nei_list = neighbor.scan_neighbors(grid)

                        # Check each of the neighbor's neighbors to see if they can receive the rumor
                        for nei in nei_list:
                            if nei not in visited and nei.generations_left == 0:
                                # visited.add(nei)
                                nei.decide_if_to_accept_rumor(grid)
                                if nei.rumor_received:
                                    num_rumor_received += 1
                                    queue.append(nei)

        # Create a new grid to show the current state of the rumor
        grid_to_show = copy_grid_by_rumors_received(grid)

        # Display the grid as an image
        ax.imshow(grid_to_show, cmap=cmap, norm=norm)

        # Pause briefly to allow the image to be displayed
        plt.pause(0.001)

    show_simulation_complete_popup()


def show_simulation_complete_popup():
    """
    Create a popup window indicating that the simulation is complete.
    """
    root = tk.Tk()
    root.title("Simulation Complete")
    message = tk.Label(root, text="Simulation complete!")
    message.pack()
    button = tk.Button(root, text="Close", command=root.destroy)
    button.pack()
    root.mainloop()


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
        Check if at least two neighbors of the person has received the rumor and change the level of skepticism
        accordingly.
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
        self.rumor_received = False
        self.define_level_of_skepticism(grid)

        if self.level_of_skepticism == S1:
            self.rumor_received = True
            self.generations_left = L
        elif self.level_of_skepticism == S2:
            if random.random() < P_S2:
                self.rumor_received = True
                self.generations_left = L
            else:
                self.generations_left -= 1

                if self.generations_left < 0:
                    self.generations_left = 0
        elif self.level_of_skepticism == S3:
            # If S3, accept rumor with probability of 1/3
            if random.random() < P_S3:
                self.rumor_received = True
                self.generations_left = L
            else:
                self.generations_left -= 1

                if self.generations_left < 0:
                    self.generations_left = 0
        elif self.level_of_skepticism == S4:
            pass


if __name__ == '__main__':
    # Create the main window and add the UpdateValuesScreen to it
    root = tk.Tk()
    root.geometry("800x800")
    root.resizable(True, True)
    update_screen = UpdateValuesScreen(root)
    update_screen.pack()

    # Start the main event loop
    root.mainloop()
    initialized_grid, list_persons = init_grid()
    main_loop(initialized_grid, list_persons)
