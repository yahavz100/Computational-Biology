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
P: float = 0.6
L: int = 5
s1_ratio: float = 0.3
s2_ratio: float = 0.35
s3_ratio: float = 0.2
s4_ratio: float = 0.15

# Define the probability of passing on the rumor for each level of skepticism
P_S2: float = 0.67
P_S3: float = 0.33

S1: int = 1
S2: int = 2
S3: int = 3
S4: int = 4

skepticism_levels = [1, 2, 3, 4]
saif_b = False

# Set the default simulation speed
simulation_speed = 0.001


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


def distance(point):
    return ((point[0] - 50) ** 2 + (point[1] - 50) ** 2) ** 0.5


def special_init_grid():
    # Create a grid of None values with the given size
    grid = np.empty((SIZE, SIZE), dtype=object)
    persons = []

    center_x, center_y = SIZE // 2, SIZE // 2
    first_quart = int(0.25 * SIZE)
    third_quart = int(0.75 * SIZE)

    # create a list of all possible locations of the grid
    center_locations = [(x, y) for x in range(center_x - first_quart, center_x + first_quart + 1)
                        for y in range(center_y - first_quart, center_y + first_quart + 1)]

    outskirts_locations = [(x, y) for x in range(0, first_quart)
                           for y in range(0, SIZE)]

    outskirts_locations.extend([(x, y) for x in range(third_quart + 1, SIZE)
                                for y in range(0, SIZE)])

    outskirts_locations.extend([(x, y) for x in range(first_quart, third_quart + 1)
                                for y in range(0, first_quart)])

    outskirts_locations.extend([(x, y) for x in range(first_quart, third_quart + 1)
                                for y in range(third_quart + 1, SIZE)])

    center_locations = sorted(center_locations, key=lambda point: distance(point))
    outskirts_locations = sorted(outskirts_locations, key=lambda point: distance(point))
    outskirts_locations.reverse()

    # Place S1 people
    s1_count = int(s1_ratio * P * SIZE * SIZE)
    for i in range(s1_count):
        if center_locations:
            x, y = center_locations.pop()
        else:
            x, y = outskirts_locations.pop()
        person = Person(x, y, 1)
        grid[x, y] = person
        persons.append(person)

    # Place S2 people
    s2_count = int(s2_ratio * P * SIZE * SIZE)
    for i in range(s2_count):
        if center_locations:
            x, y = center_locations.pop()
        else:
            x, y = outskirts_locations.pop()
        person = Person(x, y, 2)
        grid[x, y] = person
        persons.append(person)

    # Place S3 people
    s3_count = int(s3_ratio * P * SIZE * SIZE)
    for i in range(s3_count):
        if center_locations:
            x, y = center_locations.pop()
        else:
            x, y = outskirts_locations.pop()
        person = Person(x, y, 3)
        grid[x, y] = person
        persons.append(person)

    # Place S4 people
    s4_count = int(s4_ratio * P * SIZE * SIZE)
    for i in range(s4_count):
        if center_locations:
            x, y = center_locations.pop()
        else:
            x, y = outskirts_locations.pop()
        person = Person(x, y, 4)
        grid[x, y] = person
        persons.append(person)

    return grid, persons


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
        self.title_label = tk.Label(self, text="Spreading Rumours", font=("Helvetica", 24, "bold"), pady=20)

        # Create labels and entry fields for each input value
        labels = ["P", "L", "S1", "S2", "S3", "S4"]
        default_vals = [P, L, s1_ratio, s2_ratio, s3_ratio, s4_ratio]

        self.entries = []
        for i, label in enumerate(labels):
            label_text = f"Enter {label} value:"
            label = tk.Label(self, text=label_text, font=("Helvetica", 14), padx=20, pady=10)
            label.grid(row=i + 1, column=0, sticky="w")
            entry = tk.Entry(self, font=("Helvetica", 14), width=10)
            entry.grid(row=i + 1, column=1, padx=20, pady=10)
            entry.insert(0, str(default_vals[i]))
            self.entries.append(entry)

        # Create a button to update the values and display the plot
        self.rumor_button = tk.Button(self, text="Start rumor!", font=("Helvetica", 14), command=self.update_values)
        self.strategy_button = tk.Button(self, text="Our strategy", font=("Helvetica", 14), command=self.update_saif_b)

        # Layout the widgets using grid
        self.title_label.grid(row=0, column=0, columnspan=2)
        self.rumor_button.grid(row=len(labels) + 1, column=0, columnspan=2, pady=20, sticky="s")
        self.strategy_button.grid(row=len(labels) + 1, column=2, columnspan=2, pady=20, sticky="s")

        # Set the last row and column to have a weight of 1
        self.grid_rowconfigure(len(labels) + 2, weight=1)
        self.grid_columnconfigure(2, weight=1)

        # Add speed slider
        speed_slider = tk.Scale(self, from_=1, to=100, orient="horizontal", length=200, label="Simulation Speed",
                                command=on_speed_change)
        speed_slider.set(10)  # set the default speed to 10%
        speed_slider.grid(row=len(labels) + 2, column=0, columnspan=2, pady=20)

    def update_saif_b(self):
        global P, L, s2_ratio, s3_ratio, s4_ratio, saif_b
        saif_b = True
        # Get the values entered by the user
        P = float(self.entries[0].get())
        L = float(self.entries[1].get())
        s2_ratio = float(self.entries[3].get())
        s3_ratio = float(self.entries[4].get())
        s4_ratio = float(self.entries[5].get())

        self.parent.destroy()

    def update_values(self):
        global P, L, s2_ratio, s3_ratio, s4_ratio
        # Get the values entered by the user
        P = float(self.entries[0].get())
        L = float(self.entries[1].get())
        s2_ratio = float(self.entries[3].get())
        s3_ratio = float(self.entries[4].get())
        s4_ratio = float(self.entries[5].get())

        self.parent.destroy()


def main_loop(grid: np.ndarray, persons: list) -> None:
    """
    Simulate the spreading of a rumor throughout the grid.
    """
    # Create a new figure and axis for plotting
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('off')

    # Define the colors for the grid (white, black, red)
    cmap = c.ListedColormap(['white', 'black', 'red'])

    # Define the boundaries between the different colors
    bounds = [-1, 0, 1, 2]

    # Create a norm object that maps values to colors
    norm = c.BoundaryNorm(bounds, cmap.N)

    # Initialize variables for tracking the number of people who have received the rumor
    if saif_b:
        random_person = next(person for person in persons if person.x == 50 and person.y == 50)
    else:
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
                    neighbor.decide_if_to_accept_rumor(grid)
                    if neighbor.rumor_received:
                        num_rumor_received += 1
                        queue.append(neighbor)
                        nei_list = neighbor.scan_neighbors(grid)

                        # Check each of the neighbor's neighbors to see if they can receive the rumor
                        for nei in nei_list:
                            if nei not in visited and nei.generations_left == 0:
                                nei.decide_if_to_accept_rumor(grid)
                                if nei.rumor_received:
                                    num_rumor_received += 1
                                    queue.append(nei)

        # Create a new grid to show the current state of the rumor
        grid_to_show = copy_grid_by_rumors_received(grid)

        # Display the grid as an image
        ax.imshow(grid_to_show, cmap=cmap, norm=norm)

        # Pause briefly to allow the image to be displayed
        plt.pause(simulation_speed)

    show_simulation_complete_popup()


# Allow the user to adjust the simulation speed
def on_speed_change(value_str):
    global simulation_speed
    simulation_speed = float(value_str) / 100


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

    # Update the window to show the contents
    root.update()

    # Start the main event loop
    root.mainloop()

    if saif_b:
        initialized_grid, list_persons = special_init_grid()
    else:
        initialized_grid, list_persons = init_grid()
    main_loop(initialized_grid, list_persons)
