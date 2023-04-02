# Yahav Zarfati
# Noa Miara Levi
import random
from matplotlib import pyplot as plt
from matplotlib import colors as c
import numpy as np

SIZE = 100
global P

# Define the probability of passing on the rumor for each level of skepticism
P_S1 = 1.0
P_S2 = 0.67
P_S3 = 0.33
P_S4 = 0.0

s2_ratio = 0.1
s3_ratio = 0.3
s4_ratio = 0.4

class Person:
    def __init__(self, x, y, level_of_skepticism=0):
        self.x = x
        self.y = y
        self.level_of_skepticism = level_of_skepticism


# init the list of persons and the grid accordingly
def init_persons():
    # Define skepticism levels and their corresponding ratios
    skepticism_levels = [1, 2, 3, 4]
    skepticism_ratios = [1 - s2_ratio - s3_ratio - s4_ratio, s2_ratio, s3_ratio, s4_ratio]

    # Create a grid of zeros with the given size
    grid = np.zeros((SIZE, SIZE))

    # Loop through each cell in the grid and randomly assign a skepticism level to each person
    persons = []
    for i in range(SIZE):
        for j in range(SIZE):
            if np.random.rand() < P:
                level_of_skepticism = np.random.choice(skepticism_levels, p=skepticism_ratios)
                person = Person(i, j, level_of_skepticism)
                persons.append(person)

    return persons, grid


if __name__ == '__main__':
    print("hello world")
