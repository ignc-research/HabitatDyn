from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def makr_robot_loc(map_size):
    plt.scatter(map_size/2, 5, color="b", s=1000)