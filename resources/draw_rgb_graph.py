import matplotlib.pyplot as plt
import numpy as np
from skimage import color

def convert(i, j):
    L = np.ones((1, 1)) * 50
    a = np.ones((1, 1)) * i
    b = np.ones((1, 1)) * j

    Lab = np.stack([L, a, b], axis=2)

    rgb_0_1 = color.lab2rgb(Lab)
    return rgb_0_1[0][0]
    

def draw_rgb_graph():
    pts_path = "pts_in_hull.npy"
    pts_in_hull = np.load(pts_path)
    fig, ax = plt.subplots()

    for i, j in pts_in_hull:
        ax.add_patch(plt.Rectangle((j, i), 10, 10, color=convert(i, j)))
    
    ax.grid(True, linestyle=':', alpha=0.8)            
    ax.set_xlim(-110, 117)
    ax.set_ylim(117, -110)
    ax.set_aspect('equal', adjustable='box')

    fig.savefig("rgb_graph.png")

draw_rgb_graph()
