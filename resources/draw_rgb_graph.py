import matplotlib.pyplot as plt
import numpy as np
from skimage import color

def draw_rgb_graph():
    pts_path = "pts_in_hull.npy"
    pts_in_hull = np.load(pts_path)

    L = np.ones(pts_in_hull.shape[0]) * 50
    a = pts_in_hull[:, 0]
    b = pts_in_hull[:, 1]

    Lab = np.stack([L, a, b], axis=1)
    rgb_0_1 = color.lab2rgb(Lab)

    fig, ax = plt.subplots()

    for i in range(pts_in_hull.shape[0]):
        ax.add_patch(plt.Rectangle((pts_in_hull[i, 1], pts_in_hull[i, 0]), 10, 10, color=rgb_0_1[i]))

    ax.grid(True, linestyle=':', alpha=0.8)
    ax.set_xlim(-110, 117)
    ax.set_ylim(117, -110)
    ax.set_aspect('equal', adjustable='box')

    fig.savefig("rgb_graph.png")

draw_rgb_graph()
