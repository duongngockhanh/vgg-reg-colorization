import os
import os.path as osp
import numpy as np
from IPython.display import display
from PIL import Image
import time
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


def visualize_image(sample_index, train_input_dir, train_label_dir, val_input_dir, val_label_dir):
    sample_train_input = sorted(os.listdir(train_input_dir))[sample_index]
    sample_train_input_path = osp.join(train_input_dir, sample_train_input)
    input_img = Image.open(sample_train_input_path)
    display(input_img)
    # input_array = np.array(input_img)
    # input_values = np.unique(input_array)
    # print(input_values)

    sample_train_label = sorted(os.listdir(train_label_dir))[sample_index]
    sample_train_label_path = osp.join(train_label_dir, sample_train_label)
    label_img = Image.open(sample_train_label_path)
    display(label_img)
    label_array = np.array(label_img)
    label_values = np.unique(label_array)
    print(label_values)


def visualize_image_size(val_input_dir):
    heights = []
    widths = []
    count_gray = 0
    for i in tqdm(sorted(os.listdir(val_input_dir))):
        sample_val_input_path = osp.join(val_input_dir, i)
        input_img = Image.open(sample_val_input_path)
        input_img = np.array(input_img)
        heights.append(input_img.shape[0])
        widths.append(input_img.shape[1])
        if len(input_img.shape) != 3:
            count_gray += 1
    
    print(f"{count_gray=}")
    plt.figure(figsize=(8, 6))
    plt.scatter(widths, heights, c='red', alpha=0.5, marker='o', edgecolors='black', linewidths=1.5)
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    sample_index = 100
    train_input_dir = "/kaggle/input/aio-coco-stuff/train2017/train2017"
    train_label_dir = "/kaggle/input/aio-coco-stuff/stuffthingmaps_trainval2017/train2017"
    val_input_dir = "/kaggle/input/aio-coco-stuff/val2017/val2017"
    val_label_dir = "/kaggle/input/aio-coco-stuff/stuffthingmaps_trainval2017/val2017"

    visualize_image(sample_index, train_input_dir, train_label_dir, val_input_dir, val_label_dir)
    visualize_image_size(val_input_dir)