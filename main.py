import torch
import matplotlib.pyplot as plt
from colorizers import *
from dataload import create_dataloader

train_in_path = "small-coco-stuff/train2017/train2017"
val_in_path = "small-coco-stuff/train2017/train2017"

train_loader = create_dataloader(train_in_path, shuffle=True)
val_loader = create_dataloader(val_in_path, shuffle=False)

