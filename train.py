import torch
import matplotlib.pyplot as plt
from colorizers import *
from dataload import create_dataloader
from torchsummary import summary

train_in_path = "small-coco-stuff/train2017/train2017"
val_in_path = "small-coco-stuff/train2017/train2017"

train_loader = create_dataloader(train_in_path, shuffle=True)
val_loader = create_dataloader(val_in_path, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ECCV_Regression().to(device)
summary(model, (1, 256, 256))