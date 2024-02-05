import torch
import matplotlib.pyplot as plt
from colorizers import *
from dataload import create_dataloader

train_in_path = "small-coco-stuff/train2017/train2017"
val_in_path = "small-coco-stuff/train2017/train2017"

train_loader = create_dataloader(train_in_path, shuffle=True)
val_loader = create_dataloader(val_in_path, shuffle=False)

temp_loader = iter(train_loader)
temp_batch = next(temp_loader)
temp_batch = next(temp_loader)

index = 10

temp_in = temp_batch[0][:1]
temp_gt = temp_batch[1][:1]

temp_res = postprocess_tens(temp_in, temp_gt)

plt.imshow(temp_res)
plt.show()