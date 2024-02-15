import torch
import matplotlib.pyplot as plt
from colorizers import *
from dataloaders import *
from PIL import Image
from tqdm import tqdm

train_in_path = "small-coco-stuff/train2017/train2017"
val_in_path = "small-coco-stuff/train2017/train2017"

train_loader = create_dataloader(train_in_path, shuffle=True)
val_loader = create_dataloader(val_in_path, shuffle=False)

temp_loader = iter(val_loader)
temp_batch = next(temp_loader)

index = 10

temp_in = temp_batch[0][:1]
temp_gt = temp_batch[1][:1]
value = 30
temp_gt = -value * torch.ones(size=temp_gt.shape)
temp_gt = value * torch.ones(size=temp_gt.shape)


min_gt = torch.min(temp_gt)
max_gt = torch.max(temp_gt)

# for temp_in, temp_gt in tqdm(train_loader):
#     if torch.min(temp_gt) < min_gt:
#         min_gt = torch.min(temp_gt)
#     if torch.max(temp_gt) < max_gt:
#         max_gt = torch.max(temp_gt)


print(min_gt)
print(max_gt)
temp_res = postprocess_tens(temp_in, temp_gt)

print(temp_res.shape)

plt.imshow(temp_res)
plt.show()