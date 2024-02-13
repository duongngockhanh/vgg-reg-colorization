import torch
import matplotlib.pyplot as plt
from colorizers import *
from dataloaders import *

def main(train_in_path=None, val_in_path=None, weight=None):
    if train_in_path == None or val_in_path == None:
        train_in_path = "/kaggle/input/small-coco-stuff/small-coco-stuff/train2017/train2017"
        val_in_path = "/kaggle/input/small-coco-stuff/small-coco-stuff/train2017/train2017"

    train_batch_size = 32
    val_batch_size = 8

    train_loader = create_dataloader(train_in_path, batch_size=train_batch_size, shuffle=True)
    val_loader = create_dataloader(val_in_path, batch_size=val_batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECCV_Regression().to(device)

    if weight != None:
        model.load_state_dict(torch.load(weight))

    model.eval()
    with torch.no_grad():
        val_iter = iter(val_loader)
        val_first = next(val_iter)

        images_pred = []
        images_gt = []

        fixed_num_showed_image = 1
        num_showed_image = val_batch_size if val_batch_size < fixed_num_showed_image else fixed_num_showed_image

        for i in range(num_showed_image):
            l_in = val_first[0][i:i+1].to(device)
            ab_pred = model(l_in)
            temp = ab_pred.detach().cpu().numpy().reshape(-1)
            plt.hist(temp)
            plt.show()
            rgb_pred = postprocess_tens(l_in, ab_pred)
            # plt.imshow(rgb_pred)
            # plt.show()
            