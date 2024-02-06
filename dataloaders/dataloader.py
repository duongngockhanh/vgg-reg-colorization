import os
from colorizers import *
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class DatasetColor(Dataset):
    def __init__(self, data_root):
        super().__init__()
        self.data_root = data_root
        self.file_list = sorted(os.listdir(self.data_root))

    def __len__(self):
        return len(self.data_root)
    
    def __getitem__(self, index):
        img_rgb = load_img(os.path.join(self.data_root, self.file_list[index]))
        _, _, tens_rs_l, tens_rs_ab = preprocess_img(img_rgb)
        return normalize_l(tens_rs_l), normalize_ab(tens_rs_ab)

def normalize_l(tens_rs_l):
    return (tens_rs_l - 50) / 100

def denormalize_l(tens_rs_l):
    return tens_rs_l * 100 + 50

def normalize_ab(tens_rs_ab):
    return tens_rs_ab / 256

def denormalize_ab(tens_rs_ab):
    return tens_rs_ab * 255

def create_dataloader(data_root, batch_size=16, shuffle=False):
    dataset = DatasetColor(data_root)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader