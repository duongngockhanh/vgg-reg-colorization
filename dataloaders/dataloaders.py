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
        return len(self.file_list)
    
    def __getitem__(self, index):
        img_rgb = load_img(os.path.join(self.data_root, self.file_list[index]))
        _, _, tens_rs_l, tens_rs_ab = preprocess_img(img_rgb)
        return tens_rs_l, tens_rs_ab
    
def create_dataloader(data_root, batch_size=16, shuffle=False):
    dataset = DatasetColor(data_root)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader