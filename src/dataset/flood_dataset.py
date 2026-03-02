import torch
from torch.utils.data import Dataset
import numpy as np


class FloodDataset(Dataset):

    def __init__(self, image_stack, mask):

        self.image = np.load(image_stack)   # shape → (H, W, 5)
        self.mask = np.load(mask)

    def __len__(self):
        return 1

    def __getitem__(self, idx):

        x = torch.tensor(self.image).permute(2, 0, 1).float()
        y = torch.tensor(self.mask).unsqueeze(0).float()

        return x, y