import rasterio
import numpy as np
import torch
from torch.utils.data import Dataset

PATCH = 256
STRIDE = 64

class FloodDataset(Dataset):
    def __init__(self, raster_paths, label_path):

        self.rasters = [rasterio.open(p) for p in raster_paths]
        self.label = rasterio.open(label_path)

        self.H = self.label.height
        self.W = self.label.width

        self.indices = []

        for y in range(0, self.H - PATCH + 1, STRIDE):
            for x in range(0, self.W - PATCH + 1, STRIDE):
                self.indices.append((x, y))

        print(f"✅ Dataset initialized with {len(self.indices)} patches")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        x_start, y_start = self.indices[idx]
        channels = []

        for raster in self.rasters:
            patch = raster.read(
                1,
                window=((y_start, y_start + PATCH),
                        (x_start, x_start + PATCH))
            ).astype(np.float32)

            # 🔥 SAME normalization used everywhere
            if patch.max() != patch.min():
                patch = (patch - patch.min()) / (patch.max() - patch.min())

            channels.append(patch)

        image = np.stack(channels)
        label = self.label.read(
            1,
            window=((y_start, y_start + PATCH),
                    (x_start, x_start + PATCH))
        ).astype(np.int64)

        return torch.tensor(image), torch.tensor(label)