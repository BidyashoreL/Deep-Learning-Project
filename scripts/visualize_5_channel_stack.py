import numpy as np
import matplotlib.pyplot as plt
import os

BASE_DIR = "/Users/bidyashorelourembam/Deep learning_ Project"
data_path = os.path.join(BASE_DIR, "vj_cnn_input.npy")

stack = np.load(data_path)

print("Stack shape:", stack.shape)

titles = [
    "Elevation (DEM)",
    "Slope",
    "Built-up (NDBI)",
    "Distance to Drainage",
    "Rainfall (Dynamic)"
]

cmaps = ["terrain", "magma", "gray", "Blues_r", "coolwarm"]

plt.figure(figsize=(18, 10))

for i in range(stack.shape[2]):
    plt.subplot(2, 3, i + 1)
    plt.imshow(stack[:, :, i], cmap=cmaps[i])
    plt.title(titles[i])
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()