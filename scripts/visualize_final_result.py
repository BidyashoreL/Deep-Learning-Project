import rasterio
import matplotlib.pyplot as plt
import numpy as np
import os

BASE_DIR = "/Users/bidyashorelourembam/Deep learning_ Project"

rain_path   = os.path.join(BASE_DIR, "vj_rainfall_dynamic.tif")
lulc_path   = os.path.join(BASE_DIR, "vijayawada_lulc.tif")
drain_path  = os.path.join(BASE_DIR, "vj_drainage_distance.tif")
flood_path  = os.path.join(BASE_DIR, "vj_final_prediction.tif")

def load_raster(path):
    with rasterio.open(path) as src:
        return src.read(1)

rain  = load_raster(rain_path)
lulc  = load_raster(lulc_path)
drain = load_raster(drain_path)
flood = load_raster(flood_path)

titles = [
    "Dynamic Rainfall",
    "Built-up (NDBI)",
    "Distance to Drainage",
    "Predicted Flood Susceptibility"
]

cmaps = [
    "coolwarm",
    "gray",
    "Blues_r",
    "RdYlBu_r"
]

layers = [rain, lulc, drain, flood]

plt.figure(figsize=(16, 10))

for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(layers[i], cmap=cmaps[i])
    plt.title(titles[i])
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)

plt.suptitle("Urban Flood Prediction – Multi-Source Deep Learning Result", fontsize=16)

plt.tight_layout()
plt.show()