import rasterio
import numpy as np
import cv2
import os

BASE_DIR = "/Users/bidyashorelourembam/Deep learning_ Project"

dem_path = os.path.join(BASE_DIR, "vijayawada_dem.tif")
lulc_path = os.path.join(BASE_DIR, "vijayawada_lulc.tif")
drain_path = os.path.join(BASE_DIR, "vj_drainage_distance.tif")
rain_path = os.path.join(BASE_DIR, "vj_rainfall_dynamic.tif")

save_path = os.path.join(BASE_DIR, "vj_cnn_input.npy")


def normalize(arr):
    arr = arr.astype("float32")
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)


# DEM
with rasterio.open(dem_path) as src:
    dem = src.read(1)
    master_shape = dem.shape

# SLOPE
dy, dx = np.gradient(dem)
slope = np.sqrt(dx**2 + dy**2)

# LULC
with rasterio.open(lulc_path) as src:
    lulc = src.read(1)

# DRAINAGE DISTANCE
with rasterio.open(drain_path) as src:
    drainage = src.read(1)

# RAINFALL
with rasterio.open(rain_path) as src:
    rainfall = src.read(1)


def resize_to_match(layer):
    if layer.shape != master_shape:
        return cv2.resize(layer, (master_shape[1], master_shape[0]))
    return layer


lulc = resize_to_match(lulc)
drainage = resize_to_match(drainage)
rainfall = resize_to_match(rainfall)

stack = np.stack([
    normalize(dem),
    normalize(slope),
    normalize(lulc),
    normalize(drainage),
    normalize(rainfall)
], axis=-1)

np.save(save_path, stack)

print("✅ 5-Channel CNN input created:", stack.shape)