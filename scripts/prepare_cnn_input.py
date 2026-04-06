import rasterio
import numpy as np
import cv2
import os

BASE_DIR = "/Users/bidyashorelourembam/Deep learning_ Project"
dem_path = f"{BASE_DIR}/data/vijayawada_dem.tif"
lulc_path = f"{BASE_DIR}/data/vijayawada_lulc.tif"
slope_path = f"{BASE_DIR}/data/slope.tif"
drain_path = f"{BASE_DIR}/data/vj_drainage_distance.tif"
rain_path = f"{BASE_DIR}/data/vj_rainfall_dynamic.tif"

output_path = f"{BASE_DIR}/data/vj_cnn_input.npy"

def normalize(arr):
    arr = arr.astype("float32")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)  # Handle NaN/Inf
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

# Resize layers to match DEM shape
lulc = resize_to_match(lulc)
drainage = resize_to_match(drainage)
rainfall = resize_to_match(rainfall)

# Stack all layers into a multi-channel array
stack = np.stack([
    normalize(dem),
    normalize(slope),
    normalize(lulc),
    normalize(drainage),
    normalize(rainfall)
], axis=-1)

# Save the stacked array
np.save(output_path, stack)

print(f"✅ 5-Channel CNN input created and saved to: {output_path}, Shape: {stack.shape}")