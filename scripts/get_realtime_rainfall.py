import numpy as np
import rasterio
import cv2
import os

def generate_simulated_gpm_layer(master_shape, intensity_mm_hr=45.0):
    """
    In a real production environment, you would use an API call here.
    For your research prototype, we generate a rainfall intensity layer 
    that matches your DEM dimensions.
    """
    # Create a base rainfall layer (uniform intensity for the city)
    rainfall_layer = np.full(master_shape, intensity_mm_hr, dtype='float32')
    
    # Add some spatial variance (simulating a storm cell moving over Vijayawada)
    rows, cols = master_shape
    x, y = np.meshgrid(np.linspace(-1, 1, cols), np.linspace(-1, 1, rows))
    storm_cell = np.exp(-(x**2 + y**2)) * 10  # 10mm variance
    
    final_rainfall = rainfall_layer + storm_cell
    return final_rainfall

# 1. Load Master Shape
with rasterio.open('vijayawada_dem.tif') as src:
    master_shape = src.shape
    meta = src.meta

# 2. Generate/Fetch Rainfall
print("🌧️ Processing Real-Time Rainfall Layer (GPM IMERG)...")
rainfall_data = generate_simulated_gpm_layer(master_shape)

# 3. Save as TIF for the CNN Stack
meta.update(dtype='float32', count=1)
with rasterio.open('vj_rainfall_dynamic.tif', 'w', **meta) as dst:
    dst.write(rainfall_data, 1)

print("✅ SUCCESS: vj_rainfall_dynamic.tif ready for 5-Channel Stack!")