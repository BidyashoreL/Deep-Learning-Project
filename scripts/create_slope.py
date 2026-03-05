import rasterio
import numpy as np

with rasterio.open("data/vijayawada_dem.tif") as src:
    dem = src.read(1).astype("float32")
    meta = src.meta

dy, dx = np.gradient(dem)
slope = np.sqrt(dx**2 + dy**2)

meta.update(dtype="float32", count=1)

with rasterio.open("data/slope.tif", "w", **meta) as dst:
    dst.write(slope, 1)

print("✅ slope.tif created")