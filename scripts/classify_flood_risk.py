import rasterio
import numpy as np
import os

BASE_DIR = "/Users/bidyashorelourembam/Deep learning_ Project"
flood_path = os.path.join(BASE_DIR, "vj_final_prediction.tif")
out_path   = os.path.join(BASE_DIR, "vj_flood_risk_classes.tif")

with rasterio.open(flood_path) as src:
    flood = src.read(1)
    meta = src.meta.copy()

# ---------------- CLASSIFICATION ----------------
low = flood <= 0.4
moderate = (flood > 0.4) & (flood <= 0.7)
high = flood > 0.7

risk = np.zeros_like(flood, dtype="uint8")
risk[low] = 1
risk[moderate] = 2
risk[high] = 3

# ---------------- FIX METADATA ----------------
meta.update({
    "dtype": "uint8",
    "count": 1,
    "nodata": 0   # valid for uint8
})

with rasterio.open(flood_path) as src:
    flood = src.read(1)
    meta = src.meta.copy()

# DATA-DRIVEN THRESHOLDS
p33 = np.percentile(flood, 33)
p66 = np.percentile(flood, 66)

low = flood <= p33
moderate = (flood > p33) & (flood <= p66)
high = flood > p66

risk = np.zeros_like(flood, dtype="uint8")
risk[low] = 1
risk[moderate] = 2
risk[high] = 3

meta.update(dtype="uint8", count=1, nodata=0)

with rasterio.open(out_path, "w", **meta) as dst:
    dst.write(risk, 1)

print("✅ Flood risk classification saved")
print("Thresholds:", p33, p66)