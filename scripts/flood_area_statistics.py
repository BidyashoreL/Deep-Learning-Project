import rasterio
import numpy as np
import os

BASE_DIR = "/Users/bidyashorelourembam/Deep learning_ Project"
risk_path = os.path.join(BASE_DIR, "vj_flood_risk_classes.tif")

with rasterio.open(risk_path) as src:
    risk = src.read(1)

pixel_area_km2 = (30 * 30) / 1_000_000   # 30m resolution

low_area = np.sum(risk == 1) * pixel_area_km2
mod_area = np.sum(risk == 2) * pixel_area_km2
high_area = np.sum(risk == 3) * pixel_area_km2
total = risk.size * pixel_area_km2

print("🌍 FLOOD RISK AREA STATISTICS")
print(f"Low Risk:      {low_area:.2f} km²")
print(f"Moderate Risk: {mod_area:.2f} km²")
print(f"High Risk:     {high_area:.2f} km²")
print(f"Total Area:    {total:.2f} km²")