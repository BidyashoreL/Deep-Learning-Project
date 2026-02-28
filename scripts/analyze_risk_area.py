import rasterio
import numpy as np
import os

BASE_DIR = "/Users/bidyashorelourembam/Deep learning_ Project"
prediction_path = os.path.join(BASE_DIR, "vj_final_prediction.tif")

with rasterio.open(prediction_path) as src:
    risk_map = src.read(1)
    # Forced Resolution: 30m x 30m is the standard for your data sources
    pixel_area_km2 = (30 * 30) / 1_000_000 

# Statistics
high_risk_pixels = np.sum(risk_map > 0.8)
moderate_risk_pixels = np.sum((risk_map > 0.4) & (risk_map <= 0.8))
safe_pixels = np.sum(risk_map <= 0.4)

total_pixels = risk_map.size

print("--- FINAL VIJAYAWADA RESEARCH STATS ---")
print(f"🔴 High Risk Area:  {high_risk_pixels * pixel_area_km2:.2f} sq. km ({ (high_risk_pixels/total_pixels)*100 :.1f}%)")
print(f"🟡 Moderate Risk:   {moderate_risk_pixels * pixel_area_km2:.2f} sq. km ({ (moderate_risk_pixels/total_pixels)*100 :.1f}%)")
print(f"🔵 Safe Zones:      {safe_pixels * pixel_area_km2:.2f} sq. km ({ (safe_pixels/total_pixels)*100 :.1f}%)")