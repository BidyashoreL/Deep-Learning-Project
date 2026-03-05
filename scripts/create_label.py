import rasterio
import numpy as np

src_path = "outputs/vj_flood_risk_classes.tif"
dst_path = "data/label.tif"

with rasterio.open(src_path) as src:
    label = src.read(1)
    meta = src.meta

# convert 1,2,3 → 0,1,2
label = label - 1

meta.update(dtype="uint8", count=1)

with rasterio.open(dst_path, "w", **meta) as dst:
    dst.write(label.astype("uint8"), 1)

print("✅ label.tif created")